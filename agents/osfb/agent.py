"""One-Step Forward-Backward Agent (PyTorch).

Default behavior matches the JAX reference (the source paper):
    - Squared MSE TD loss on the off-diagonal, normalized by (batch_size - 1).
    - No F/B normalization, no M clamp, no Q clamp, no grad clamp.

Set loss_type="huber" + stabilize=True if training diverges in your setting.
"""

import math
from pathlib import Path
from typing import Tuple, Dict, Optional

import torch
import numpy as np

from agents.base import AbstractAgent, Batch
from agents.fb.base import ActorModel
from agents.osfb.models import OneStepForwardBackwardRepresentation
from agents.utils import schedule


class OneStepFB(AbstractAgent):
    """One-step FB agent: F(s, a) + B(s, a) + actor, trained jointly."""

    def __init__(
        self,
        observation_length: int,
        action_length: int,
        preprocessor_hidden_dimension: int,
        preprocessor_output_dimension: int,
        preprocessor_hidden_layers: int,
        preprocessor_activation: str,
        z_dimension: int,
        forward_hidden_dimension: int,
        forward_hidden_layers: int,
        forward_number_of_features: int,
        backward_hidden_dimension: int,
        backward_hidden_layers: int,
        actor_hidden_dimension: int,
        actor_hidden_layers: int,
        forward_activation: str,
        backward_activation: str,
        actor_activation: str,
        actor_learning_rate: float,
        critic_learning_rate: float,
        learning_rate_coefficient: float,
        orthonormalisation_coefficient: float,
        discount: float,
        batch_size: int,
        z_mix_ratio: float,
        gaussian_actor: bool,
        std_dev_clip: float,
        std_dev_schedule: str,
        tau: float,
        device: torch.device,
        name: str,
        # ---- one-step FB specific ----
        repr_agg: str = "mean",
        q_agg: str = "min",
        alpha: float = 1.0,
        normalize_q_loss: bool = True,
        const_std: bool = True,
        # ---- loss flavour ----
        loss_type: str = "huber",   # "mse" matches paper, "huber" is the stabilized variant
        stabilize: bool =  False,  # if True, also normalize F/B and clamp M, Q, grads
    ):
        super().__init__(
            observation_length=observation_length,
            action_length=action_length,
            name=name,
        )

        # FB nets (forward + backward, online + target).
        self.FB = OneStepForwardBackwardRepresentation(
            observation_length=observation_length,
            action_length=action_length,
            preprocessor_hidden_dimension=preprocessor_hidden_dimension,
            preprocessor_feature_space_dimension=preprocessor_output_dimension,
            preprocessor_hidden_layers=preprocessor_hidden_layers,
            preprocessor_activation=preprocessor_activation,
            number_of_features=forward_number_of_features,
            z_dimension=z_dimension,
            forward_hidden_dimension=forward_hidden_dimension,
            forward_hidden_layers=forward_hidden_layers,
            backward_hidden_dimension=backward_hidden_dimension,
            backward_hidden_layers=backward_hidden_layers,
            forward_activation=forward_activation,
            backward_activation=backward_activation,
            orthonormalisation_coefficient=orthonormalisation_coefficient,
            discount=discount,
            device=device,
        )

        # Policy: pi(a | s, z).
        self.actor = ActorModel(
            observation_length=observation_length,
            action_length=action_length,
            preprocessor_hidden_dimension=preprocessor_hidden_dimension,
            preprocessor_feature_space_dimension=preprocessor_output_dimension,
            preprocessor_hidden_layers=preprocessor_hidden_layers,
            preprocessor_activation=preprocessor_activation,
            z_dimension=z_dimension,
            number_of_features=forward_number_of_features,
            actor_hidden_dimension=actor_hidden_dimension,
            actor_hidden_layers=actor_hidden_layers,
            actor_activation=actor_activation,
            gaussian_actor=gaussian_actor,
            std_dev_clip=std_dev_clip,
            device=device,
        )

        self.encoder = torch.nn.Identity()
        self.augmentation = torch.nn.Identity()

        # Initialize target nets to match online nets.
        self.FB.forward_representation_target.load_state_dict(
            self.FB.forward_representation.state_dict()
        )
        self.FB.backward_representation_target.load_state_dict(
            self.FB.backward_representation.state_dict()
        )

        # Single Adam optimizer (paper: optax.adam(lr) on the whole network).
        # Per-group LRs preserved for parity with the original PyTorch FB code.
        self.combined_optimizer = torch.optim.Adam(
            [
                {"params": self.FB.forward_representation.parameters()},
                {
                    "params": self.FB.backward_representation.parameters(),
                    "lr": critic_learning_rate * learning_rate_coefficient,
                },
                {"params": self.actor.parameters(), "lr": actor_learning_rate},
            ],
            lr=critic_learning_rate,
        )

        # Stash config.
        self._device = device
        self.batch_size = batch_size
        self._z_mix_ratio = z_mix_ratio
        self._tau = tau
        self._z_dimension = z_dimension
        self._discount = discount
        self.std_dev_schedule = std_dev_schedule
        self._repr_agg = repr_agg
        self._q_agg = q_agg
        self._alpha = alpha
        self._normalize_q_loss = normalize_q_loss
        self._const_std = const_std

        assert loss_type in ("mse", "huber"), f"loss_type must be 'mse' or 'huber', got {loss_type}"
        self._loss_type = loss_type
        self._stabilize = stabilize

    # =========================================================================
    # Inference
    # =========================================================================
    @torch.no_grad()
    def act(self, observation, task, step, sample=False):
        """Pick an action for one obs given task z."""
        observation = torch.as_tensor(
            observation, dtype=torch.float32, device=self._device
        ).unsqueeze(0)
        h = self.encoder(observation)
        z = torch.as_tensor(task, dtype=torch.float32, device=self._device).unsqueeze(0)
        std_dev = schedule(self.std_dev_schedule, step)
        action, _ = self.actor(h, z, std_dev, sample=sample)
        return action.detach().cpu().numpy()[0], std_dev

    # =========================================================================
    # Training
    # =========================================================================
    def update(self, batch: Batch, step: int) -> Dict[str, float]:
        """One gradient step on a replay batch."""
        # Sample latents (gaussian sphere, mixed with B(s, a) of permuted batch).
        zs = self._sample_latents(batch.observations, batch.actions)

        total_loss, metrics = self._compute_total_loss(
            observations=batch.observations,
            actions=batch.actions,
            next_observations=batch.next_observations,
            next_actions=batch.next_actions,
            zs=zs,
            step=step,
        )

        self.combined_optimizer.zero_grad(set_to_none=True)
        total_loss.backward()
        # Grad clamp is NOT in the paper. Only enable when stabilize=True.
        if self._stabilize:
            for p in self.FB.parameters():
                if p.grad is not None:
                    p.grad.data.clamp_(-1, 1)
            for p in self.actor.parameters():
                if p.grad is not None:
                    p.grad.data.clamp_(-1, 1)
        self.combined_optimizer.step()

        # Polyak update both target nets.
        self.soft_update_params(
            self.FB.forward_representation,
            self.FB.forward_representation_target,
            self._tau,
        )
        self.soft_update_params(
            self.FB.backward_representation,
            self.FB.backward_representation_target,
            self._tau,
        )
        return metrics

    def _sample_latents(self, observations: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        """Mirror JAX sample_latents: gaussian z's, some replaced by B(s, a) of perm batch."""
        B = observations.shape[0]

        # Gaussian z's projected onto the sphere of radius sqrt(D).
        zs = torch.randn(B, self._z_dimension, device=self._device)
        zs = math.sqrt(self._z_dimension) * torch.nn.functional.normalize(zs, dim=1)

        # B-embeddings of a permuted (obs, act) batch, also on the sphere.
        perm = torch.randperm(B, device=self._device)
        with torch.no_grad():
            b_zs = self.FB.backward_representation(
                observations[perm], actions[perm]
            ).detach()
            b_zs = math.sqrt(self._z_dimension) * torch.nn.functional.normalize(b_zs, dim=1)

        # With prob `_z_mix_ratio`, replace the gaussian z with a B-embedding.
        mix = torch.rand(B, 1, device=self._device) < self._z_mix_ratio
        zs = torch.where(mix, b_zs, zs)
        return zs

    def _compute_total_loss(
        self,
        observations: torch.Tensor,
        actions: torch.Tensor,
        next_observations: torch.Tensor,
        next_actions: torch.Tensor,
        zs: torch.Tensor,
        step: int,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """FB repr loss + ortho loss + actor (Q + BC) loss. Matches the JAX paper."""

        N = observations.shape[0]

        # ---------- Target successor measure (no grad) ----------
        with torch.no_grad():
            # F target uses next_actions from the batch (one-step FB).
            target_F1, target_F2 = self.FB.forward_representation_target(
                observation=next_observations, action=next_actions
            )
            target_B = self.FB.backward_representation_target(observations, actions)

            # Optional stabilization: keep M bounded by normalizing F and B.
            if self._stabilize:
                target_F1 = torch.nn.functional.normalize(target_F1, dim=-1)
                target_F2 = torch.nn.functional.normalize(target_F2, dim=-1)
                target_B = torch.nn.functional.normalize(target_B, dim=-1)

            # M[i, j] = <F(s_i'), B(s_j)>
            target_M1 = torch.einsum("sd, td -> st", target_F1, target_B)
            target_M2 = torch.einsum("sd, td -> st", target_F2, target_B)

            # Aggregate the two F heads.
            if self._repr_agg == "mean":
                target_M = 0.5 * (target_M1 + target_M2)
            else:
                target_M = torch.min(target_M1, target_M2)

        # ---------- Online F, B and successor measure ----------
        F1, F2 = self.FB.forward_representation(observation=observations, action=actions)
        B = self.FB.backward_representation(observations, actions)

        if self._stabilize:
            F1m = torch.nn.functional.normalize(F1, dim=-1)
            F2m = torch.nn.functional.normalize(F2, dim=-1)
            Bm = torch.nn.functional.normalize(B, dim=-1)
        else:
            F1m, F2m, Bm = F1, F2, B

        M1 = torch.einsum("sd, td -> st", F1m, Bm)
        M2 = torch.einsum("sd, td -> st", F2m, Bm)
        if self._stabilize:
            M1 = M1.clamp(-2.0, 2.0)
            M2 = M2.clamp(-2.0, 2.0)

        I = torch.eye(N, device=self._device)

        # ---------- TD LSIF off-diagonal loss ----------
        # JAX (paper): squared diff, sum over j, divide by (N - 1), mean over i.
        # We replicate that exactly when loss_type == "mse".
        if self._loss_type == "mse":
            fb_off_diag_loss = 0.0
            for M in [M1, M2]:
                diff = (M - self._discount * target_M) * (1 - I)
                fb_off_diag_loss = fb_off_diag_loss + (diff.pow(2).sum(dim=-1) / (N - 1)).mean()
            fb_off_diag_loss = 0.5 * fb_off_diag_loss
        else:  # "huber" — the stabilized variant
            fb_off_diag_loss = 0.5 * sum(
                torch.nn.functional.smooth_l1_loss(
                    M * (1 - I),
                    self._discount * target_M * (1 - I),
                    reduction="mean",
                    beta=1.0,
                )
                for M in [M1, M2]
            )

        # ---------- Diagonal loss: -(1 - gamma) * mean(diag) ----------
        fb_diag_loss = -(1 - self._discount) * sum(M.diag().mean() for M in [M1, M2])

        repr_loss = fb_diag_loss + fb_off_diag_loss

        # ---------- Orthonormalization regularizer on B ----------
        # JAX uses raw B; we follow the same.
        B_for_ortho = Bm if self._stabilize else B
        covariance = torch.matmul(B_for_ortho, B_for_ortho.T)
        ortho_diag_loss = -covariance.diag().mean()
        ortho_off_diag_loss = 0.5 * (
            (covariance * (1 - I)).pow(2).sum(dim=-1) / (N - 1)
        ).mean()
        ortho_loss = ortho_diag_loss + ortho_off_diag_loss

        # JAX: fb_loss = repr_loss + orthonorm_coeff * ortho_loss
        fb_loss = repr_loss + self.FB.orthonormalisation_coefficient * ortho_loss

        # ---------- Actor loss (RPG + BC) ----------
        std_dev = schedule(self.std_dev_schedule, step)
        q_actions, action_dist = self.actor(
            observations, zs, std_dev, sample=not self._const_std
        )
        q_actions = q_actions.clamp(-1, 1)

        # Q(s, a, z) = <F(s, a), z>. Note: F has no z input.
        Q_F1, Q_F2 = self.FB.forward_representation(
            observation=observations, action=q_actions
        )
        Q1 = torch.einsum("sd, sd -> s", Q_F1, zs)
        Q2 = torch.einsum("sd, sd -> s", Q_F2, zs)
        Q = 0.5 * (Q1 + Q2) if self._q_agg == "mean" else torch.min(Q1, Q2)
        if self._stabilize:
            Q = Q.clamp(-100.0, 100.0)

        q_loss = -Q.mean()
        if self._normalize_q_loss:
            # JAX: lam = 1 / |Q|.mean().
            denom = Q.abs().mean()
            if self._stabilize:
                denom = denom + 1e-6
            lam = (1.0 / denom).detach()
            q_loss = lam * q_loss

        if action_dist is not None:
            log_prob = action_dist.log_prob(actions).sum(-1)
            bc_loss = -log_prob.mean()
            mean_log_prob = log_prob.mean().item()
        else:
            bc_loss = torch.tensor(0.0, device=self._device)
            mean_log_prob = 0.0

        actor_loss = q_loss + self._alpha * bc_loss

        # ---------- Total ----------
        total_loss = fb_loss + actor_loss

        # NaN safety only when stabilizing.
        if self._stabilize and (torch.isnan(total_loss) or total_loss.abs() > 1e6):
            print(f"[WARNING] Loss exploded: {total_loss.item():.2e}")
            total_loss = torch.tensor(0.0, device=self._device, requires_grad=True)

        metrics = {
            "train/total_loss": total_loss.item(),
            "train/fb_loss": fb_loss.item(),
            "train/repr_loss": repr_loss.item(),
            "train/repr_diag_loss": fb_diag_loss.item(),
            "train/repr_off_diag_loss": fb_off_diag_loss.item(),
            "train/ortho_loss": ortho_loss.item(),
            "train/ortho_diag_loss": ortho_diag_loss.item(),
            "train/ortho_off_diag_loss": ortho_off_diag_loss.item(),
            "train/succ_measure_mean": M1.mean().item(),
            "train/succ_measure_max": M1.max().item(),
            "train/succ_measure_min": M1.min().item(),
            "train/F_norm": F1.norm(dim=-1).mean().item(),
            "train/B_norm": B.norm(dim=-1).mean().item(),
            "train/target_M": target_M.mean().item(),
            "train/actor_loss": actor_loss.item(),
            "train/q_loss": q_loss.item(),
            "train/bc_loss": bc_loss.item(),
            "train/q_mean": Q.mean().item(),
            "train/q_abs_mean": Q.abs().mean().item(),
            "train/bc_log_prob": mean_log_prob,
        }
        return total_loss, metrics

    # =========================================================================
    # z utilities
    # =========================================================================
    def sample_z(self, size: int) -> torch.Tensor:
        """Sample z's uniformly on a sphere of radius sqrt(z_dim)."""
        g = torch.randn(size, self._z_dimension, dtype=torch.float32, device=self._device)
        g = torch.nn.functional.normalize(g, dim=1)
        return math.sqrt(self._z_dimension) * g

    def infer_z(
        self,
        observations: torch.Tensor,
        actions: torch.Tensor,
        rewards: Optional[torch.Tensor] = None,
        reward_temperature: float = 10.0,
    ) -> torch.Tensor:
        """Infer task z from a few (s, a, r) samples. Matches JAX infer_latent."""
        with torch.no_grad():
            z = self.FB.backward_representation(observations, actions)
        if rewards is not None:
            weights = torch.softmax(reward_temperature * rewards, dim=0)
            z = (weights * rewards * z).mean(dim=0, keepdim=True)
        # Project to sphere of radius sqrt(z_dim).
        z = math.sqrt(self._z_dimension) * torch.nn.functional.normalize(z, dim=1)
        return z.squeeze().cpu().numpy()

    # =========================================================================
    # Q prediction
    # =========================================================================
    def predict_q(
        self, observation: torch.Tensor, z: torch.Tensor, action: torch.Tensor
    ) -> torch.Tensor:
        """Q(s, a, z) = min(<F1(s, a), z>, <F2(s, a), z>)."""
        F1, F2 = self.FB.forward_representation(observation=observation, action=action)
        Q1 = torch.einsum("sd, sd -> s", F1, z)
        Q2 = torch.einsum("sd, sd -> s", F2, z)
        return torch.min(Q1, Q2)

    @staticmethod
    def soft_update_params(network, target_network, tau):
        """Polyak: target <- tau * online + (1 - tau) * target."""
        for p, tp in zip(network.parameters(), target_network.parameters()):
            tp.data.copy_(tau * p.data + (1 - tau) * tp.data)

    def load(self, filepath: Path):
        """Placeholder — implement checkpoint loading to suit your format."""
        pass
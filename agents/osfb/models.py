"""Models for the One-Step Forward-Backward agent.

Key change vs original FB:
    - Forward F(s, a) — NO z input, so only ONE preprocessor (obs_action).
    - Backward B(s, a) — action-conditioned.
"""
import math
from typing import Tuple
import torch

from agents.fb.base import (
    BackwardModel,
    ForwardModel,
    AbstractPreprocessor,
)


# =============================================================================
# Forward representation WITHOUT z
# =============================================================================
class OneStepForwardRepresentation(torch.nn.Module):
    """Forward net F(s, a) -> (F1, F2). Two heads for double-Q style training."""

    def __init__(
        self,
        observation_length: int,
        action_length: int,
        preprocessor_hidden_dimension: int,
        preprocessor_feature_space_dimension: int,
        preprocessor_hidden_layers: int,
        preprocessor_activation: str,
        number_of_features: int,  # kept for API parity; F heads use 1 here
        z_dimension: int,
        forward_hidden_dimension: int,
        forward_hidden_layers: int,
        device: torch.device,
        forward_activation: str,
    ):
        super().__init__()
        self._zdim=z_dimension

        # Single preprocessor for (obs, action). NO obs_z preprocessor.
        # Original FB API: pass obs and the extra concatenated var as separate sizes.
        self.obs_action_preprocessor = AbstractPreprocessor(
            observation_length=observation_length,
            concatenated_variable_length=action_length,
            hidden_dimension=preprocessor_hidden_dimension,
            feature_space_dimension=preprocessor_feature_space_dimension,
            hidden_layers=preprocessor_hidden_layers,
            device=device,
            activation=preprocessor_activation,
        )

        # Important: original FB feeds ForwardModel with TWO preprocessor outputs
        # concatenated (obs_action + obs_z), so it sets number_of_preprocessed_features=2.
        # One-step FB has ONLY ONE preprocessor (obs_action), so this must be 1.
        self.F1 = ForwardModel(
            preprocessor_feature_space_dimension=preprocessor_feature_space_dimension,
            number_of_preprocessed_features=1,
            z_dimension=z_dimension,
            hidden_dimension=forward_hidden_dimension,
            hidden_layers=forward_hidden_layers,
            device=device,
            activation=forward_activation,
            layernorm=True,  # keep layernorm in F1 head for stability (original FB ablation showed it helps a lot
        )
        self.F2 = ForwardModel(
            preprocessor_feature_space_dimension=preprocessor_feature_space_dimension,
            number_of_preprocessed_features=1,
            z_dimension=z_dimension,
            hidden_dimension=forward_hidden_dimension,
            hidden_layers=forward_hidden_layers,
            device=device,
            activation=forward_activation,
            layernorm=True,
        )

    def forward(
        self,
        observation: torch.Tensor,
        action: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.obs_action_preprocessor(
            torch.cat([observation, action], dim=-1)
        )
        f1 = self.F1(h)
        f2 = self.F2(h)
        f1 = math.sqrt(self._zdim) * torch.nn.functional.normalize(f1, dim=-1)
        f2 = math.sqrt(self._zdim) * torch.nn.functional.normalize(f2, dim=-1)
        return f1, f2


# =============================================================================
# Backward representation WITH action
# =============================================================================
class ActionConditionedBackwardRepresentation(torch.nn.Module):
    """Backward net B(s, a) — concatenates [obs, action] before the MLP."""

    def __init__(
        self,
        observation_length: int,
        action_length: int,
        z_dimension: int,
        backward_hidden_dimension: int,
        backward_hidden_layers: int,
        device: torch.device,
        backward_activation: str,
    ):
        super().__init__()

        # BackwardModel internally L2-normalises and scales output to sqrt(z_dim).
        # We widen its input to accept the concatenated [obs, action].
        self.B = BackwardModel(
            observation_length=observation_length + action_length,
            z_dimension=z_dimension,
            hidden_dimension=backward_hidden_dimension,
            hidden_layers=backward_hidden_layers,
            device=device,
            activation=backward_activation,
            layernorm=True,  # keep layernorm in B for stability (original FB ablation showed it helps a lot
        )

    def forward(
        self,
        observation: torch.Tensor,
        action: torch.Tensor,
    ) -> torch.Tensor:
        """Run B(s, a). Returns [B, z_dim] embedding."""
        return self.B(torch.cat([observation, action], dim=-1))


# =============================================================================
# Combined FB representation for one-step FB
# =============================================================================
class OneStepForwardBackwardRepresentation(torch.nn.Module):
    """Bundle of all forward/backward nets (online + target) for one-step FB."""

    def __init__(
        self,
        observation_length: int,
        action_length: int,
        preprocessor_hidden_dimension: int,
        preprocessor_feature_space_dimension: int,
        preprocessor_hidden_layers: int,
        preprocessor_activation: str,
        number_of_features: int,
        z_dimension: int,
        forward_hidden_dimension: int,
        forward_hidden_layers: int,
        backward_hidden_dimension: int,
        backward_hidden_layers: int,
        forward_activation: str,
        backward_activation: str,
        orthonormalisation_coefficient: float,
        discount: float,
        device: torch.device,
    ):
        super().__init__()

        # ----- Forward nets: online + target (both F(s, a), no z) -----
        forward_kwargs = dict(
            observation_length=observation_length,
            action_length=action_length,
            preprocessor_hidden_dimension=preprocessor_hidden_dimension,
            preprocessor_feature_space_dimension=preprocessor_feature_space_dimension,
            preprocessor_hidden_layers=preprocessor_hidden_layers,
            preprocessor_activation=preprocessor_activation,
            number_of_features=number_of_features,
            z_dimension=z_dimension,
            forward_hidden_dimension=forward_hidden_dimension,
            forward_hidden_layers=forward_hidden_layers,
            device=device,
            forward_activation=forward_activation,
        )
        self.forward_representation = OneStepForwardRepresentation(**forward_kwargs)
        self.forward_representation_target = OneStepForwardRepresentation(**forward_kwargs)

        # ----- Backward nets: online + target (both B(s, a)) -----
        backward_kwargs = dict(
            observation_length=observation_length,
            action_length=action_length,
            z_dimension=z_dimension,
            backward_hidden_dimension=backward_hidden_dimension,
            backward_hidden_layers=backward_hidden_layers,
            device=device,
            backward_activation=backward_activation,
        )
        self.backward_representation = ActionConditionedBackwardRepresentation(**backward_kwargs)
        self.backward_representation_target = ActionConditionedBackwardRepresentation(**backward_kwargs)

        # Stash hyperparams used by the trainer.
        self._discount = discount
        self.orthonormalisation_coefficient = orthonormalisation_coefficient
        self._device = device
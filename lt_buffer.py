import os, glob
import numpy as np
import torch
from data_loader import load_light_transport_npz

class LightTransportReplayBuffer:
    def __init__(self, root_dir, device, reward_mode="luminance", norm_out="norm_stats.npz"):
        self.device = device
        pattern = os.path.join(root_dir, "rl_reward_light*_batch_*.npz")
        file_list = sorted(glob.glob(pattern))
        if not file_list:
            raise FileNotFoundError(f"No LT dataset files found in {root_dir}")

        all_states, all_actions, all_rewards, all_next = [], [], [], []
        for file_path in file_list:
            S, A, R, S_next = load_light_transport_npz(file_path)
            all_states.append(S); all_actions.append(A); all_rewards.append(R); all_next.append(S_next)

        states = np.concatenate(all_states, axis=0)
        actions = np.concatenate(all_actions, axis=0)
        rewards = np.concatenate(all_rewards, axis=0)
        next_states = np.concatenate(all_next, axis=0)

        state_mean = states.mean(axis=0, keepdims=True)
        state_std  = states.std(axis=0, keepdims=True) + 1e-6
        action_mean = actions.mean(axis=0, keepdims=True)
        action_std  = actions.std(axis=0, keepdims=True) + 1e-6

        np.savez(norm_out,
                 state_mean=state_mean, state_std=state_std,
                 action_mean=action_mean, action_std=action_std)
        print(f"[Saved] normalization stats -> {norm_out}", flush=True)

        # normalize
        states = (states - state_mean) / state_std
        next_states = (next_states - state_mean) / state_std
        actions = (actions - action_mean) / action_std

        if reward_mode == "luminance":
            rewards = (0.2126*rewards[:,0] + 0.7152*rewards[:,1] + 0.0722*rewards[:,2]).astype(np.float32)
        else:
            rewards = rewards.astype(np.float32)

        is_terminal_np = (next_states == 0).all(axis=1).astype(np.float32)
        discounts_np = 1.0 - is_terminal_np

        self.states = torch.tensor(states, dtype=torch.float32)
        self.actions = torch.tensor(actions, dtype=torch.float32)
        self.rewards = torch.tensor(rewards, dtype=torch.float32)
        self.next_states = torch.tensor(next_states, dtype=torch.float32)
        self.discounts = torch.tensor(discounts_np, dtype=torch.float32)
        self.size = len(self.states)

    def sample(self, batch_size):
        idx = torch.randint(0, self.size, (batch_size,), device="cpu")
        return (
            self.states[idx].to(self.device),
            self.actions[idx].to(self.device),
            self.rewards[idx].to(self.device),
            self.next_states[idx].to(self.device),
            self.discounts[idx].unsqueeze(-1).to(self.device),
        )


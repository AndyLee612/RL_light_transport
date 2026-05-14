# pylint: disable=protected-access

# Modify this code to work on light transport dataset
# State has dim 6 (xyz + normal vector)
# Action has dim 3 (direction)
# Reward has dim 3

"""
Trains agents on a static, offline dataset and
evaluates their performance periodically.
"""

import yaml
import torch
from argparse import ArgumentParser
import datetime
from pathlib import Path

from agents.workspaces_lt import LTOfflineWorkspace
from agents.cql.agent import CQL
from agents.fb.agent import FB
from agents.cfb.agent import CFB
from agents.td3.agent import TD3
from agents.gciql.agent import GCIQL
from agents.sf.agent import SF
from agents.osfb.agent import OneStepFB
# from agents.osfb.replay_buffer import 
from utils import set_seed_everywhere
import numpy as np

from data_loader import load_light_transport_npz

import os
import glob

def generate_sweep_run_name(config):
    """
    Generates a run name for sweep experiments.
    Includes:
        - the word 'sweep'
        - key hyperparameters depends on model(e.g. seed, alpha, z_dimension)
        - timestamp for uniqueness
    """
    time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    if (config['name']=="cql"):
        seed = str(config["seed"])

        name=f"sweep--seed{seed}-time{time}"
    elif(config['name']=="osfb"):
        zdim = str(config["z_dimension"]).replace(".", "_")
        alpha = str(config["alpha"]).replace(".", "_")
        seed = str(config["seed"])
        name=f"sweep-z{zdim}-alpha{alpha}--seed{seed}-{time}"

    return name

class LightTransportReplayBuffer:
    """
    Replay buffer that automatically loads and concatenates
    ALL LT dataset files of the form:

        rl_reward_light{ID}_batch_{B}.npz

    across multiple light positions.
    """

    def __init__(self, root_dir, device, reward_mode="luminance"):
        """
        Args:
            root_dir: directory containing *.npz files
            device: the target device for RETURNED batches (GPU or CPU)
            reward_mode: "luminance" or "rgb"
        """

        self.device = device   # only used when sampling batches

        # ---------------------------------------------------------
        # STEP 1 — Discover dataset files
        # ---------------------------------------------------------
        pattern = os.path.join(root_dir, "rl_reward_light*_batch_*.npz")
        
        file_list = sorted(glob.glob(pattern))

        if len(file_list) == 0:
            raise FileNotFoundError(f"No LT dataset files found in {root_dir}")

        print(f"Found {len(file_list)} datasets:")
        for f in file_list:
            print("   ", os.path.basename(f))

        # ---------------------------------------------------------
        # STEP 2 — Load and concatenate ALL transitions (CPU)
        # ---------------------------------------------------------
        all_states = []
        all_actions = []
        all_rewards = []
        all_next = []

        for file_path in file_list:
            S, A, R, S_next = load_light_transport_npz(file_path)
            all_states.append(S)
            all_actions.append(A)
            all_rewards.append(R)
            all_next.append(S_next)

        states = np.concatenate(all_states, axis=0)
        actions = np.concatenate(all_actions, axis=0)
        rewards = np.concatenate(all_rewards, axis=0)
        next_states = np.concatenate(all_next, axis=0)

        print("Total concatenated transitions:", states.shape[0])

        # ---------------------------------------------------------
        # STEP 3 — Compute terminals (CPU)
        # ---------------------------------------------------------
        is_terminal_np = (next_states == 0).all(axis=1).astype(np.float32)
        discounts_np = 1.0 - is_terminal_np  # 1 = continue, 0 = terminal

        # ---------------------------------------------------------
        # STEP 4 — Normalize states and actions (CPU)
        # ---------------------------------------------------------
        state_mean = states.mean(axis=0, keepdims=True)
        state_std = states.std(axis=0, keepdims=True) + 1e-6
        states = (states - state_mean) / state_std
        next_states = (next_states - state_mean) / state_std

        action_mean = actions.mean(axis=0, keepdims=True)
        action_std = actions.std(axis=0, keepdims=True) + 1e-6
        actions = (actions - action_mean) / action_std

        # ---------------------------------------------------------
        # STEP 5 — Reward conversion
        # ---------------------------------------------------------
        if reward_mode == "luminance":
            rewards = (
                0.2126 * rewards[:, 0]
                + 0.7152 * rewards[:, 1]
                + 0.0722 * rewards[:, 2]
            ).astype(np.float32)
        else:
            rewards = rewards.astype(np.float32)

        # ---------------------------------------------------------
        # STEP 6 — Compute next_actions (shift by one step)
        # ---------------------------------------------------------
        # next_actions[i] = actions[i+1] within the same trajectory.
        # At terminal steps (discount=0) the value is unused, so we
        # pad the last entry with zeros.
        next_actions = np.zeros_like(actions)
        next_actions[:-1] = actions[1:]
        # Zero out next_actions at terminal transitions (where the
        # next step belongs to a different episode)
        next_actions[is_terminal_np.astype(bool)] = 0.0

        # ---------------------------------------------------------
        # STEP 7 — Store ON CPU (IMPORTANT)
        # ---------------------------------------------------------
        self.states = torch.tensor(states, dtype=torch.float32)       # CPU
        self.actions = torch.tensor(actions, dtype=torch.float32)     # CPU
        self.next_actions = torch.tensor(next_actions, dtype=torch.float32)  # CPU
        self.rewards = torch.tensor(rewards, dtype=torch.float32)     # CPU
        self.next_states = torch.tensor(next_states, dtype=torch.float32)
        self.discounts = torch.tensor(discounts_np, dtype=torch.float32)

        self.size = len(self.states)

        print("=== FINAL REPLAY BUFFER ===")
        print("states:", self.states.shape)
        print("actions:", self.actions.shape)
        print("next_actions:", self.next_actions.shape)
        print("rewards:", self.rewards.shape)
        print("next_states:", self.next_states.shape)
        print("discounts:", self.discounts.shape)
        print("size:", self.size)
        print("===========================")

    # ---------------------------------------------------------
    # Sampling (returns tensors on GPU)
    # ---------------------------------------------------------
    def sample(self, batch_size):
        # indices sampled on CPU
        idx = torch.randint(0, self.size, (batch_size,), device="cpu")

        # move only the batch to GPU
        return (
            self.states[idx].to(self.device),
            self.actions[idx].to(self.device),
            self.rewards[idx].to(self.device),
            self.next_states[idx].to(self.device),
            self.discounts[idx].unsqueeze(-1).to(self.device),
            self.next_actions[idx].to(self.device),
        )

# ===========================================
# Light Transport fixed dimensions
# ===========================================
observation_length = 6      # pos(3) + normal(3)
action_length = 3           # direction vector (x, y, z)
action_range = [-1.0, 1.0]  # normalize direction vectors

parser = ArgumentParser()
parser.add_argument("algorithm", type=str)
parser.add_argument("--wandb_logging", type=str, default="True")
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--alpha", type=float, default=None)
parser.add_argument("--cql_alpha", type=float, default=0.01)
parser.add_argument("--discount", type=float, default=0.98)
parser.add_argument("--z_dimension", type=int, default=None)
parser.add_argument("--weighted_cml", type=bool, default=False)
parser.add_argument("--total_action_samples", type=int, default=12)
parser.add_argument("--ood_action_weight", type=float, default=0.5)
parser.add_argument("--train_task", type=str)
parser.add_argument("--dataset_transitions", type=int, default=100000)
parser.add_argument("--learning_steps", type=int, default=1000000)
parser.add_argument("--z_inference_steps", type=int, default=10000)
parser.add_argument("--run_name", type=str, default=None)
parser.add_argument("--model_name", type=str, default=None)
parser.add_argument("--lagrange", type=str, default="True")
parser.add_argument("--target_conservative_penalty", type=float, default=50.0)
parser.add_argument("--action_condition_index", type=int)
parser.add_argument("--action_condition_value", type=float)

args = parser.parse_args()

if args.wandb_logging == "True":
    args.wandb_logging = True
elif args.wandb_logging == "False":
    args.wandb_logging = False
else:
    raise ValueError("wandb_logging must be either True or False")

if args.algorithm in ("vcfb"):
    args.vcfb = True
    args.mcfb = False
elif args.algorithm in ("mcfb"):
    args.vcfb = False
    args.mcfb = True

if args.lagrange == "True":
    args.lagrange = True
elif args.lagrange == "False":
    args.lagrange = False

# action condition for subsampling dataset
if args.action_condition_index is not None:
    args.action_condition = {args.action_condition_index: args.action_condition_value}
else:
    args.action_condition = None



working_dir = Path.cwd()

if args.algorithm in ("vcfb", "mcfb"):
    algo_dir = "calfb" if "cal" in args.algorithm else "cfb"
    config_path = working_dir / "agents" / algo_dir / "config.yaml"
    model_dir = working_dir / "agents" / algo_dir / "saved_models" 
elif args.algorithm in ("sf-lap", "sf-hilp"):
    algo_dir = "sf"
    config_path = working_dir / "agents" / algo_dir / "config.yaml"
    model_dir = working_dir / "agents" / algo_dir / "saved_models"
elif args.algorithm == "osfb":
    config_path = working_dir / "agents" / "osfb" / "config.yaml"
    model_dir = working_dir / "agents" / "osfb" / "saved_models"
else:
    config_path = working_dir / "agents" / args.algorithm / "config.yaml"
    model_dir = working_dir / "agents" / args.algorithm / "saved_models"


# load the config file (YAML)
with open(config_path, "rb") as f:
    config = yaml.safe_load(f)

# Merge CLI arguments (CLI has priority)
# config.update(vars(args))

# (config has priority over args)
# only update config with args that were explicitly set by user
cli_args = {k: v for k, v in vars(args).items() 
            if v is not None}
config.update(cli_args) 

# Correct run name: call your sweep name generator
config["run_name"] = generate_sweep_run_name(config)

print("Run name:", config["run_name"])

# Select device
config["device"] = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else ("mps" if torch.backends.mps.is_built() else "cpu")
)

set_seed_everywhere(config["seed"])

# ==============================
# Light-Transport Dataset Loader
# ==============================
# We bypass RewardFunctionConstructor and all env logic.

# relabel not used for LT dataset
relabel = False

if config["algorithm"] == "cql":
    agent = CQL(
        observation_length=observation_length,
        action_length=action_length,
        device=config["device"],
        name=config["name"],
        batch_size=config["batch_size"],
        discount=config["discount"],
        critic_hidden_dimension=config["critic_hidden_dimension"],
        critic_hidden_layers=config["critic_hidden_layers"],
        critic_betas=config["critic_betas"],
        critic_tau=config["critic_tau"],
        critic_learning_rate=config["critic_learning_rate"],
        critic_target_update_frequency=config["critic_target_update_frequency"],
        actor_hidden_dimension=config["actor_hidden_dimension"],
        actor_hidden_layers=config["actor_hidden_layers"],
        actor_betas=config["actor_betas"],
        actor_learning_rate=config["actor_learning_rate"],
        actor_log_std_bounds=config["actor_log_std_bounds"],
        alpha_learning_rate=config["alpha_learning_rate"],
        alpha_betas=config["alpha_betas"],
        actor_update_frequency=config["actor_update_frequency"],
        init_temperature=config["init_temperature"],
        learnable_temperature=config["learnable_temperature"],
        activation=config["activation"],
        action_range=action_range,
        normalisation_samples=None,
        cql_n_samples=config["cql_n_samples"],
        cql_lagrange=config["lagrange"],
        cql_alpha=config["cql_alpha"],
        cql_target_penalty=config["target_conservative_penalty"],
    )

    buffer = LightTransportReplayBuffer(
        root_dir="./dataset",
        device=torch.device("cuda"),
        reward_mode="luminance"
    )

    z_inference_steps = None
    train_std = None
    eval_std = None

elif config["algorithm"] == "td3":
    agent = TD3(
        observation_length=observation_length,
        action_length=action_length,
        device=config["device"],
        name=config["name"],
        critic_hidden_dimension=config["critic_hidden_dimension"],
        critic_hidden_layers=config["critic_hidden_layers"],
        critic_learning_rate=config["critic_learning_rate"],
        critic_activation=config["activation"],
        actor_hidden_dimension=config["actor_hidden_dimension"],
        actor_hidden_layers=config["actor_hidden_layers"],
        actor_learning_rate=config["actor_learning_rate"],
        actor_activation=config["activation"],
        std_dev_clip=config["std_dev_clip"],
        std_dev_schedule=config["std_dev_schedule"],
        batch_size=config["batch_size"],
        discount=config["discount"],
        tau=config["critic_tau"],
    )

    buffer = LightTransportReplayBuffer(
        root_dir="./dataset",
        device=torch.device("cuda"),
        reward_mode="luminance"
    )

    z_inference_steps = None
    train_std = None
    eval_std = None

elif config["algorithm"] == "fb":

    agent = FB(
        observation_length=observation_length,
        action_length=action_length,
        preprocessor_hidden_dimension=config["preprocessor_hidden_dimension"],
        preprocessor_output_dimension=config["preprocessor_output_dimension"],
        preprocessor_hidden_layers=config["preprocessor_hidden_layers"],
        forward_hidden_dimension=config["forward_hidden_dimension"],
        forward_hidden_layers=config["forward_hidden_layers"],
        forward_number_of_features=config["forward_number_of_features"],
        backward_hidden_dimension=config["backward_hidden_dimension"],
        backward_hidden_layers=config["backward_hidden_layers"],
        actor_hidden_dimension=config["actor_hidden_dimension"],
        actor_hidden_layers=config["actor_hidden_layers"],
        preprocessor_activation=config["preprocessor_activation"],
        forward_activation=config["forward_activation"],
        backward_activation=config["backward_activation"],
        actor_activation=config["actor_activation"],
        z_dimension=config["z_dimension"],
        critic_learning_rate=config["critic_learning_rate"],
        actor_learning_rate=config["actor_learning_rate"],
        learning_rate_coefficient=config["learning_rate_coefficient"],
        orthonormalisation_coefficient=config["orthonormalisation_coefficient"],
        discount=config["discount"],
        batch_size=config["batch_size"],
        z_mix_ratio=config["z_mix_ratio"],
        gaussian_actor=config["gaussian_actor"],
        std_dev_clip=config["std_dev_clip"],
        std_dev_schedule=config["std_dev_schedule"],
        tau=config["tau"],
        device=config["device"],
        name=config["name"],
    )

    buffer = LightTransportReplayBuffer(
        root_dir="./dataset",
        device=torch.device("cuda"),
        reward_mode="luminance"
    )

    z_inference_steps = config["z_inference_steps"]
    train_std = config["std_dev_schedule"]
    eval_std = config["std_dev_eval"]

elif config["algorithm"] == "osfb":

    agent = OneStepFB(
        observation_length=observation_length,
        action_length=action_length,
        preprocessor_hidden_dimension=config["preprocessor_hidden_dimension"],
        preprocessor_output_dimension=config["preprocessor_output_dimension"],
        preprocessor_hidden_layers=config["preprocessor_hidden_layers"],
        preprocessor_activation=config["preprocessor_activation"],
        forward_hidden_dimension=config["forward_hidden_dimension"],
        forward_hidden_layers=config["forward_hidden_layers"],
        forward_number_of_features=config["forward_number_of_features"],
        forward_activation=config["forward_activation"],
        backward_hidden_dimension=config["backward_hidden_dimension"],
        backward_hidden_layers=config["backward_hidden_layers"],
        backward_activation=config["backward_activation"],
        actor_hidden_dimension=config["actor_hidden_dimension"],
        actor_hidden_layers=config["actor_hidden_layers"],
        actor_activation=config["actor_activation"],
        z_dimension=config["z_dimension"],
        critic_learning_rate=config["critic_learning_rate"],
        actor_learning_rate=config["actor_learning_rate"],
        learning_rate_coefficient=config["learning_rate_coefficient"],
        orthonormalisation_coefficient=config["orthonormalisation_coefficient"],
        discount=config["discount"],
        batch_size=config["batch_size"],
        z_mix_ratio=config["z_mix_ratio"],
        gaussian_actor=config["gaussian_actor"],
        std_dev_clip=config["std_dev_clip"],
        std_dev_schedule=config["std_dev_schedule"],
        tau=config["tau"],
        device=config["device"],
        name=config["name"],
        # one-step FB specific
        repr_agg=config["repr_agg"],
        q_agg=config["q_agg"],
        alpha=config["alpha"],
        normalize_q_loss=config["normalize_q_loss"],
        const_std=config["const_std"],
    )

    buffer = LightTransportReplayBuffer(
        root_dir="./dataset",
        device=torch.device("cuda"),
        reward_mode="luminance"
    )

    z_inference_steps = config["z_inference_steps"]
    train_std = config["std_dev_schedule"]
    eval_std = config["std_dev_eval"]

elif config["algorithm"] in ("vcfb", "mcfb"):

    agent = CFB(
        observation_length=observation_length,
        action_length=action_length,
        preprocessor_hidden_dimension=config["preprocessor_hidden_dimension"],
        preprocessor_output_dimension=config["preprocessor_output_dimension"],
        preprocessor_hidden_layers=config["preprocessor_hidden_layers"],
        forward_hidden_dimension=config["forward_hidden_dimension"],
        forward_hidden_layers=config["forward_hidden_layers"],
        forward_number_of_features=config["forward_number_of_features"],
        backward_hidden_dimension=config["backward_hidden_dimension"],
        backward_hidden_layers=config["backward_hidden_layers"],
        actor_hidden_dimension=config["actor_hidden_dimension"],
        actor_hidden_layers=config["actor_hidden_layers"],
        preprocessor_activation=config["preprocessor_activation"],
        forward_activation=config["forward_activation"],
        backward_activation=config["backward_activation"],
        actor_activation=config["actor_activation"],
        z_dimension=config["z_dimension"],
        actor_learning_rate=config["actor_learning_rate"],
        critic_learning_rate=config["critic_learning_rate"],
        learning_rate_coefficient=config["learning_rate_coefficient"],
        orthonormalisation_coefficient=config["orthonormalisation_coefficient"],
        discount=config["discount"],
        batch_size=config["batch_size"],
        z_mix_ratio=config["z_mix_ratio"],
        gaussian_actor=config["gaussian_actor"],
        std_dev_clip=config["std_dev_clip"],
        std_dev_schedule=config["std_dev_schedule"],
        tau=config["tau"],
        device=config["device"],
        vcfb=config["vcfb"],
        mcfb=config["mcfb"],
        total_action_samples=config["total_action_samples"],
        ood_action_weight=config["ood_action_weight"],
        alpha=config["alpha"],
        target_conservative_penalty=config["target_conservative_penalty"],
        lagrange=config["lagrange"],
    )

    buffer = LightTransportReplayBuffer(
        root_dir="./dataset",
        device=torch.device("cuda"),
        reward_mode="luminance"
    )

    z_inference_steps = config["z_inference_steps"]
    train_std = config["std_dev_schedule"]
    eval_std = config["std_dev_eval"]

elif config["algorithm"] == "gciql":
    agent = GCIQL(
        observation_length=observation_length,
        action_length=action_length,
        device=config["device"],
        name=config["name"],
        critic_hidden_dimension=config["critic_hidden_dimension"],
        critic_hidden_layers=config["critic_hidden_layers"],
        critic_activation=config["activation"],
        actor_hidden_dimension=config["actor_hidden_dimension"],
        actor_hidden_layers=config["actor_hidden_layers"],
        actor_learning_rate=config["actor_learning_rate"],
        actor_activation=config["activation"],
        batch_size=config["batch_size"],
        discount=config["discount"],
        tau=config["critic_tau"],
        actor_update_frequency=config["actor_update_frequency"],
        temperature=config["temperature"],
        expectile=config["expectile"],
        value_learning_rate=config["value_learning_rate"],
        critic_target_update_frequency=config["critic_target_update_frequency"],
    )

    # load buffer
    buffer = LightTransportReplayBuffer(
        root_dir="./dataset",
        device=torch.device("cuda"),
        reward_mode="luminance"
    )

    z_inference_steps = config["z_inference_steps"]
    train_std = None
    eval_std = None

elif config["algorithm"] == "sf-lap":

    agent = SF(
        observation_length=observation_length,
        action_length=action_length,
        preprocessor_hidden_dimension=config["preprocessor_hidden_dimension"],
        preprocessor_output_dimension=config["preprocessor_output_dimension"],
        preprocessor_hidden_layers=config["preprocessor_hidden_layers"],
        forward_hidden_dimension=config["forward_hidden_dimension"],
        forward_hidden_layers=config["forward_hidden_layers"],
        forward_number_of_features=config["forward_number_of_features"],
        features_hidden_dimension=config["features_hidden_dimension"],
        features_hidden_layers=config["features_hidden_layers"],
        features_activation=config["features_activation"],
        actor_hidden_dimension=config["actor_hidden_dimension"],
        actor_hidden_layers=config["actor_hidden_layers"],
        preprocessor_activation=config["preprocessor_activation"],
        forward_activation=config["forward_activation"],
        actor_activation=config["actor_activation"],
        z_dimension=config["z_dimension"],
        sf_learning_rate=config["sf_learning_rate"],
        feature_learning_rate=config["feature_learning_rate"],
        actor_learning_rate=config["actor_learning_rate"],
        batch_size=config["batch_size"],
        gaussian_actor=config["gaussian_actor"],
        std_dev_clip=config["std_dev_clip"],
        std_dev_schedule=config["std_dev_schedule"],
        tau=config["tau"],
        device=config["device"],
        name=config["name"],
        z_mix_ratio=config["z_mix_ratio"],
        q_loss=True,
    )

    buffer = LightTransportReplayBuffer(
        root_dir="./dataset",
        device=torch.device("cuda"),
        reward_mode="luminance"
    )

    z_inference_steps = config["z_inference_steps"]
    train_std = config["std_dev_schedule"]
    eval_std = config["std_dev_eval"]

else:
    raise NotImplementedError(f"Algorithm {config['algorithm']} not implemented")

workspace = LTOfflineWorkspace(
    learning_steps=config["learning_steps"],
    model_dir=model_dir,
    eval_frequency=config["eval_frequency"],
    wandb_logging=config["wandb_logging"],
    device=config["device"],
    run_name=config["run_name"],     # <-- ADD THIS
)

if __name__ == "__main__":
    workspace.train(
        agent=agent,
        agent_config=config,
        replay_buffer=buffer,
    )
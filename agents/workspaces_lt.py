import wandb
import torch
import shutil
import os 
from os import makedirs
from loguru import logger
from tqdm import tqdm
import numpy as np
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional
from agents.base import Batch
from dotenv import load_dotenv


class LTOfflineWorkspace:
    """
    Minimal offline RL workspace for Light Transport.
    Saves ONLY the best model based on FB total loss,
    checked only every `eval_frequency` steps.
    """

    def __init__(
        self,
        learning_steps: int,
        model_dir: Path,
        eval_frequency: int,
        wandb_logging: bool,
        device: torch.device,
        run_name: Optional[str] = None
    ):
        self.learning_steps = learning_steps
        self.model_dir = model_dir
        self.eval_frequency = eval_frequency
        self.wandb_logging = wandb_logging
        self.device = device
        self.run_name = run_name      

    def train(self, agent, agent_config: Dict, replay_buffer):
        """
        Train an offline RL agent on the Light Transport dataset.
        """

        # -------------------------
        # Setup run directory
        # -------------------------
        if self.wandb_logging:
            load_dotenv()
            wandb.login(key=os.environ.get("WANDB_API_KEY"))
            run = wandb.init(
                name=self.run_name,
                config=agent_config,
                tags=[agent.name],
                reinit="finish_previous",
            )
            model_path = self.model_dir / run.name
        else:
            date = datetime.today().strftime("Y-%m-%d-%H-%M-%S")
            model_path = self.model_dir / f"local-run-{date}"

        makedirs(model_path, exist_ok=True)
        logger.info(f"Training {agent.name}.")

        # -------------------------
        # Track best model
        # -------------------------
        best_loss = float("inf")
        best_model_path = model_path / "best_model.pt"

        # -------------------------
        # Main training loop
        # -------------------------
        for step in tqdm(range(self.learning_steps + 1)):
            # Sample batch
            ## 04232026 fix it with 
            s, a, r, s_next, discounts,a_n = replay_buffer.sample(agent.batch_size)
            not_dones = (discounts > 0).float()

            batch = Batch(
                observations=s,
                actions=a,
                rewards=r,
                next_observations=s_next,
                discounts=discounts,
                next_actions=a_n,
                not_dones=not_dones,
            )

            # Update agent
            train_metrics = agent.update(batch, step=step)

            # -------------------------
            # Periodic evaluation + checkpoint
            # -------------------------
            if step % self.eval_frequency == 0:
                current_loss = float(train_metrics["train/total_loss"])

                logger.info(f"[Step {step}] Eval: FB Loss = {current_loss:.6f}")

                # Check if best
                if current_loss < best_loss:
                    best_loss = current_loss
                    logger.info(f"  ↳ New BEST model! Saving to {best_model_path}")

                    agent._name = f"best-step{step}"
                    agent.save(best_model_path)

            # -------------------------
            # W&B Logging
            # -------------------------
            if self.wandb_logging:
                wandb.log({**train_metrics, "best_loss": best_loss})

        # Finalize
        if self.wandb_logging:
            run.save(best_model_path.as_posix(), base_path=model_path.as_posix())
            run.finish()

        logger.info(f"Training complete. Best model saved at:\n  {best_model_path}")

import os
from dotenv import load_dotenv
import wandb

load_dotenv()

result = wandb.login(key=os.environ.get("WANDB_API_KEY"))
print("Login success:", result)
print("Entity:", wandb.api.default_entity)
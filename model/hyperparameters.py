from loguru import logger
import os

def get_hparams():
    params = {
        "learning_rate": 1e-4,
        "n_epochs": 50,
        "batch_size": 2,
        "size": (512, 512),
        "checkpoint_path": os.path.join(os.getcwd(), "checkpoints/checkpoint.pth")
        }

    logger.info(f"Hyperparameters: {params}")

    return params
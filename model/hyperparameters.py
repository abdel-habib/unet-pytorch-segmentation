from loguru import logger

def get_hparams():
    params = {
        "learning_rate": 1e-4,
        "n_epochs": 50,
        "batch_size": 2,
        "size": (512, 512),
        "checkpoint_path": "checkpoints/checkpoint.pth"
        }

    logger.info(f"Hyperparameters: {params}")

    return params
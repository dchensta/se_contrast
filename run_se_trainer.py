from SE_Trainer import SE_Trainer
import random
import numpy as np

if __name__ == "__main__":
    sd_train_dir = "sd_train_official"
    sd_model_dir = "sd_model"

    random.seed(0)
    np.random.seed(0)

    trainer = SE_Trainer(sd_train_dir, sd_model_dir)
    clf_results = trainer.train()
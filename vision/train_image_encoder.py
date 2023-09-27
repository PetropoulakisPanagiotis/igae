from config import Config
import os
import argparse

import torch
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
import optuna

from img_processing.ae.autoencoder import Autoencoder
from img_processing.mae.masked_autoencoder import MaskedAutoencoder
from img_processing.sp.state_predictor import StatePredictor
from img_processing.dataloader import Img_RGB_Dataset, Img_Dataset_Seg_Masks, Img_Dataset_States, get_data_loader


def objective(trial, image_path: str, n_epochs: int, ae_type: str, log_dir: str, augmentation_prob: float, split=0.8,
              seed=0, bs=8, log_interval=5, save_path=None, image_dim=128, num_workers=8, **kwargs) -> float:

    ae_logger = TensorBoardLogger(save_dir=log_dir)
    save_path = save_path or os.path.join("agent/img_processing", ae_type, "trained_models")

    pl.seed_everything(seed)
    trainer = pl.Trainer(max_epochs=n_epochs, check_val_every_n_epoch=log_interval, logger=ae_logger,
                         callbacks=[optuna.integration.PyTorchLightningPruningCallback(trial, monitor="val_loss")],
                         devices=1, accelerator="gpu" if torch.cuda.is_available() else "cpu")

    if ae_type == "ae":
        hparams = {
            "learning_rate": 0.006, # trial.suggest_float("learning_rate", 0.00001, 0.01), # 0.005
            "weight_decay": 4e-8, # trial.suggest_float("weight_decay", 2e-8, 4e-8),
            "batch_size": bs, # trial.suggest_categorical("batch_size", [32, 64, 128, 256]),
        }
        dataset = Img_RGB_Dataset(image_path=image_path, image_dim=image_dim, augmentation_prob=augmentation_prob)
        train_loader, val_loader = get_data_loader(dataset, split=split, bs=bs, num_workers=num_workers)
        model = Autoencoder(hparams, image_dim, train_loader, val_loader, dataset.mean, dataset.std)
    elif ae_type == "mae":
        hparams = {
            "learning_rate": 0.006, # trial.suggest_float("learning_rate", 0.007, 0.0075), # trial.suggest_float("learning_rate", 0.001, 0.0075),
            "weight_decay": 4e-8, # trial.suggest_float("weight_decay", 3e-8, 5e-8),
            "box_reg": 20,
            "gripper_reg": 10,
            "batch_size": bs,
        }
        dataset = Img_Dataset_Seg_Masks(image_path=image_path, image_dim=image_dim, augmentation_prob=augmentation_prob)
        train_loader, val_loader = get_data_loader(dataset, split=split, bs=bs, num_workers=num_workers)
        model = MaskedAutoencoder(hparams, image_dim, train_loader, val_loader, dataset.mean, dataset.std)
    elif ae_type == "sp":
        hparams = {
            "learning_rate": 0.006, # trial.suggest_float("learning_rate", 0.007, 0.0075), # trial.suggest_float("learning_rate", 0.001, 0.0075),
            "weight_decay": 4e-8, # trial.suggest_float("weight_decay", 3e-8, 5e-8),
            "batch_size": bs,
        }
        dataset = Img_Dataset_States(image_path=image_path, image_dim=image_dim, augmentation_prob=augmentation_prob)
        train_loader, val_loader = get_data_loader(dataset, split=split, bs=bs, num_workers=num_workers)
        model = StatePredictor(hparams, image_dim, train_loader, val_loader, dataset.mean, dataset.std)
    else:
        raise NotImplementedError("Use either ae, mae or sp")

    trainer.logger.log_hyperparams(hparams)
    trainer.fit(model)

    loss = trainer.callback_metrics["train_loss"].item()
    model_path = os.path.join(save_path, f"model_{trial.number}")
    trainer.save_checkpoint(model_path)

    return loss


def optuna_search(n_trials=10, **kwargs) -> None:
    pruner: optuna.pruners.BasePruner = (optuna.pruners.MedianPruner(n_startup_trials=3))
    study = optuna.create_study(direction="minimize", pruner=pruner, sampler=optuna.samplers.TPESampler(seed=12))
    study.optimize(lambda trial: objective(trial, **kwargs), n_trials=n_trials, timeout=None)

    print(f"Number of finished trials: {len(study.trials)}")
    trial = study.best_trial
    print(f"Best trial: Value: {trial.value}")
    print("Params: ")
    for key, value in trial.params.items():
        print(f"\t{key}: {value}")


if __name__ == "__main__":
    args = Config.get_ae_training_params()
    optuna_search(**args)

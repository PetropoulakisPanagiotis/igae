import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
import pytorch_lightning as pl

from img_processing.nn_parts import Encoder


class StatePredictor(pl.LightningModule):
    def __init__(self, hparams=None, image_dim=None, train_loader=None, val_loader=None, mean=None, std=None):
        super().__init__()

        if hparams is not None:
            for key in hparams.keys():
                self.hparams[key] = hparams[key]

        self.train_loader = train_loader
        self.val_loader = val_loader

        self.mean = mean
        self.std = std

        self.channels = 3
        self.image_dim = image_dim
        self.build_model(image_dim)

    def on_train_start(self):
        self.logger.log_hyperparams(self.hparams)

    def forward(self, x):
        latent_dim = self.encoder(x)
        state = self.linear(latent_dim)
        return state

    def general_step(self, batch, mode):
        [img], ground_truth = batch
        state = self.forward(img)
        loss = F.mse_loss(state, ground_truth)
        return loss, state

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        # this calls forward
        return self(batch)

    def validation_step(self, batch, mode):
        """
        The batch consists of the images that were augmented and the original ones.
        We want to train the network to reconstruct the real image.
        """
        loss, _ = self.general_step(batch, "val")
        self.log("val_loss", loss)
        return loss

    def general_end(self, outputs, mode):
        avg_loss = torch.stack([x[mode + '_loss'] for x in outputs]).mean()
        return avg_loss

    def training_step(self, batch):
        loss, _ = self.general_step(batch, "train")
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def train_dataloader(self):
        return self.train_loader

    def val_dataloader(self):
        return self.val_loader

    def configure_optimizers(self):
        params = [{'params': self.encoder.parameters()}, {'params': self.linear.parameters()}]
        optim = torch.optim.Adam(params, self.hparams["learning_rate"], weight_decay=self.hparams["weight_decay"])
        return [optim]

    def build_model(self, image_dim: int):
        self.encoder = Encoder(image_dim)

        self.linear = nn.Sequential(
            nn.Linear(128, 64, bias=False),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Linear(64, 32, bias=False),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Linear(32, 16, bias=False),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.Linear(16, 9),
        )

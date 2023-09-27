import torch
import torch.nn.functional as F
import pytorch_lightning as pl

from img_processing.nn_parts import Encoder, Decoder


class Autoencoder(pl.LightningModule):
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

        self.encoder = Encoder(image_dim)
        self.decoder = Decoder(image_dim, output_channels=3)

    def on_train_start(self):
        self.logger.log_hyperparams(self.hparams)

    def forward(self, x):
        latent_dim = self.encoder(x)
        reconstruction = self.decoder(latent_dim[:, :, None, None])
        # reconstruction = self.decoder(latent_dim)
        return reconstruction

    def general_step(self, batch, mode):
        x, x_augmented = batch[:, 0], batch[:, 1]
        reconstruction = self.forward(x_augmented)
        loss = F.mse_loss(reconstruction, x)
        return loss, reconstruction

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        # this calls forward
        return self(batch)

    def validation_step(self, batch, mode):
        """
        The batch consists of the images that were augmented and the original ones.
        We want to train the network to reconstruct the real image.
        """
        loss, reconstruction = self.general_step(batch, "val")
        reconstruction, batch = reconstruction.cpu(), batch.cpu()
        self.log("val_loss", loss)
        return loss, reconstruction, batch

    def validation_epoch_end(self, validation_step_outputs):
        reconstruction = validation_step_outputs[0][1]
        batch = validation_step_outputs[0][2]
        images = torch.zeros((3 * len(reconstruction), 3, self.image_dim, self.image_dim))
        for i in range(len(reconstruction)):
            images[3 * i] = reconstruction[i] * self.std[:, None, None] + self.mean[:, None, None]
            images[3 * i + 1] = batch[i, 0] * self.std[:, None, None] + self.mean[:, None, None]
            images[3 * i + 2] = batch[i, 1] * self.std[:, None, None] + self.mean[:, None, None]
        tensorboard = self.logger.experiment
        tensorboard.add_images('images_val', images.cpu(), self.current_epoch)

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
        params = [{'params': self.encoder.parameters()}, {'params': self.decoder.parameters()}]
        optim = torch.optim.Adam(params, self.hparams["learning_rate"], weight_decay=self.hparams["weight_decay"])
        return [optim]

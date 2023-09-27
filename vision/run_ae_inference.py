import os
import json
import argparse
from timeit import default_timer as timer

import torch
import torchvision.transforms as T
import torchvision

from img_processing.ae.autoencoder import Autoencoder
from img_processing.mae.masked_autoencoder import MaskedAutoencoder


class ImageInference:
    def __init__(self, path: str, img_meta_path: str):
        if "mae" in path:
            self.model = MaskedAutoencoder.load_from_checkpoint(path, image_dim=128).eval()
        else:
            self.model = Autoencoder.load_from_checkpoint(path, image_dim=128).eval()

        self.device = torch.device("cpu")
        with open(img_meta_path, 'r') as f:
            meta_json = json.load(f)
            self.mean = torch.Tensor(meta_json["mean"])
            self.std = torch.Tensor(meta_json["std"])

        self.normalize_transform = T.Normalize(self.mean, self.std)
        if torch.cuda.is_available():
            print("Feature-Backbone on CUDA")
            self.model = self.model.to(self.device)

    def __call__(self, imgs: torch.Tensor, use_timer: bool = False) -> None:
        imgs = self.normalize_transform(imgs.to(self.device))
        if use_timer == True:
            start = timer()
            reconstructions = self.model.predict_step(imgs, 0)
            end = timer()
            print(f"Took {end - start:2f} [s] to compute")  # Time in seconds
        else:
            reconstructions = self.model.predict_step(imgs, 0)
        return reconstructions


def run_inference(img_path: str, model_path: str, image_meta_path: str, save_dir: str, **kwargs) -> None:
    resize_transform = T.Resize((224, 224))
    image_inferer = ImageInference(model_path, image_meta_path)
    imgs_raw = [
        torchvision.io.read_image(os.path.join(img_path, img_name)) / 255
        for img_name in sorted(os.listdir(img_path), key=lambda x: int(x.split("_")[1].split(".")[0]))
    ]
    # imgs = torch.stack([resize_transform(img)[0:3] for img in imgs_raw])
    imgs = torch.stack([(img)[0:3] for img in imgs_raw])
    imgs = T.GaussianBlur(3)(imgs)

    reconstructions = image_inferer(imgs, True)
    os.makedirs(save_dir, exist_ok=True)
    for ii, rec in enumerate(reconstructions.detach().cpu()):
        rec = ((rec[0:3] * image_inferer.std[:, None, None] + image_inferer.mean[:, None, None]) * 255).byte()
        torchvision.io.write_jpeg(rec, os.path.join(save_dir, f"reconstruction_{ii}.jpeg"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Extracts images for training from Manipulator Environment')
    parser.add_argument('--img_path', help='Where to find the original images', type=str, required=True)
    parser.add_argument('--model_path', help='Where to find the model to infere', type=str, required=True)
    parser.add_argument('--image_meta_path', help='Image meta to normalization', type=str, required=True)
    parser.add_argument('--save_dir', help='Where to store the reconstructions', type=str, required=False)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_inference(**vars(args))

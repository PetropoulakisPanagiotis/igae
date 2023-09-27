import os
import PIL
import json
from typing import List

import torch
from torch.utils.data import Dataset, DataLoader, random_split
import numpy as np
from imgaug import augmenters as iaa
import torchvision.transforms as T
import torchvision.transforms.functional as F

MAX_JOINT_POS = np.deg2rad([170, 120, 170, 120, 170, 120, 175])


def randomly_draw_augmentations(augmentations: list, img_size: int) -> T.Compose:
    random_augmentation = np.random.choice(augmentations, size=np.random.randint(0, len(augmentations)))
    return augmentation_to_ops(random_augmentation, img_size)


def augmentation_to_ops(augmentation_keys: List[str], img_size: int):
    crop_size = int(np.random.uniform(0.9, 1.0) * img_size)
    brightness = np.random.uniform(0.65, 1.35)
    red_add = np.random.randint(-30, 30)
    green_add = np.random.randint(-30, 30)
    blue_add = np.random.randint(-30, 30)

    startpoints, endpoints = T.RandomPerspective.get_params(img_size, img_size, np.random.uniform(0.01, 0.05))

    aug = {
        'Crop': T.Compose([T.CenterCrop(crop_size), T.Resize(img_size)]),
        'Blur': T.GaussianBlur(kernel_size=3, sigma=np.random.uniform(0.1, 0.6)),
        'Brightness': T.Lambda(lambda x: F.adjust_brightness(x, brightness)),
        'PerspectiveTransform': T.Lambda(lambda x: F.perspective(x, startpoints, endpoints, fill=0)),
        'RedAdd': T.Lambda(lambda x: torch.stack([torch.clip(x[0] + red_add, 0, 255), x[1], x[2]])),
        'GreenAdd': T.Lambda(lambda x: torch.stack([x[0], torch.clip(x[1] + green_add, 0, 255), x[2]])),
        'BlueAdd': T.Lambda(lambda x: torch.stack([x[0], x[1], torch.clip(x[2] + blue_add, 0, 255)])),
    }
    return T.Compose([aug[key] for key in augmentation_keys])


def get_augmentations() -> dict:
    augmentations = {
        't_0': [
            'Crop',
            'Blur',
            'Brightness',
            'PerspectiveTransform',
            'RedAdd',
            'GreenAdd',
            'BlueAdd',
        ],
        't_1': [
            iaa.size.Crop((0, 20)),
            iaa.GaussianBlur(sigma=(0.1, 0.6)),
            iaa.MultiplyBrightness(mul=(0.65, 1.35)),
            iaa.PerspectiveTransform(scale=(0.01, 0.05)),
            iaa.WithChannels(0, iaa.Add((-30, 30))),
            iaa.WithChannels(1, iaa.Add((-30, 30))),
            iaa.WithChannels(2, iaa.Add((-30, 30)))
        ]
    }
    return augmentations


def get_data_loader(dataset, split=0.8, bs=8, num_workers=8) -> (DataLoader, DataLoader):
    train_size = int(split * len(dataset))
    trainset, valset = random_split(dataset, [train_size, len(dataset) - train_size])

    train_loader = DataLoader(trainset, batch_size=bs, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(valset, batch_size=bs, shuffle=False, num_workers=num_workers)
    return train_loader, val_loader


def pil_loader(path: str) -> PIL.Image.Image:
    with open(path, "rb") as f:
        img = PIL.Image.open(f)
        return img.convert("RGB")


class Img_RGB_Dataset(Dataset):
    def __init__(self, image_path, image_dim, augmentation_prob):
        with open(os.path.join(image_path, "image_meta.json"), 'r') as f:
            self.meta_json = json.load(f)
        self.mean = torch.Tensor(self.meta_json["mean"])
        self.std = torch.Tensor(self.meta_json["std"])
        self.augmentation_prob = augmentation_prob

        image_path_imgs = os.path.join(image_path, "images")
        image_path_augs = os.path.join(image_path, "augmentations")

        self.image_paths = [os.path.join(image_path_imgs, f) for f in sorted(os.listdir(image_path_imgs))]
        self.aug_paths = [os.path.join(image_path_augs, f) for f in sorted(os.listdir(image_path_augs))]
        self.image_dim = image_dim

        self.augmentation = T.Compose([
            iaa.Sometimes(augmentation_prob, iaa.SomeOf((1, None),
                                                        get_augmentations()["t_1"])).augment_image,
            T.ToTensor()
        ])
        self.normalize_transform = T.Normalize(self.mean, self.std)

    def __getitem__(self, index):
        image_path = self.image_paths[index]
        aug_path = self.aug_paths[index]

        x_real = np.array(pil_loader(image_path))
        if np.random.uniform() < self.augmentation_prob:
            x_augmented = np.array(pil_loader(aug_path))
            x_augmented = self.augmentation(x_augmented)
        else:
            x_augmented = self.augmentation(x_real)

        x_stacked = torch.stack([T.functional.to_tensor(x_real), x_augmented])
        x_stacked = self.normalize_transform(x_stacked)
        return x_stacked

    def __len__(self):
        return len(self.image_paths)


class Img_Dataset_Seg_Masks(torch.utils.data.Dataset):
    def __init__(self, image_path, image_dim, augmentation_prob):
        with open(os.path.join(image_path, "image_meta.json"), 'r') as f:
            self.meta_json = json.load(f)
        self.mean = torch.Tensor(self.meta_json["mean"])
        self.std = torch.Tensor(self.meta_json["std"])
        self.augmentation_prob = augmentation_prob

        image_path_imgs = os.path.join(image_path, "images")
        image_path_box = os.path.join(image_path, "box")
        image_path_gripper = os.path.join(image_path, "gripper")

        image_path_augs = os.path.join(image_path, "augmentations")

        self.image_paths = [os.path.join(image_path_imgs, f) for f in sorted(os.listdir(image_path_imgs))]
        self.mask_box_paths = [os.path.join(image_path_box, f) for f in sorted(os.listdir(image_path_imgs))]
        self.mask_gripper_paths = [os.path.join(image_path_gripper, f) for f in sorted(os.listdir(image_path_imgs))]

        self.aug_paths = [os.path.join(image_path_augs, f) for f in sorted(os.listdir(image_path_augs))]

        self.image_dim = image_dim

        self.augmentation = T.Compose([
            iaa.Sometimes(augmentation_prob, iaa.SomeOf((1, None),
                                                        get_augmentations()["t_1"])).augment_image,
            T.ToTensor()
        ])
        self.normalize_transform = T.Normalize(self.mean, self.std)

    def __getitem__(self, index):
        mask_box_path = self.mask_box_paths[index]
        mask_gripper_path = self.mask_gripper_paths[index]
        image_path = self.image_paths[index]
        aug_path = self.aug_paths[index]

        x_real = np.array(pil_loader(image_path))
        mask_box = T.functional.rgb_to_grayscale(T.functional.to_tensor(pil_loader(mask_box_path)))
        mask_gripper = T.functional.rgb_to_grayscale(T.functional.to_tensor(pil_loader(mask_gripper_path)))

        if np.random.uniform() < self.augmentation_prob:
            x_augmented = np.array(pil_loader(aug_path))
            x_augmented = self.augmentation(x_augmented)
        else:
            x_augmented = self.augmentation(x_real)

        xs = torch.concat([
            self.normalize_transform(T.functional.to_tensor(x_real)),
            self.normalize_transform(x_augmented), mask_box, mask_gripper
        ])
        return xs

    def __len__(self):
        return len(self.image_paths)


class Img_Dataset_States(torch.utils.data.Dataset):
    def __init__(self, image_path, image_dim, augmentation_prob):
        with open(os.path.join(image_path, "image_meta.json"), 'r') as f:
            self.meta_json = json.load(f)
        self.mean = torch.Tensor(self.meta_json["mean"])
        self.std = torch.Tensor(self.meta_json["std"])
        self.augmentation_prob = augmentation_prob

        image_path_imgs = os.path.join(image_path, "images")
        image_path_augs = os.path.join(image_path, "augmentations")

        state_path = os.path.join(image_path, "states")

        self.image_paths = [os.path.join(image_path_imgs, f) for f in sorted(os.listdir(image_path_imgs))]
        self.aug_paths = [os.path.join(image_path_augs, f) for f in sorted(os.listdir(image_path_augs))]
        self.state_paths = [os.path.join(state_path, f) for f in sorted(os.listdir(state_path))]

        self.image_dim = image_dim

        self.augmentation = T.Compose([
            iaa.Sometimes(augmentation_prob, iaa.SomeOf((1, None),
                                                        get_augmentations()["t_1"])).augment_image,
            T.ToTensor()
        ])
        self.normalize_transform = T.Normalize(self.mean, self.std)

    def __getitem__(self, index):
        image_path = self.image_paths[index]
        aug_path = self.aug_paths[index]
        state_path = self.state_paths[index]

        x_real = np.array(pil_loader(image_path))
        if np.random.uniform() < self.augmentation_prob:
            x_augmented = np.array(pil_loader(aug_path))
            x_augmented = self.augmentation(x_augmented)
        else:
            x_augmented = self.augmentation(x_real)
        xs = self.normalize_transform(x_augmented),

        with open(state_path) as f:
            state = torch.Tensor(json.load(f)["state"])

        return xs, state

    def __len__(self):
        return len(self.image_paths)
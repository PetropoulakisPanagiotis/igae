import os
import json
import gym
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as T
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

from img_processing.nn_parts import Encoder


class ImgBaseFeaturesExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.spaces.Box, features_dim: int) -> None:
        super(ImgBaseFeaturesExtractor, self).__init__(observation_space, features_dim)

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        observations_cnn = self.cnn(observations)
        return self.linear(observations_cnn)


class EndToEndExtractor(ImgBaseFeaturesExtractor):
    def __init__(self, observation_space: gym.spaces.Box, features_dim: int, image_dim: int) -> None:
        super(EndToEndExtractor, self).__init__(observation_space, features_dim)

        self.cnn = Encoder(image_dim)
        self.linear = nn.Sequential(nn.Identity())

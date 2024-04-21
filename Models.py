import torch
import torch.nn as nn
from transformers import ViTModel
from torchvision import models


class ViT(nn.Module):
    """
    Vision Transformer Model.
    """
    def __init__(self, num_classes = 1):
        """
        Constructor for the Vision Transformer Model by pre-trained ViT
        model from vit-base-patch16-224-in21k.
        :param num_classes: int, number of classes to classify
        """
        self.checkpoint = "google/vit-base-patch16-224-in21k"
        super(ViT, self).__init__()
        self.vit = ViTModel.from_pretrained(self.checkpoint)
        self.classifier = nn.Linear(self.vit.config.hidden_size, num_classes)

    def forward(self, x):
        """
        Forward pass of the Vision Transformer Model.
        :param x: tensor of shape [B, C, H, W]
        :return: classifications probabilities
        """
        x = self.vit(x)["last_hidden_state"]
        output = self.classifier(x[:, 0, :])
        return torch.sigmoid(output)


class ResNet(nn.Module):
    """
    ResNet Model.
    """
    def __init__(self, num_classes = 1):
        """
        Constructor for the ResNet Model by pre-trained ResNet50 from Pytorch.
        :param num_classes: int, number of classes to classify
        """
        super(ResNet, self).__init__()
        self.resnet50 = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        self.resnet50.fc = nn.Identity()
        self.classifier = nn.Linear(2048, num_classes)

    def forward(self, x):
        """
        Forward pass of the ResNet Model.
        :param x: tensor of shape [B, C, H, W]
        :return: classifications probabilities
        """
        x = self.resnet50(x)
        x = self.resnet50.fc(x)
        output = self.classifier(x)
        return torch.sigmoid(output)

"""ResNet-50 image features + sex input. Forward returns z-score; months = z*STD+MEAN."""
import torch
import torch.nn as nn
import torchvision


class BoneAgeNet(nn.Module):
    def __init__(self, sex_dim=32, head_hidden=256):
        super().__init__()
        self.cnn = torchvision.models.resnet50(weights="IMAGENET1K_V2")
        n = self.cnn.fc.in_features  # 2048
        self.cnn.fc = nn.Identity()
        self.sex = nn.Linear(1, sex_dim)
        self.head = nn.Sequential(
            nn.Linear(n + sex_dim, head_hidden),
            nn.ReLU(),
            nn.Linear(head_hidden, 1),
        )

    def forward(self, x, male):
        feats = self.cnn(x)
        s = torch.relu(self.sex(male.float().view(-1, 1)))
        return self.head(torch.cat([feats, s], dim=1)).squeeze(1)

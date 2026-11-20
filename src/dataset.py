"""Dataset: CSV id,boneage,male + PNG folder. Trains on z=(y-MEAN)/STD."""
import os
import pandas as pd
import torch
from torch.utils.data import Dataset
from .preprocessing import load_hand, to_model_tensor


class BoneAgeDataset(Dataset):
    def __init__(self, csv_path, img_dir, mean, std, size=512, train=False):
        df = pd.read_csv(csv_path)
        df["male"] = df["male"].astype(float)
        self.ids = df["id"].tolist()
        self.boneage = df["boneage"].to_numpy(dtype="float32")
        self.male = df["male"].to_numpy(dtype="float32")
        self.img_dir = img_dir
        self.mean, self.std = float(mean), float(std)
        self.size = size
        self.train = train
        self.target = (self.boneage - self.mean) / self.std

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, i):
        path = os.path.join(self.img_dir, f"{self.ids[i]}.png")
        gray = load_hand(path, self.size)
        # Light train-time brightness jitter; geometry aug lives in train.py
        x = to_model_tensor(gray)
        return x, torch.tensor(self.male[i]), torch.tensor(self.target[i])

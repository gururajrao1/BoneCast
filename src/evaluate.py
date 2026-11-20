"""Evaluate: MAE in months overall + boys/girls vs mean baselines. Uses GPU if present."""
import argparse, json
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from .dataset import BoneAgeDataset
from .model import BoneAgeNet


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--csv", default="data/boneage-training-dataset.csv")
    p.add_argument("--img-dir", default="data/boneage-training-dataset")
    p.add_argument("--ckpt", default="checkpoints/best.pt")
    p.add_argument("--size", type=int, default=512)
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    norm = json.load(open("checkpoints/normalizer.json"))
    mean, std = norm["mean"], norm["std"]
    ds = BoneAgeDataset(args.csv, args.img_dir, mean, std, size=args.size)
    loader = DataLoader(ds, batch_size=32)

    model = BoneAgeNet().to(device)
    model.load_state_dict(torch.load(args.ckpt, map_location=device))
    model.eval()

    preds, trues, sexes = [], [], []
    with torch.no_grad():
        for x, male, z in loader:
            pm = model(x.to(device), male.to(device)).cpu().numpy() * std + mean
            preds.append(pm)
            trues.append(z.numpy() * std + mean)
            sexes.append(male.numpy())
    pred, true, sex = (np.concatenate(a) for a in (preds, trues, sexes))
    df = pd.DataFrame({"pred": pred, "true": true, "male": sex > 0.5})

    overall = np.abs(df.pred - df.true).mean()
    boys = np.abs(df[df.male].pred - df[df.male].true).mean()
    girls = np.abs(df[~df.male].pred - df[~df.male].true).mean()
    base = np.abs(df.true - df.true.mean()).mean()
    print(f"MAE months overall={overall:.2f} boys={boys:.2f} girls={girls:.2f}")
    print(f"mean-baseline MAE={base:.2f} (target: beat this; winner 4.2mo)")


if __name__ == "__main__":
    main()

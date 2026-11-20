"""Train on GPU when available: z-score MSE, AMP, report MAE in months."""
import argparse, json, os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, random_split
from torchvision.transforms import v2 as T
from .dataset import BoneAgeDataset
from .model import BoneAgeNet


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--csv", default="data/boneage-training-dataset.csv")
    p.add_argument("--img-dir", default="data/boneage-training-dataset")
    p.add_argument("--size", type=int, default=512)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--out", default="checkpoints")
    args = p.parse_args()

    device = get_device()
    print(f"device: {device} cuda_available={torch.cuda.is_available()}")
    if device.type == "cuda":
        print(f"gpu: {torch.cuda.get_device_name(0)}")

    df = pd.read_csv(args.csv)
    mean, std = float(df.boneage.mean()), float(df.boneage.std())
    os.makedirs(args.out, exist_ok=True)
    json.dump({"mean": mean, "std": std}, open(f"{args.out}/normalizer.json", "w"))
    print(f"train mean={mean:.2f} std={std:.2f} (months)")

    full = BoneAgeDataset(args.csv, args.img_dir, mean, std, size=args.size)
    n_val = min(1425, len(full) // 10)
    train_ds, val_ds = random_split(
        full, [len(full) - n_val, n_val],
        generator=torch.Generator().manual_seed(42))

    aug = T.Compose([
        T.RandomAffine(degrees=15, translate=(0.1, 0.1)),
        T.RandomApply([T.ColorJitter(brightness=0.2, contrast=0.2)], p=0.5),
    ])

    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True,
                              num_workers=2, pin_memory=device.type == "cuda")
    val_loader = DataLoader(val_ds, batch_size=args.batch, num_workers=2,
                            pin_memory=device.type == "cuda")

    model = BoneAgeNet().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=3)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    loss_fn = torch.nn.MSELoss()
    best = float("inf")

    for ep in range(args.epochs):
        model.train()
        for x, male, z in train_loader:
            x, male, z = x.to(device, non_blocking=True), male.to(device), z.to(device)
            if aug is not None:
                x = aug(x)
            opt.zero_grad()
            with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
                pred = model(x, male)
                loss = loss_fn(pred, z)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
        # val MAE in months
        model.eval()
        errs, sexes = [], []
        with torch.no_grad():
            for x, male, z in val_loader:
                x, male = x.to(device), male.to(device)
                pred_m = (model(x, male).cpu().numpy() * std + mean)
                true_m = (z.numpy() * std + mean)
                errs.append(np.abs(pred_m - true_m))
                sexes.append(male.cpu().numpy())
        err = np.concatenate(errs)
        mae = err.mean()
        sched.step(mae)
        print(f"epoch {ep+1}: val MAE {mae:.2f} months")
        if mae < best:
            best = mae
            torch.save(model.state_dict(), f"{args.out}/best.pt")
    print(f"best val MAE: {best:.2f} months")


if __name__ == "__main__":
    main()

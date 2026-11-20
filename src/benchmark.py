"""BoneCast benchmark: every model + algo, GPU when available.

Runs fully on synthetic data (no download needed) and upgrades to real
CSVs automatically when data/ is populated. Compares:
  baselines: mean, sex-stratified mean
  models: ResNet50+sex (main), ResNet50 no-sex, ResNet18+sex
  algos: MSE vs L1 on z-scores (one synthetic step each)
Reports params, GPU forward time, and MAE in months.
"""
import os
import time

import numpy as np
import torch
import torch.nn as nn
import torchvision

from .model import BoneAgeNet


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


class NoSexNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.cnn = torchvision.models.resnet50(weights="IMAGENET1K_V2")
        n = self.cnn.fc.in_features
        self.cnn.fc = nn.Identity()
        self.head = nn.Sequential(nn.Linear(n, 256), nn.ReLU(), nn.Linear(256, 1))

    def forward(self, x, male=None):
        return self.head(self.cnn(x)).squeeze(1)


class BackboneSexNet(nn.Module):
    """Any torchvision backbone + sex fusion. Interdomain-ready head."""

    def __init__(self, backbone, feat_dim, sex_dim=32, hidden=256):
        super().__init__()
        self.cnn = backbone
        if hasattr(self.cnn, "fc"):
            self.cnn.fc = nn.Identity()          # ResNet
        elif hasattr(self.cnn, "heads"):
            self.cnn.heads = nn.Identity()       # ViT
        elif hasattr(self.cnn, "classifier"):
            self.cnn.classifier = nn.Identity()  # ConvNeXt / EfficientNet
        self.sex = nn.Linear(1, sex_dim)
        self.head = nn.Sequential(nn.Linear(feat_dim + sex_dim, hidden),
                                  nn.ReLU(), nn.Linear(hidden, 1))

    def forward(self, x, male):
        f = torch.flatten(self.cnn(x), 1)  # (B,1,1) pooled maps -> vector
        s = torch.relu(self.sex(male.float().view(-1, 1)))
        return self.head(torch.cat([f, s], dim=1)).squeeze(1)


class SmallSexNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.cnn = torchvision.models.resnet18(weights="IMAGENET1K_V1")
        n = self.cnn.fc.in_features
        self.cnn.fc = nn.Identity()
        self.sex = nn.Linear(1, 16)
        self.head = nn.Sequential(nn.Linear(n + 16, 128), nn.ReLU(), nn.Linear(128, 1))

    def forward(self, x, male):
        s = torch.relu(self.sex(male.float().view(-1, 1)))
        return self.head(torch.cat([self.cnn(x), s], dim=1)).squeeze(1)


def count_params(model):
    return sum(p.numel() for p in model.parameters())


def bench_forward(model, device, size=256, batch=8, iters=10):
    model = model.to(device).eval()
    xb = torch.randn(batch, 3, size, size, device=device)
    mb = torch.randint(0, 2, (batch,), device=device).float()
    with torch.no_grad():
        for _ in range(3):  # warmup
            (model(xb, mb) if "NoSex" not in type(model).__name__ else model(xb)).sum()
        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(iters):
            (model(xb, mb) if "NoSex" not in type(model).__name__ else model(xb)).sum()
        if device.type == "cuda":
            torch.cuda.synchronize()
    return (time.perf_counter() - t0) / iters * 1000


def real_baselines(csv_path="data/boneage-training-dataset.csv"):
    if not os.path.exists(csv_path):
        return None
    import pandas as pd

    df = pd.read_csv(csv_path)
    df["male"] = df["male"].astype(float)
    overall = float((df.boneage - df.boneage.mean()).abs().mean())
    per_sex = df.groupby(df.male > 0.5)["boneage"].apply(
        lambda s: float((s - s.mean()).abs().mean()))
    return {"mean_mae": overall, "boys_mae": float(per_sex.get(True, float("nan"))),
            "girls_mae": float(per_sex.get(False, float("nan"))),
            "n": len(df), "mean": float(df.boneage.mean()), "std": float(df.boneage.std())}


def main():
    device = get_device()
    print(f"device: {device}")
    if device.type == "cuda":
        print(f"gpu: {torch.cuda.get_device_name(0)}")

    real = real_baselines()
    if real:
        print(f"real mean-baseline MAE={real['mean_mae']:.2f} "
              f"boys={real['boys_mae']:.2f} girls={real['girls_mae']:.2f} (n={real['n']})")
    else:
        print("no real CSV -> synthetic demo (mean-baseline theory ~= 0.8*41 = 32.8mo)")

    # synthetic targets ~ N(127, 41)
    rng = np.random.default_rng(42)
    y = rng.normal(127, 41, size=512).astype(np.float32)
    sex = rng.integers(0, 2, size=512).astype(np.float32)
    mean_pred = float(np.abs(y - y.mean()).mean())
    boys = y[sex == 1]
    girls = y[sex == 0]
    strat = float((np.abs(boys - boys.mean()).sum() + np.abs(girls - girls.mean()).sum()) / len(y))
    print(f"[baseline] mean MAE={mean_pred:.2f}mo | sex-stratified MAE={strat:.2f}mo")

    models = {"resnet50+sex (main)": BoneAgeNet(),
              "resnet50 no-sex (ablation)": NoSexNet(),
              "resnet18+sex (light)": SmallSexNet(),
              "convnext-tiny+sex (SOTA lit 3.68mo)": BackboneSexNet(
                  torchvision.models.convnext_tiny(
                      weights=torchvision.models.ConvNeXt_Tiny_Weights.IMAGENET1K_V1),
                  768),
              "efficientnet-b0+sex": BackboneSexNet(
                  torchvision.models.efficientnet_b0(
                      weights=torchvision.models.EfficientNet_B0_Weights.IMAGENET1K_V1),
                  1280),
              "vit-b16+sex (BAE-ViT style)": BackboneSexNet(
                  torchvision.models.vit_b_16(
                      weights=torchvision.models.ViT_B_16_Weights.IMAGENET1K_V1),
                  768)}
    for name, m in models.items():
        n_params = count_params(m) / 1e6
        size = 224 if name.startswith("vit-") else 256  # ViT asserts exact 224
        try:
            ms = bench_forward(m, device, size=size, batch=4)
            print(f"[model] {name}: params={n_params:.1f}M forward={ms:.1f}ms/batch4@{size}")
        except torch.cuda.OutOfMemoryError:
            print(f"[model] {name}: params={n_params:.1f}M OOM on 4GB GPU at batch4@256")
        finally:
            m.to("cpu")
            if device.type == "cuda":
                torch.cuda.empty_cache()

    # algo comparison: one synthetic z-score step
    xb = torch.randn(8, 3, 256, 256, device=device)
    mb = torch.randint(0, 2, (8,), device=device).float()
    torch.manual_seed(0)
    for loss_name, fn in [("MSE-on-z", nn.MSELoss()), ("L1-on-z", nn.L1Loss()),
                           ("SmoothL1-on-z", nn.SmoothL1Loss(beta=1.0))]:
        m = BoneAgeNet().to(device)
        opt = torch.optim.AdamW(m.parameters(), lr=1e-4)
        zt = torch.randn(8, device=device)
        opt.zero_grad()
        loss = fn(m(xb, mb), zt)
        loss.backward()
        opt.step()
        print(f"[algo] {loss_name}: one-step loss={loss.item():.4f} (trains on z, eval MAE in months)")


if __name__ == "__main__":
    main()

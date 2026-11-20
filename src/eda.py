"""EDA: bone-age distribution overall + boys vs girls. Run: python -m src.eda."""
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("data/boneage-training-dataset.csv")
df["male"] = df["male"].astype(float)
print(df.describe())
print(f"mean={df.boneage.mean():.1f} std={df.boneage.std():.1f}")
print(f"boys={int(df.male.sum())} girls={int((1-df.male).sum())}")

fig, ax = plt.subplots()
ax.hist(df[df.male == 1].boneage, bins=40, alpha=0.6, label="boys")
ax.hist(df[df.male == 0].boneage, bins=40, alpha=0.6, label="girls")
ax.set_xlabel("bone age (months)")
ax.legend()
fig.savefig("figs/target_by_sex.png", dpi=120)
print("wrote figs/target_by_sex.png")

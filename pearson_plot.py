import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)

# 強い正の相関
x1 = np.linspace(0, 10, 50)
y1 = 2 * x1 + np.random.normal(0, 1, 50)

# 強い負の相関
x2 = np.linspace(0, 10, 50)
y2 = -2 * x2 + np.random.normal(0, 1, 50)

# ほぼ無相関
x3 = np.linspace(0, 10, 50)
y3 = np.random.normal(0, 5, 50)

fig, axes = plt.subplots(1, 3, figsize=(15, 4))

axes[0].scatter(x1, y1, color='tab:blue')
axes[0].set_title("r ≈ +0.98")
axes[0].set_xlabel("X"); axes[0].set_ylabel("Y")

axes[1].scatter(x2, y2, color='tab:red')
axes[1].set_title("r ≈ -0.98")
axes[1].set_xlabel("X"); axes[1].set_ylabel("Y")

axes[2].scatter(x3, y3, color='tab:green')
axes[2].set_title("r ≈ 0.02")
axes[2].set_xlabel("X"); axes[2].set_ylabel("Y")

plt.tight_layout()
plt.show()

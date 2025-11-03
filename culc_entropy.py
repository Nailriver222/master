import cv2
import numpy as np
from PIL import Image
from skimage.measure import shannon_entropy
import matplotlib.pyplot as plt

def visualize_hue_excluding_black(image_path, sat_thresh=30, val_thresh=30):
    # 画像読み込み (BGR)
    img_bgr = cv2.imread(image_path)
    if img_bgr is None:
        raise FileNotFoundError(f"画像が見つかりません: {image_path}")

    # HSV変換
    img_hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(img_hsv)

    # 黒・白・灰色を除外するマスク
    valid_mask = (s > sat_thresh) & (v > val_thresh)

    # Hueチャンネルを彩度・明度最大にして再構成（視覚的再現用）
    hsv_vis = np.zeros_like(img_hsv)
    hsv_vis[:, :, 0] = h
    hsv_vis[:, :, 1] = 255
    hsv_vis[:, :, 2] = 255

    # 無効画素（黒など）は黒塗り
    hsv_vis[~valid_mask] = [0, 0, 0]

    # HSV → BGR → RGB（matplotlib用）
    bgr_vis = cv2.cvtColor(hsv_vis, cv2.COLOR_HSV2BGR)
    rgb_vis = cv2.cvtColor(bgr_vis, cv2.COLOR_BGR2RGB)

    # 可視化
    plt.figure(figsize=(8, 8))
    plt.imshow(rgb_vis)
    plt.title("Hue Channel (excluding black/gray/white)")
    plt.axis("off")
    plt.show()


def calculate_hue_entropy(image_path, hue_bins=180, show_hist=False):
    # 画像をBGRで読み込む
    img_bgr = cv2.imread(image_path)
    if img_bgr is None:
        raise FileNotFoundError(f"画像が見つかりません: {image_path}")
    
    # BGR -> HSV 変換
    img_hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(img_hsv)

    # --- 黒・白・灰色を除外するマスク作成 ---
    # 彩度と明度が一定以上のピクセルだけを残す
    sat_thresh = 30   # 彩度の閾値（0〜255）
    val_thresh = 30   # 明度の閾値（0〜255）
    valid_mask = (s > sat_thresh) & (v > val_thresh)

    # 有効ピクセルのHueのみ抽出
    hue_valid = h[valid_mask]

    if hue_valid.size == 0:
        raise ValueError("有効なHueを持つピクセルが見つかりません。")

    # 統計情報
    mean_hue = np.mean(hue_valid)
    std_hue = np.std(hue_valid)
    
    # 輝度のエントロピーを計算（グレースケールで）
    img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    entropy = shannon_entropy(img_gray)

    # Hueヒストグラム作成（黒除外済み）
    hist, bins = np.histogram(hue_valid, bins=hue_bins, range=(0, 180), density=True)

    # エントロピー計算（ゼロ除外）
    hist_nonzero = hist[hist > 0]
    hue_entropy = -np.sum(hist_nonzero * np.log2(hist_nonzero))

    # === ヒストグラム表示 ===
    if show_hist:
        plt.figure(figsize=(10, 5))
        plt.bar(bins[:-1], hist, width=1, color='orange', edgecolor='black')
        plt.title("Hue Histogram (excluding black/gray/white)")
        plt.xlabel("Hue value (0–179)")
        plt.ylabel("Normalized Frequency")
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.show()

    return {
        "mean_hue": mean_hue,
        "std_hue": std_hue,
        "entropy": entropy,
        "hue_entropy": hue_entropy,
        "num_valid_pixels": hue_valid.size,
        "total_pixels": h.size
    }

# --- 使用例 ---

if __name__ == "__main__":
    image_path = "C:\\Users\\nailr\\VScode Projects\\master\\output\\output_cost\\temp_images\\generated_image_cycle_53.png"
    try:
        visualize_hue_excluding_black(image_path, sat_thresh=30, val_thresh=30)
        results = calculate_hue_entropy(image_path, hue_bins=180, show_hist=True)
        print(f"平均色相 (Mean Hue): {results['mean_hue']:.2f}")
        print(f"色相の標準偏差 (Hue Std): {results['std_hue']:.2f}")
        print(f"輝度のエントロピー (Entropy): {results['entropy']:.4f}")
        print(f"色相のエントロピー (Hue Entropy): {results['hue_entropy']:.4f}")
        print(f"有効ピクセル数: {results['num_valid_pixels']} / {results['total_pixels']}")
    except Exception as e:
        print("エラー:", e)

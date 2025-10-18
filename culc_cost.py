import cv2
import numpy as np
from PIL import Image
from skimage.measure import shannon_entropy

def calculate_hue_entropy(image_path, hue_bins=180):
    # 画像をBGRで読み込む
    img_bgr = cv2.imread(image_path)
    if img_bgr is None:
        raise FileNotFoundError(f"画像が見つかりません: {image_path}")
    
    # BGR -> HSV 変換
    img_hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    
    # HSVのHueチャネルのみ抽出（0〜179の値）
    hue_channel = img_hsv[:, :, 0]
    
    # Hueの統計情報
    mean_hue = np.mean(hue_channel)
    std_hue = np.std(hue_channel)
    
    # 輝度エントロピーを計算（グレースケールに変換してから）
    img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    entropy = shannon_entropy(img_gray)

    # ヒストグラムを作成（0〜179の範囲でビン数指定）
    hist, _ = np.histogram(hue_channel, bins=hue_bins, range=(0, 180), density=True)

    # 色相エントロピーを計算（ゼロ除外）
    hist_nonzero = hist[hist > 0]
    hue_entropy = -np.sum(hist_nonzero * np.log2(hist_nonzero))

    return {
        "mean_hue": mean_hue,
        "std_hue": std_hue,
        "entropy": entropy,
        "hue_entropy": hue_entropy
    }

    x

# 使用例
if __name__ == "__main__":
    image_path = "C:\\Users\\nailr\\VScode Projects\\master\\output\\output_rgb\\final_combined_image.png" 
    try:
        results = calculate_hue_entropy(image_path, hue_bins=180)
        print(f"平均色相 (Mean Hue): {results['mean_hue']:.2f}")
        print(f"色相の標準偏差 (Hue Std): {results['std_hue']:.2f}")
        print(f"輝度のエントロピー (Entropy): {results['entropy']:.4f}")
        print(f"色相のエントロピー (Hue Entropy): {results['hue_entropy']:.4f}")
    except Exception as e:
        print("エラー:", e)

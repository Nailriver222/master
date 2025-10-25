import os
import random
import numpy as np
from PIL import Image
import cv2
from skimage.measure import shannon_entropy
import pandas as pd
import csv

# ======== ユーザー設定 ========
image_files = [
    'img\\left_w.bmp',
    'img\\center_w.bmp',
    'img\\right_w.bmp'
]

output_folder = 'C:\\Users\\nailr\\VScode Projects\\master\\output\\output_cost'
temp_folder = os.path.join(output_folder, "temp_images")

use_random_color = True  # ランダムカラー
specified_colors = [
    (255, 0, 0),
    (255, 0, 0),
    (255, 0, 0),
]

# --- CSV & 定数設定 ---
audio_csv_path = "C:\\Users\\nailr\\VScode Projects\\master\\output\\output_features\\pearson\\2018\\going_down\\audio_features.csv"
video_csv_path = "C:\\Users\\nailr\\VScode Projects\\master\\output\\output_features\\pearson\\2018\\going_down\\video_features.csv"
audio_column = "rms_mean"
video_column = "h_mean"
alpha1 = 1
alpha2 = 1
# ============================

# ----------- 画像処理関数 -----------
def ensure_output_dir(path):
    os.makedirs(path, exist_ok=True)

def load_grayscale_image(path):
    return Image.open(path).convert('L')

def create_color_image(size, color):
    return Image.new('RGB', size, color)

def multiply_images(gray_img, color_img):
    gray_np = np.array(gray_img) / 255.0
    color_np = np.array(color_img)
    result_np = (color_np * gray_np[..., None]).astype(np.uint8)
    return Image.fromarray(result_np)

def add_all_images_hdr(image_list):
    result = np.zeros_like(np.array(image_list[0]), dtype=np.float32)
    for img in image_list:
        result += np.array(img, dtype=np.float32)
    max_val = np.max(result)
    if max_val > 0:
        result = (result / max_val) * 255
    return Image.fromarray(np.clip(result, 0, 255).astype(np.uint8))

def get_color(index):
    if use_random_color:
        return tuple(random.randint(0, 255) for _ in range(3))
    else:
        return specified_colors

def process_images(image_paths):
    colored_images = []
    rgb_values = []
    for i, path in enumerate(image_paths):
        gray_img = load_grayscale_image(path)
        color = get_color(i)
        color_img = create_color_image(gray_img.size, color)
        colored_img = multiply_images(gray_img, color_img)
        colored_images.append(colored_img)
        rgb_values.append(color)
    final_image = add_all_images_hdr(colored_images)
    return final_image, rgb_values

# ----------- エントロピー計算 -----------
def calculate_hue_entropy(image_path, hue_bins=180):
    img_bgr = cv2.imread(image_path)
    if img_bgr is None:
        raise FileNotFoundError(f"画像が見つかりません: {image_path}")

    # 輝度エントロピー
    img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    entropy_gray = shannon_entropy(img_gray)

    # 色相エントロピー
    img_hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    hue_channel = img_hsv[:, :, 0]
    hist, _ = np.histogram(hue_channel, bins=hue_bins, range=(0,180), density=True)
    hist_nonzero = hist[hist > 0]
    hue_entropy = -np.sum(hist_nonzero * np.log2(hist_nonzero))
    return entropy_gray, hue_entropy

# ----------- O1計算 -----------
def calculate_O1(audio_csv_path, video_csv_path, audio_col, video_col):
    audio_df = pd.read_csv(audio_csv_path)
    video_df = pd.read_csv(video_csv_path)
    audio_mean = audio_df[audio_col].mean()
    video_mean = video_df[video_col].mean()
    O1 = abs(audio_mean - video_mean)
    return O1

# ----------- Cost計算 -----------
def compute_cost(O1, entropy_gray, hue_entropy, alpha1, alpha2):
    cost_gray = alpha1*O1 + alpha2*entropy_gray
    cost_hue = alpha1*O1 + alpha2*hue_entropy
    return cost_gray, cost_hue

# ----------- 最適化サイクル -----------
def optimize_images(num_cycles=10):
    ensure_output_dir(output_folder)
    ensure_output_dir(temp_folder)

    best_gray_cost = float('inf')
    best_hue_cost = float('inf')
    best_gray_info = None
    best_hue_info = None

    csv_path = os.path.join(output_folder, "cycle_results.csv")
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(["Cycle","RGB_values","O1","Entropy_gray","Hue_entropy","Cost_gray","Cost_hue"])

        for cycle in range(1, num_cycles+1):
            final_image, rgb_values = process_images(image_files)

            # サイクル画像をtempフォルダに保存
            temp_image_path = os.path.join(temp_folder, f"generated_image_cycle_{cycle}.png")
            final_image.save(temp_image_path)

            # エントロピー計算
            entropy_gray, hue_entropy = calculate_hue_entropy(temp_image_path)

            # O1計算
            O1 = calculate_O1(audio_csv_path, video_csv_path, audio_column, video_column)

            # Cost計算
            cost_gray, cost_hue = compute_cost(O1, entropy_gray, hue_entropy, alpha1, alpha2)

            # CSV記録
            writer.writerow([cycle, rgb_values, O1, entropy_gray, hue_entropy, cost_gray, cost_hue])

            # 最小Cost更新
            if cost_gray < best_gray_cost:
                best_gray_cost = cost_gray
                best_gray_info = {
                    "cycle": cycle, "O1": O1, "entropy_gray": entropy_gray,
                    "hue_entropy": hue_entropy, "cost": cost_gray,
                    "RGB_values": rgb_values,
                    "image": final_image.copy()
                }

            if cost_hue < best_hue_cost:
                best_hue_cost = cost_hue
                best_hue_info = {
                    "cycle": cycle, "O1": O1, "entropy_gray": entropy_gray,
                    "hue_entropy": hue_entropy, "cost": cost_hue,
                    "RGB_values": rgb_values,
                    "image": final_image.copy()
                }

    # 最終結果画像をoutputに保存
    best_gray_info["image"].save(os.path.join(output_folder, "best_gray_cost.png"))
    best_hue_info["image"].save(os.path.join(output_folder, "best_hue_cost.png"))

    print(f"\n✅ 全Cycle完了。結果CSV: {csv_path}")
    return best_gray_info, best_hue_info

# ======== 実行 ========
if __name__ == "__main__":
    best_gray, best_hue = optimize_images(num_cycles=5)
    
    print("\n====== 最終結果（見やすく整理） ======")
    print("■ 輝度基準最小Cost")
    print(f"Cycle: {best_gray['cycle']}")
    print(f"RGB値: {best_gray['RGB_values']}")
    print(f"O1: {best_gray['O1']:.4f}")
    print(f"輝度エントロピー: {best_gray['entropy_gray']:.4f}")
    print(f"色相エントロピー: {best_gray['hue_entropy']:.4f}")
    print(f"Cost_gray: {best_gray['cost']:.4f}")
    
    print("\n■ 色相基準最小Cost")
    print(f"Cycle: {best_hue['cycle']}")
    print(f"RGB値: {best_hue['RGB_values']}")
    print(f"O1: {best_hue['O1']:.4f}")
    print(f"輝度エントロピー: {best_hue['entropy_gray']:.4f}")
    print(f"色相エントロピー: {best_hue['hue_entropy']:.4f}")
    print(f"Cost_hue: {best_hue['cost']:.4f}")
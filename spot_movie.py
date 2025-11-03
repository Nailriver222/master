import os
import glob
import cv2
import numpy as np
import pandas as pd
from moviepy.editor import ImageSequenceClip, VideoFileClip

# ===== ユーザー入力 =====
name_input = input("名前を入力してください: ")

csv_base_dir = "C:\\Users\\nailr\\VScode Projects\\master\\output\\output_features\\pearson\\2018"
video_base_dir = "C:\\Users\\nailr\\VScode Projects\\master\\target_movie\\2018"
output_video_path = f"C:\\Users\\nailr\\VScode Projects\\master\\output\\output_spot\\{name_input}_output.mp4"
input_image_path = "img\\center_w.bmp"
brightness_col = "flux_mean"
hue_col = "rms_mean"
brightness_multiplier = 5.0
hue_offset = 60
fps = 25

# ===== CSV検索 =====
csv_path = os.path.join(csv_base_dir, name_input, "audio_features.csv")
if not os.path.exists(csv_path):
    raise FileNotFoundError(f"CSVファイルが見つかりません: {csv_path}")

# ===== 元動画検索 =====
video_pattern = os.path.join(video_base_dir, f"{name_input}.*")  # mp4, movなど
video_files = glob.glob(video_pattern)
if not video_files:
    raise FileNotFoundError(f"元動画が見つかりません: {video_pattern}")
video_path = video_files[0]

# ===== CSV読み込み =====
df = pd.read_csv(csv_path)
brightness_values = df[brightness_col].values
hue_values = df[hue_col].values

brightness_norm = (brightness_values - brightness_values.min()) / max(brightness_values.max() - brightness_values.min(), 1e-6)
hue_norm = (hue_values - hue_values.min()) / max(hue_values.max() - hue_values.min(), 1e-6) * 179

# ===== 元画像読み込み =====
img = cv2.imread(input_image_path)
if img is None:
    raise FileNotFoundError(f"画像が見つかりません: {input_image_path}")
if len(img.shape) == 2:
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
height, width = img.shape[:2]

# ===== フレーム生成 =====
frames = []
for b_factor, h_val in zip(brightness_norm, hue_norm):
    frame = img.astype(np.float32) / 255.0
    frame = frame * (b_factor * brightness_multiplier)
    frame = np.clip(frame, 0, 1)

    hue_val = (h_val + hue_offset) % 180
    hsv_color = np.zeros((height, width, 3), dtype=np.uint8)
    hsv_color[:, :, 0] = int(hue_val)
    hsv_color[:, :, 1] = 255
    hsv_color[:, :, 2] = 255
    color_img = cv2.cvtColor(hsv_color, cv2.COLOR_HSV2BGR).astype(np.float32) / 255.0

    frame = frame * color_img
    frame = np.clip(frame * 255, 0, 255).astype(np.uint8)
    frames.append(frame[:, :, ::-1])  # BGR→RGBに変換

# ===== 動画作成 =====
clip = ImageSequenceClip(frames, fps=fps)

# ===== 音声合成 =====
original_audio = VideoFileClip(video_path).audio
clip = clip.set_audio(original_audio)
clip.write_videofile(output_video_path, codec="libx264", audio_codec="aac")

print("動画生成完了:", output_video_path)

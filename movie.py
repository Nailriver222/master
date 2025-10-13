import os
import cv2
import pandas as pd
import numpy as np
import subprocess

# === 設定 ===
SEARCH_BASE = 'C:/Users/nailr/北大/研究室/研究/修士/ステージ照明'
MOVIE_DIR = os.path.join(SEARCH_BASE, 'target_movie/2018')
FEATURE_CSV_DIR = os.path.join(SEARCH_BASE, 'output/output_features/pearson/2018')
OUTPUT_VIDEO_DIR = os.path.join(SEARCH_BASE, 'generated_videos/2018')

# 色相変換に使う3枚の画像パスを絶対パスで指定してください
INPUT_IMAGES = [
    'img\\left.bmp',
    'img\\center.bmp',
    'img\\right.bmp',
]

# 各画像に使う特徴量と適用するHSV成分を指定
# 'H' = 色相, 'S' = 彩度, 'V' = 明度
IMAGE_FEATURE_SETTINGS = [
    {'feature': 'rms_mean', 'component': 'H'},
    {'feature': 'flux_mean', 'component': 'H'},
    {'feature': 'rms_mean', 'component': 'H'},
]

FPS = 25  # 元動画と合わせて
TEMP_VIDEO_NAME = 'temp_no_audio.mp4'
FINAL_VIDEO_NAME = 'final_output.mp4'

# === ユーザー入力から対象フォルダ検索 ===
search_term = input("検索キーワードを入力してください: ").strip()

def find_target_video_and_feature_folder():
    for file in os.listdir(MOVIE_DIR):
        if search_term in file and file.lower().endswith('.mp4'):
            video_path = os.path.join(MOVIE_DIR, file)
            name_no_ext = os.path.splitext(file)[0]
            feature_csv_folder = os.path.join(FEATURE_CSV_DIR, name_no_ext)
            feature_csv = os.path.join(feature_csv_folder, 'audio_features.csv')
            output_video_folder = os.path.join(OUTPUT_VIDEO_DIR, name_no_ext)
            if os.path.exists(feature_csv):
                return video_path, output_video_folder, feature_csv
    return None, None, None

video_path, output_video_folder, feature_csv = find_target_video_and_feature_folder()
os.makedirs(output_video_folder, exist_ok=True)

if not video_path:
    print(f"❌ 対象が見つかりませんでした: {search_term}")
    exit()

print(f"✅ 対象動画: {video_path}")
print(f"✅ 特徴量CSV: {feature_csv}")

# === 音響特徴量を読み込む ===
df = pd.read_csv(feature_csv)

# 各画像用の変換データ（255スケール）を保存
value_sequences = []

for setting in IMAGE_FEATURE_SETTINGS:
    col = setting['feature']
    if col not in df.columns:
        print(f"❌ 指定の特徴量列が見つかりません: {col}")
        exit()

    raw = df[col].values
    if len(raw) == 0:
        print(f"❌ 特徴量列 '{col}' にデータがありません。")
        exit()

    log = np.log1p(raw)
    norm = (log - np.min(log)) / (np.max(log) - np.min(log))
    values = (norm * 255).astype(np.uint8)
    value_sequences.append(values)

# === 3枚の画像を読み込み（リサイズなし） ===
imgs = []
for path in INPUT_IMAGES:
    img = cv2.imread(path)
    if img is None:
        print(f"❌ 入力画像が見つかりません: {path}")
        exit()
    imgs.append(img.astype(np.float32) / 255.0)

# 動画サイズは最初の画像のサイズに合わせる
height, width, _ = imgs[0].shape

# === 動画書き込み準備 ===
temp_video_path = os.path.join(output_video_folder, TEMP_VIDEO_NAME)
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
video_writer = cv2.VideoWriter(temp_video_path, fourcc, FPS, (width, height))

def hsv_shift(img, value, component):
    """
    img: 0〜1のfloat32 BGR画像
    value: 0〜255の値（1つのスカラー）
    component: 'H', 'S', 'V' のどれを変更するか
    """
    hsv = cv2.cvtColor((img * 255).astype(np.uint8), cv2.COLOR_BGR2HSV)
    
    if component == 'H':
        hsv[:, :, 0] = np.full_like(hsv[:, :, 0], int(value))
    elif component == 'S':
        hsv[:, :, 1] = np.full_like(hsv[:, :, 1], int(value))
    elif component == 'V':
        hsv[:, :, 2] = np.full_like(hsv[:, :, 2], int(value))
    else:
        raise ValueError(f"Unknown component: {component}")
    
    bgr_shifted = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR).astype(np.float32) / 255.0
    return bgr_shifted

# === フレームごとに色相変換＆合成 ===
frame_count = len(value_sequences[0])  # 全列が同じ長さであること前提

for i in range(frame_count):
    shifted_imgs = [
        hsv_shift(
            imgs[j],                 # 元画像
            value_sequences[j][i],  # このフレームの値
            IMAGE_FEATURE_SETTINGS[j]['component']  # 'H', 'S', 'V'
        )
        for j in range(3)
    ]
    combined = sum(shifted_imgs)
    combined = np.clip(combined, 0, 1)
    frame = (combined * 255).astype(np.uint8)
    video_writer.write(frame)


video_writer.release()
print(f"🎥 無音動画を保存: {temp_video_path}")

# === FFmpegで音声を合成 ===
final_output_path = os.path.join(output_video_folder, FINAL_VIDEO_NAME)

ffmpeg_command = [
    'ffmpeg',
    '-y',
    '-i', temp_video_path,
    '-i', video_path,
    '-c:v', 'copy',
    '-map', '0:v:0',
    '-map', '1:a:0',
    '-shortest',
    final_output_path
]

print("🎧 音声を合成中...")
subprocess.run(ffmpeg_command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

print(f"✅ 最終動画を保存: {final_output_path}")

# === 一時動画ファイルを削除 ===
if os.path.exists(temp_video_path):
    os.remove(temp_video_path)
    print(f"🗑️ 一時ファイルを削除: {temp_video_path}")
else:
    print("⚠️ 一時ファイルが見つかりませんでした。削除スキップ。")

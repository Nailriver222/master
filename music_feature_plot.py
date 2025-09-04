import os
import cv2
import numpy as np
import librosa
import pandas as pd
import matplotlib.pyplot as plt
import moviepy.editor as mp
from scipy.signal import correlate

# === パラメータ ===
VIDEO_PATH = "C:\\Users\\nailr\北大\研究室\研究\修士\ステージ照明\\target_movie\\Eat_The_Rich.mp4"
OUTPUT_DIR = 'output_features'
AUDIO_TEMP_WAV = 'temp_audio.wav'
os.makedirs(OUTPUT_DIR, exist_ok=True)

FRAME_INTERVAL = 1
FRAME_LENGTH = 2048
HOP_LENGTH = 512
WINDOW_SIZE = 10
MA_STEP = 1
AUDIO_SR = 22050  # librosa sampling rate

# === 音声抽出 (.wavに変換) ===
def extract_audio_from_video(video_path, output_wav='temp_audio.wav'):
    clip = mp.VideoFileClip(video_path)
    clip.audio.write_audiofile(output_wav, codec='pcm_s16le')
    return output_wav

# === 音響特徴量抽出 ===
def extract_audio_features_from_wav(wav_path, sr=22050):
    y, _ = librosa.load(wav_path, sr=sr)
    rms = librosa.feature.rms(y=y, frame_length=FRAME_LENGTH, hop_length=HOP_LENGTH)[0]
    centroid = librosa.feature.spectral_centroid(y=y, sr=sr, hop_length=HOP_LENGTH)[0]
    bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr, hop_length=HOP_LENGTH)[0]
    rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr, hop_length=HOP_LENGTH)[0]
    S = np.abs(librosa.stft(y, hop_length=HOP_LENGTH))
    flux = np.sqrt(np.sum(np.diff(S, axis=1)**2, axis=0))
    return {
        'rms': rms,
        'centroid': centroid,
        'bandwidth': bandwidth,
        'rolloff': rolloff,
        'flux': flux
    }

# === 映像特徴量抽出（平均HSV） ===
def extract_video_features(video_path, frame_interval=1):
    cap = cv2.VideoCapture(video_path)
    hsv_values = {'h': [], 's': [], 'v': []}
    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx % frame_interval == 0:
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            h, s, v = cv2.split(hsv)
            hsv_values['h'].append(np.mean(h))
            hsv_values['s'].append(np.mean(s))
            hsv_values['v'].append(np.mean(v))
        frame_idx += 1
    cap.release()
    return hsv_values

# === 移動平均 & 標準偏差計算 + 長さ揃え ===
def moving_stats(x, window_size, step):
    mean_list, std_list = [], []
    for i in range(0, len(x) - window_size + 1, step):
        window = x[i:i + window_size]
        mean_list.append(np.mean(window))
        std_list.append(np.std(window))
    return np.array(mean_list), np.array(std_list)

def align_feature_lengths(feature_dict):
    # 各特徴量の長さを確認し、最小長に揃える
    min_len = min(len(v) for v in feature_dict.values())
    aligned = {k: v[:min_len] for k, v in feature_dict.items()}
    return aligned

# === 相互相関とラグ検出 ===
def compute_cross_correlation(x, y, sr):
    # 標準化
    x = (x - np.mean(x)) / (np.std(x) + 1e-8)
    y = (y - np.mean(y)) / (np.std(y) + 1e-8)
    # 相互相関（フル）
    corr = correlate(x, y, mode='full')
    lags = np.arange(-len(y) + 1, len(x))
    # ✅ 正規化（相関係数化）
    corr /= len(x)  # or len(y), 一般的には元の系列長で割る
    # ピークのラグ検出
    max_idx = np.argmax(np.abs(corr))
    max_lag = lags[max_idx]
    lag_time = max_lag / sr
    return corr, lags, max_lag, lag_time

# === メイン処理 ===
def main():
    print("🔊 音声抽出中...")
    wav_path = extract_audio_from_video(VIDEO_PATH, AUDIO_TEMP_WAV)

    print("📈 音響特徴量抽出中...")
    audio_features = extract_audio_features_from_wav(wav_path, sr=AUDIO_SR)

    print("🎥 映像特徴量抽出中...")
    video_features = extract_video_features(VIDEO_PATH, frame_interval=FRAME_INTERVAL)

    print("📊 移動統計量計算中...")
    audio_stats = {}
    for k, v in audio_features.items():
        mean, std = moving_stats(v, WINDOW_SIZE, MA_STEP)
        audio_stats[k + '_mean'] = mean
        audio_stats[k + '_std'] = std
    audio_stats = align_feature_lengths(audio_stats)  # ✅ 長さ揃え

    video_stats = {}
    for k, v in video_features.items():
        mean, std = moving_stats(np.array(v), WINDOW_SIZE, MA_STEP)
        video_stats[k + '_mean'] = mean
        video_stats[k + '_std'] = std
    video_stats = align_feature_lengths(video_stats)  # ✅ 長さ揃え

    print("💾 CSV出力中...")
    pd.DataFrame(audio_stats).to_csv(os.path.join(OUTPUT_DIR, 'audio_features.csv'), index=False)
    pd.DataFrame(video_stats).to_csv(os.path.join(OUTPUT_DIR, 'video_features.csv'), index=False)

    print("🔄 相互相関計算中...")
    time_resolution = HOP_LENGTH / AUDIO_SR
    results = []

    for ak, av in audio_stats.items():
        for vk, vv in video_stats.items():
            min_len = min(len(av), len(vv))
            x = av[:min_len]
            y = vv[:min_len]
            corr, lags, max_lag, lag_time = compute_cross_correlation(x, y, 1 / time_resolution)
            results.append((ak, vk, np.max(np.abs(corr)), lag_time))

            # グラフ保存
            plt.figure(figsize=(8, 4))
            plt.plot(lags * time_resolution, corr)
            plt.title(f'Cross-Correlation: {ak} vs {vk}')
            plt.xlabel('Lag (seconds)')
            plt.ylabel('Correlation')
            plt.grid()
            plt.tight_layout()
            fname = f'{ak}_x_{vk}.png'.replace('/', '_')
            plt.savefig(os.path.join(OUTPUT_DIR, fname))
            plt.close()

    # 結果表示
    print("\n=== 相互相関の結果 ===")
    for ak, vk, corr_val, lag_sec in results:
        print(f"{ak} × {vk} → 相関: {corr_val:.4f}, ラグ: {lag_sec:.3f} 秒")

    # 一時ファイル削除
    if os.path.exists(wav_path):
        os.remove(wav_path)

if __name__ == '__main__':
    main()

import os
import cv2
import numpy as np
import librosa
import pandas as pd
import matplotlib.pyplot as plt
import moviepy.editor as mp
from scipy.signal import correlate
from scipy.stats import pearsonr

# === パラメータ ===
SEARCH_DIR = 'C:\\Users\\nailr\北大\研究室\研究\修士\ステージ照明\\target_movie\\2018'
OUTPUT_BASE_DIR = 'C:\\Users\\nailr\北大\研究室\研究\修士\ステージ照明\\output\\output_features\\pearson\\2018'
AUDIO_TEMP_WAV = 'temp_audio.wav'
os.makedirs(OUTPUT_BASE_DIR, exist_ok=True)

FRAME_INTERVAL = 1
FRAME_LENGTH = 2048
HOP_LENGTH = 512
WINDOW_SIZE = 10
MA_STEP = 1
AUDIO_SR = 22050

# 対象の動画拡張子
video_extensions = ['.mp4']

def find_and_create_output_folder(search_string):
    # フォルダ内のすべてのファイルをチェック
    for filename in os.listdir(SEARCH_DIR):
        file_path = os.path.join(SEARCH_DIR, filename)
        
        # ファイルかつ動画ファイルであるかを確認
        if os.path.isfile(file_path) and any(filename.lower().endswith(ext) for ext in video_extensions):
            if search_string in filename:
                found = False
                # 出力用フォルダ名を決定（拡張子なしのファイル名）
                name_without_ext = os.path.splitext(filename)[0]
                output_folder_path = os.path.join(OUTPUT_BASE_DIR, name_without_ext)
                
                # 出力フォルダを作成（既に存在していなければ）
                os.makedirs(output_folder_path, exist_ok=True)
                print(f"✅ 出力フォルダを作成しました: {output_folder_path}")

                return file_path, output_folder_path
    
    print(f"⚠️ 一致する動画ファイルが見つかりませんでした: '{search_string}'")
    return None, None

# === 音声抽出 (.wavに変換) ===
def extract_audio_from_video(video_path, output_wav='temp_audio.wav'):
    clip = mp.VideoFileClip(video_path)
    clip.audio.write_audiofile(output_wav, codec='pcm_s16le')
    return output_wav

# === 音響特徴量抽出 ===
def extract_audio_features_resampled(wav_path, sr, video_fps, video_duration=None):
    import numpy as np
    from scipy.interpolate import interp1d

    y, _ = librosa.load(wav_path, sr=sr)

    # 音響特徴量
    features = {
        'rms': librosa.feature.rms(y=y, frame_length=FRAME_LENGTH, hop_length=HOP_LENGTH)[0],
        'centroid': librosa.feature.spectral_centroid(y=y, sr=sr, hop_length=HOP_LENGTH)[0],
        'bandwidth': librosa.feature.spectral_bandwidth(y=y, sr=sr, hop_length=HOP_LENGTH)[0],
        'rolloff': librosa.feature.spectral_rolloff(y=y, sr=sr, hop_length=HOP_LENGTH)[0],
        'flux': np.sqrt(np.sum(np.diff(np.abs(librosa.stft(y, hop_length=HOP_LENGTH)), axis=1)**2, axis=0))
    }

    # 映像フレームに対応する時間軸
    if video_duration is None:
        video_duration = librosa.get_duration(y=y, sr=sr)
    num_video_frames = int(np.floor(video_duration * video_fps))
    video_times = np.arange(num_video_frames) / video_fps

    # 各特徴量をvideo_timesに補間
    resampled = {}
    for name, values in features.items():
        audio_times = np.arange(len(values)) * HOP_LENGTH / sr
        interp = interp1d(audio_times, values, kind='linear', fill_value="extrapolate")
        resampled[name] = interp(video_times)

    return resampled

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

# === 移動平均のみ計算 + 長さ揃え ===
def moving_avg(x, window_size, step):
    mean_list = []
    for i in range(0, len(x) - window_size + 1, step):
        window = x[i:i + window_size]
        mean_list.append(np.mean(window))
    return np.array(mean_list)

def align_feature_lengths(feature_dict):
    min_len = min(len(v) for v in feature_dict.values())
    aligned = {k: v[:min_len] for k, v in feature_dict.items()}
    return aligned

# === 相互相関とラグ検出 ===
def compute_cross_correlation(x, y, sr, max_lag_sec=1.0):
    x = (x - np.mean(x)) / (np.std(x) + 1e-8)
    y = (y - np.mean(y)) / (np.std(y) + 1e-8)
    corr = correlate(x, y, mode='full')
    lags = np.arange(-len(y) + 1, len(x))
    corr /= len(x)
    
    # 秒単位に変換したラグ
    lag_times = lags / sr

    # ±1秒以内の範囲をマスク
    valid_idx = np.where(np.abs(lag_times) <= max_lag_sec)[0]

    # 制限範囲内でのピーク検出
    restricted_corr = corr[valid_idx]
    restricted_lags = lags[valid_idx]
    restricted_lag_times = lag_times[valid_idx]

    max_idx = np.argmax(np.abs(restricted_corr))
    max_lag = restricted_lags[max_idx]
    lag_time = restricted_lag_times[max_idx]

    return corr, lags, max_lag, lag_time

def compute_normalized_cross_correlation(x, y, sr, max_lag_sec=1.0):
    """
    正規化された相互相関係数を計算。
    ラグは ±max_lag_sec の範囲内で最大値を検出。

    Parameters:
        x, y : np.ndarray
            中心化された信号（平均0）
        sr : float
            サンプリングレート（秒あたりのフレーム数）
        max_lag_sec : float
            ピークを探すラグの最大秒数（±）

    Returns:
        max_r : float
            最大の正規化相互相関係数（±1の範囲）
        best_lag_sec : float
            そのときのラグ（秒）
        corr : np.ndarray
            全体の相互相関配列（正規化前）
        lags : np.ndarray
            ラグ（フレーム単位）
    """
    # 中心化（平均を引く）
    x = x - np.mean(x)
    y = y - np.mean(y)

    # 相互相関（フルモード）
    corr = correlate(x, y, mode='full')
    lags = np.arange(-len(y)+1, len(x))

    # 正規化の分母（自己相関のラグ0）
    norm_factor = np.sqrt(np.sum(x**2) * np.sum(y**2))
    norm_corr = corr / (norm_factor + 1e-8)

    # 秒単位のラグを取得
    lag_times = lags / sr

    # ±max_lag_sec の範囲内を抽出
    valid_idx = np.where(np.abs(lag_times) <= max_lag_sec)[0]
    restricted_corr = norm_corr[valid_idx]
    restricted_lags = lag_times[valid_idx]

    # 最大の相関係数とそのラグ
    max_idx = np.argmax(np.abs(restricted_corr))
    max_r = restricted_corr[max_idx]
    best_lag_sec = restricted_lags[max_idx]

    return max_r, best_lag_sec, norm_corr, lag_times


# === メイン処理 ===
def main():
    user_input = input("検索する動画ファイル名の一部を入力してください: ")
    video_path, output_dir = find_and_create_output_folder(user_input.strip())

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    video_duration = frame_count / fps
    cap.release()

    if video_path is None:
        print("❌ 対象ファイルが見つからなかったため、処理を終了します。")
        return

    print("🔊 音声抽出中...")
    wav_path = extract_audio_from_video(video_path, AUDIO_TEMP_WAV)

    print("📈 音響特徴量抽出中...")
    audio_features = extract_audio_features_resampled(wav_path, sr=AUDIO_SR, video_fps=fps, video_duration=video_duration)

    print("🎥 映像特徴量抽出中...")
    video_features = extract_video_features(video_path, frame_interval=FRAME_INTERVAL)

    print("📊 移動平均計算中...")
    audio_stats = {}
    for k, v in audio_features.items():
        mean = moving_avg(v, WINDOW_SIZE, MA_STEP)
        audio_stats[k + '_mean'] = mean
    audio_stats = align_feature_lengths(audio_stats)

    video_stats = {}
    for k, v in video_features.items():
        mean = moving_avg(np.array(v), WINDOW_SIZE, MA_STEP)
        video_stats[k + '_mean'] = mean
    video_stats = align_feature_lengths(video_stats)

    print("💾 CSV出力中...")
    pd.DataFrame(audio_stats).to_csv(os.path.join(output_dir, 'audio_features.csv'), index=False)
    pd.DataFrame(video_stats).to_csv(os.path.join(output_dir, 'video_features.csv'), index=False)

    print("🔄 相互相関計算中...")
    time_resolution = HOP_LENGTH / AUDIO_SR
    results = []

    for ak, av in audio_stats.items():
        for vk, vv in video_stats.items():
            min_len = min(len(av), len(vv))
            x = av[:min_len]
            y = vv[:min_len]
            corr, lags, max_lag, lag_time = compute_cross_correlation(x, y, 1 / time_resolution, max_lag_sec=1.0)
            rms_corr = np.sqrt(np.mean(corr**2))
            results.append((ak, vk, np.max(np.abs(corr)), lag_time, rms_corr))


            plt.figure(figsize=(8, 4))
            plt.plot(lags * time_resolution, corr)
            plt.title(f'Cross-Correlation: {ak} vs {vk}')
            plt.xlabel('Lag (seconds)')
            plt.ylabel('Correlation')
            plt.grid()
            plt.tight_layout()
            fname = f'{ak}_x_{vk}.png'.replace('/', '_')
            plt.savefig(os.path.join(output_dir, fname))
            plt.close()

    print("\n=== 相互相関の結果 ===")
    for ak, vk, peak_corr, lag_sec, rms_corr in results:
        print(f"{ak} × {vk} → 最大相関: {peak_corr:.4f}, ラグ: {lag_sec:.3f} 秒, RMS相関: {rms_corr:.4f}")
    
    # 💾 結果をCSVに保存
    result_df = pd.DataFrame(results, columns=[
        'Audio Feature', 'Video Feature', 'Peak Correlation', 'Lag (sec)', 'RMS Correlation'
    ])
    result_df.to_csv(os.path.join(output_dir, 'cross_correlation_results.csv'), index=False)
    print(f"\n💾 相互相関の結果をCSVに保存しました: {os.path.join(output_dir, 'cross_correlation_results.csv')}")


    print("📐 ピアソン相関係数の計算中...")
    norm_corr_results = []

    for ak, av in audio_stats.items():
        for vk, vv in video_stats.items():
            min_len = min(len(av), len(vv))
            x = av[:min_len]
            y = vv[:min_len]
            r_norm, lag_sec, norm_corr, lag_times = compute_normalized_cross_correlation(
                x, y, sr=1 / time_resolution, max_lag_sec=1.0
            )
            norm_corr_results.append((ak, vk, r_norm))

    # 結果表示と保存
    print("\n=== ピアソン相関係数の結果 ===")
    for ak, vk, r_norm in norm_corr_results:
        print(f"{ak} × {vk} → r = {r_norm:.4f}")

    norm_df = pd.DataFrame(norm_corr_results, columns=[
        'Audio Feature', 'Video Feature', 'Pearson r'
    ])
    norm_df.to_csv(os.path.join(output_dir, 'pearson_correlation.csv'), index=False)
    print(f"\n💾 ピアソン相関係数の結果を保存しました。")



    if os.path.exists(wav_path):
        os.remove(wav_path)

if __name__ == '__main__':
    main()

import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt

# 分幀函數
def framing(signal, frame_size, hop_size):
    num_frames = 1 + int((len(signal) - frame_size) / hop_size)
    frames = np.zeros((num_frames, frame_size))
    for i in range(num_frames):
        start = i * hop_size
        frames[i, :] = signal[start:start + frame_size]
    return frames

# 漢明窗應用函數
def apply_hamming_window(frames):
    hamming_window = np.hamming(frames.shape[1])  # 根據幀大小生成漢明窗
    windowed_frames = frames * hamming_window
    return windowed_frames

# 提取 MFCC 特徵的函數
def extract_mfcc(windowed_frames, sr, n_mfcc=13):
    mfcc_features = []
    for frame in windowed_frames:
        # 計算該幀的 MFCC
        mfcc = librosa.feature.mfcc(y=frame, sr=sr, n_mfcc=n_mfcc)
        mfcc_features.append(mfcc.mean(axis=1))  # 計算 MFCC 平均值
    return np.array(mfcc_features)

# 主程式流程
# 讀取音頻檔案
audio_signal, sampling_rate = librosa.load('1.wav', sr=22050)

# 設置分幀參數
frame_size = int(0.5 * sampling_rate)  # 幀大小 50ms
hop_size = int(0.25 * sampling_rate)   # 幀間距 25ms

# 分幀處理
frames = framing(audio_signal, frame_size, hop_size)

# 對每個幀應用漢明窗
windowed_frames = apply_hamming_window(frames)

# 提取每個應用完漢明窗的幀的 MFCC 特徵
mfcc_features = extract_mfcc(windowed_frames, sampling_rate)

# 輸出 MFCC 特徵
print("MFCC Features Shape:", mfcc_features.shape)

# 繪製 MFCC 特徵的可視化
plt.figure(figsize=(10, 4))
librosa.display.specshow(mfcc_features.T, x_axis='time', sr=sampling_rate)
plt.colorbar()
plt.title('MFCC Features After Hamming Window')
plt.tight_layout()
plt.show()

import numpy as np
from scipy.io import wavfile
import matplotlib.pyplot as plt
import librosa

# 分幀的函數
def framing(signal, frame_size, hop_size):
    num_frames = 1 + int((len(signal) - frame_size) / hop_size)
    frames = np.zeros((num_frames, frame_size))
    
    for i in range(num_frames):
        start = i * hop_size
        frames[i, :] = signal[start:start + frame_size]
    
    return frames

# 漢明窗應用的函數
def apply_hamming_window(frames):
    hamming_window = np.hamming(frames.shape[1])  # 根據幀大小生成漢明窗
    windowed_frames = frames * hamming_window
    return windowed_frames

# 讀取音頻檔案 (假設是 mono 單聲道的 wav 檔)
audio_signal, sampling_rate = librosa.load('test.wav', sr=22050)

# 音頻參數設置
frame_size = int(0.05 * sampling_rate)  # 每幀大小為 25ms
hop_size = int(0.025 * sampling_rate)    # 幀間距為 10ms

# 分幀
frames = framing(audio_signal, frame_size, hop_size)

# 應用漢明窗
windowed_frames = apply_hamming_window(frames)

# 顯示第一個幀的原始波形與加窗後的波形
plt.figure(figsize=(10, 4))

# 計算頻率軸
frequencies = np.fft.rfftfreq(frame_size, d=1/sampling_rate)

# 原始幀
plt.subplot(1, 2, 1)
plt.plot(frames[0])
plt.title("Original Frame")

# 漢明窗後的幀
plt.subplot(1, 2, 2)
plt.plot(windowed_frames[0])
plt.title("Hamming Windowed Frame")

plt.tight_layout()
plt.show()

stft_result = librosa.stft(audio_signal)
stft_db = librosa.amplitude_to_db(np.abs(stft_result), ref=np.max)

# 繪製頻譜圖
plt.figure(figsize=(10, 6))
librosa.display.specshow(stft_db, sr=sampling_rate, x_axis='time', y_axis='log')
plt.colorbar(format='%+2.0f dB')
plt.title('Spectrogram (Log scale)')
plt.xlabel('Time')
plt.ylabel('Frequency')
plt.tight_layout()
plt.show()
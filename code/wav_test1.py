import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt

def high_pass_filter(signal, cutoff_freq, sampling_rate, order=4):
    nyquist = 0.5 * sampling_rate
    normalized_cutoff_freq = cutoff_freq / nyquist
    b, a = butter(order, normalized_cutoff_freq, btype='high', analog=False)
    filtered_signal = filtfilt(b, a, signal)
    return filtered_signal

def apply_hamming_window(signal):
    hamming_window = np.hamming(len(signal))
    return signal * hamming_window

# 讀取音頻檔案
audio_signal, sampling_rate = librosa.load('test.wav', sr=22050)

# 1. 原始波形和高通濾波後的波形比較
filtered_signal = high_pass_filter(audio_signal, cutoff_freq=500, sampling_rate=sampling_rate)

plt.figure(figsize=(12, 6))
time = np.arange(len(audio_signal)) / sampling_rate
plt.plot(time, audio_signal, label='Original', alpha=0.7)
plt.plot(time, filtered_signal, label='High-pass Filtered', alpha=0.7)
plt.title('Original vs High-pass Filtered Signal')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.legend()
plt.tight_layout()
plt.show()

# 2. 原始信號和加窗後的信號比較
windowed_signal = apply_hamming_window(filtered_signal)

plt.figure(figsize=(12, 6))
plt.plot(time, filtered_signal, label='Filtered Signal', alpha=0.7)
plt.plot(time, windowed_signal, label='Hamming Windowed', alpha=0.7)
plt.title('Filtered Signal vs Hamming Windowed Signal')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.legend()
plt.tight_layout()
plt.show()

# 3. STFT頻譜圖比較
plt.figure(figsize=(12, 6))
D_filtered = librosa.stft(filtered_signal)
D_db_filtered = librosa.amplitude_to_db(np.abs(D_filtered), ref=np.max)
librosa.display.specshow(D_db_filtered, sr=sampling_rate, x_axis='time', y_axis='hz')
plt.colorbar(format='%+2.0f dB')
plt.title('STFT Spectrogram of Filtered Signal')
plt.tight_layout()
plt.show()

# 4. MEL頻譜圖
plt.figure(figsize=(12, 6))
mel_spect = librosa.feature.melspectrogram(
    y=windowed_signal,
    sr=sampling_rate,
    n_mels=128,
    fmax=8000
)
mel_spect_db = librosa.power_to_db(mel_spect, ref=np.max)
librosa.display.specshow(
    mel_spect_db,
    sr=sampling_rate,
    x_axis='time',
    y_axis='mel',
    fmax=8000
)
plt.colorbar(format='%+2.0f dB')
plt.title('Mel Spectrogram of Processed Signal')
plt.tight_layout()
plt.show()

# 打印處理信息
print("Signal Processing Information:")
print(f"Sampling rate: {sampling_rate} Hz")
print(f"Signal duration: {len(audio_signal)/sampling_rate:.2f} seconds")
print(f"High-pass filter cutoff frequency: 500 Hz")
print(f"Number of mel bands: 128")
print(f"Maximum frequency for mel spectrogram: 8000 Hz")
import librosa
import numpy as np
import scipy.signal as signal
import matplotlib.pyplot as plt

# 音頻加載
def load_audio(file_path, sample_rate=22050):
    audio, sr = librosa.load(file_path, sr=sample_rate)
    return audio, sr

# 去噪 (使用簡單的高通濾波器)
def denoise_audio(audio, sr, cutoff_freq=1000):
    b, a = signal.butter(4, cutoff_freq / (0.5 * sr), btype='high')
    denoised_audio = signal.filtfilt(b, a, audio)
    return denoised_audio

# 重採樣
def resample_audio(audio, orig_sr, target_sr=16000):
    resampled_audio = librosa.resample(audio, orig_sr, target_sr)
    return resampled_audio

# 特徵提取 (MFCC)
def extract_features(audio, sr, n_mfcc=13):
    mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=n_mfcc)
    return mfccs

# 畫出音頻波形和特徵
def plot_waveform_and_mfcc(audio, sr, mfccs):
    plt.figure(figsize=(12, 6))

    # 畫出波形
    plt.subplot(2, 1, 1)
    librosa.display.waveshow(audio, sr=sr)
    plt.title('Waveform')

    # 畫出MFCC
    plt.subplot(2, 1, 2)
    librosa.display.specshow(mfccs, x_axis='time', sr=sr)
    plt.colorbar()
    plt.title('MFCC')

    plt.tight_layout()
    plt.show()

# 主程式
if __name__ == '__main__':
    file_path = 'C:\a_jiao\forest.wav'  # 替換為你的音頻文件路徑
    target_sr = 16000  # 設定目標取樣率

    # 加載音頻
    audio, sr = load_audio(file_path)
    print(f"Original Sample Rate: {sr}")

    # 去噪
    denoised_audio = denoise_audio(audio, sr)

    # 重採樣
    resampled_audio = resample_audio(denoised_audio, sr, target_sr)
    print(f"Resampled to: {target_sr}")

    # 提取MFCC特徵
    mfccs = extract_features(resampled_audio, target_sr)

    # 畫出波形和MFCC
    plot_waveform_and_mfcc(resampled_audio, target_sr, mfccs)

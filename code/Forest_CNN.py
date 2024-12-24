import numpy as np
import librosa
import os
from pathlib import Path
from scipy.signal import butter, lfilter
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
from datetime import datetime

class AudioPreprocessor:
    """音訊預處理類別"""
    
    def __init__(self, sample_rate=22050, n_mels=128, hop_length=512):
        self.sr = sample_rate
        self.n_mels = n_mels
        self.hop_length = hop_length
        self.segment_duration = 2  # 每段音訊長度(秒)
        self.n_segments = 5  # 切割段數
        
    def highpass_filter(self, signal, cutoff=500, order=5):
        """
        應用高通濾波器
        Args:
            signal: 輸入音訊信號
            cutoff: 截止頻率 (Hz)
            order: 濾波器階數
        Returns:
            經過濾波的信號
        """
        nyquist = 0.5 * self.sr
        normal_cutoff = cutoff / nyquist
        b, a = butter(order, normal_cutoff, btype='high', analog=False)
        return lfilter(b, a, signal)

    def segment_audio(self, signal):
        """
        將音訊切割成等長片段
        Args:
            signal: 輸入音訊信號
        Returns:
            切割後的音訊片段列表
        """
        # 計算期望的音訊長度(10秒)
        expected_length = self.sr * self.segment_duration * self.n_segments
        
        # 處理音訊長度
        if len(signal) < expected_length:
            signal = np.pad(signal, (0, expected_length - len(signal)))
        elif len(signal) > expected_length:
            signal = signal[:expected_length]
            
        # 切割為5個相等長度的片段
        segment_length = len(signal) // self.n_segments
        return np.array_split(signal, self.n_segments)

    def apply_hamming(self, signal):
        """
        應用漢明窗
        Args:
            signal: 輸入音訊信號
        Returns:
            應用漢明窗後的信號
        """
        return signal * np.hamming(len(signal))

    def create_melspectrogram(self, signal):
        """
        創建梅爾頻譜圖
        Args:
            signal: 輸入音訊信號
        Returns:
            梅爾頻譜圖
        """
        mel_spect = librosa.feature.melspectrogram(
            y=signal,
            sr=self.sr,
            n_mels=self.n_mels,
            hop_length=self.hop_length,
            fmax=self.sr/2
        )
        mel_spect_db = librosa.power_to_db(mel_spect, ref=np.max)
        return mel_spect_db

    def process_file(self, file_path):
        """
        處理單個音訊檔案
        Args:
            file_path: 音訊檔案路徑
        Returns:
            處理後的梅爾頻譜圖列表
        """
        # 讀取音訊
        signal, _ = librosa.load(file_path, sr=self.sr)
        
        # 高通濾波
        filtered_signal = self.highpass_filter(signal)
        
        # 切割音訊
        segments = self.segment_audio(filtered_signal)
        
        # 處理每個片段
        spectrograms = []
        for segment in segments:
            windowed_segment = self.apply_hamming(segment)
            mel_spect = self.create_melspectrogram(windowed_segment)
            spectrograms.append(mel_spect)
            
        return np.array(spectrograms)

class DataLoader:
    """數據載入和管理類別"""
    
    def __init__(self, preprocessor):
        self.preprocessor = preprocessor
        self.label_encoder = LabelEncoder()
        
    def load_dataset(self, dataset_path):
        """
        載入並處理整個數據集
        Args:
            dataset_path: 數據集根目錄路徑
        Returns:
            特徵和標籤
        """
        features = []
        labels = []
        dataset_path = Path(dataset_path)
        
        # 遍歷所有類別
        for class_dir in dataset_path.iterdir():
            if not class_dir.is_dir():
                continue
                
            print(f"Processing class: {class_dir.name}")
            
            # 遍歷該類別下的所有音訊檔案
            for audio_file in class_dir.glob('*.wav'):
                try:
                    # 處理音訊檔案
                    spectrograms = self.preprocessor.process_file(str(audio_file))
                    
                    # 儲存特徵和標籤
                    for spectrogram in spectrograms:
                        features.append(spectrogram)
                        labels.append(class_dir.name)
                        
                except Exception as e:
                    print(f"Error processing {audio_file}: {str(e)}")
        
        # 轉換為numpy數組
        X = np.array(features)
        y = self.label_encoder.fit_transform(labels)
        
        # 添加通道維度
        X = X[..., np.newaxis]
        
        return X, y

class CNNModel:
    """CNN模型類別"""
    
    def __init__(self, input_shape, num_classes):
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.model = self._build_model()
        
    def _build_model(self):
        """
        建立CNN模型架構
        Returns:
            編譯後的模型
        """
        model = models.Sequential([
            # 第一個卷積層組
            layers.Conv2D(32, (3, 3), activation='relu', padding='same', 
                         input_shape=self.input_shape),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            
            # 第二個卷積層組
            layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            
            # 第三個卷積層組
            layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            
            # 全連接層
            layers.Flatten(),
            layers.Dense(256, activation='relu'),
            layers.Dropout(0.5),
            layers.Dense(self.num_classes, activation='softmax')
        ])
        
        # 編譯模型
        model.compile(
            optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return model

class ModelTrainer:
    """模型訓練和評估類別"""
    
    def __init__(self, model, checkpoint_dir='checkpoints'):
        self.model = model
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(exist_ok=True)
        
    def create_callbacks(self):
        """
        創建訓練回調函數
        Returns:
            回調函數列表
        """
        # 模型檢查點
        checkpoint_path = self.checkpoint_dir / 'model_{epoch:02d}-{val_accuracy:.2f}.h5'
        '''checkpoint_callback = callbacks.ModelCheckpoint(
            str(checkpoint_path),
            monitor='val_accuracy',
            save_best_only=True,
            mode='max',
            verbose=1
        )'''
        
        # 早停
        # early_stopping = callbacks.EarlyStopping(
        #     monitor='val_loss',
        #     patience=10,
        #     restore_best_weights=True
        # )
        
        # 學習率調整
        reduce_lr = callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.2,
            patience=5,
            min_lr=1e-6
        )
        
        return [reduce_lr]#checkpoint_callback, early_stopping, 

    def train(self, X, y, validation_split=0.2, epochs=50, batch_size=50):
        """
        訓練模型
        Args:
            X: 特徵數據
            y: 標籤
            validation_split: 驗證集比例
            epochs: 訓練輪數
            batch_size: 批次大小
        Returns:
            訓練歷史和測試數據
        """
        # 分割數據
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=validation_split, random_state=42
        )
        
        # 訓練模型
        history = self.model.fit(
            X_train, y_train,
            validation_data=(X_test, y_test),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=self.create_callbacks()
        )
        
        return history, (X_test, y_test)

class ModelEvaluator:
    """模型評估和可視化類別"""
    
    def __init__(self, model, label_encoder):
        self.model = model
        self.label_encoder = label_encoder
        
    def evaluate(self, X_test, y_test, history):
        """
        評估模型並顯示結果
        Args:
            X_test: 測試特徵
            y_test: 測試標籤
            history: 訓練歷史
        """
        # 評估模型
        test_loss, test_acc = self.model.evaluate(X_test, y_test)
        print(f"\nTest accuracy: {test_acc:.3f}")
        
        # 獲取預測結果
        y_pred = self.model.predict(X_test)
        y_pred_classes = np.argmax(y_pred, axis=1)
        
        # 顯示結果
        self._plot_confusion_matrix(y_test, y_pred_classes)
        self._plot_training_history(history)
        self._print_classification_report(y_test, y_pred_classes)
        
    def _plot_confusion_matrix(self, y_true, y_pred):
        """繪製混淆矩陣"""
        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=self.label_encoder.classes_,
                   yticklabels=self.label_encoder.classes_)
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.show()
        
    def _plot_training_history(self, history):
        """繪製訓練歷史"""
        plt.figure(figsize=(12, 4))
        
        # 繪製準確率
        plt.subplot(1, 2, 1)
        plt.plot(history.history['accuracy'], label='Training')
        plt.plot(history.history['val_accuracy'], label='Validation')
        plt.title('Model Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
        
        # 繪製損失
        plt.subplot(1, 2, 2)
        plt.plot(history.history['loss'], label='Training')
        plt.plot(history.history['val_loss'], label='Validation')
        plt.title('Model Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        
        plt.tight_layout()
        plt.show()
        
    def _print_classification_report(self, y_true, y_pred):
        """打印分類報告"""
        print("\nClassification Report:")
        print(classification_report(y_true, y_pred,
                                 target_names=self.label_encoder.classes_))

def main():
    """主函數"""
    # 設定參數
    #Class
    #   class1
    #   class2
    #   class3
    #   class4
    DATASET_PATH = r"C:\\a_jiao\\forest\\test"
    SAMPLE_RATE = 22050
    N_MELS = 128
    
    # 初始化預處理器
    preprocessor = AudioPreprocessor(
        sample_rate=SAMPLE_RATE,
        n_mels=N_MELS
    )
    
    # 載入數據
    data_loader = DataLoader(preprocessor)
    X, y = data_loader.load_dataset(DATASET_PATH)
    print(f"Dataset loaded: {X.shape} features, {len(np.unique(y))} classes")
    
    # 創建模型
    model = CNNModel(
        input_shape=X.shape[1:],
        num_classes=len(np.unique(y))
    ).model
    
    # 訓練模型
    trainer = ModelTrainer(model)
    history, (X_test, y_test) = trainer.train(X, y)
    
    # 評估模型
    evaluator = ModelEvaluator(model, data_loader.label_encoder)
    evaluator.evaluate(X_test, y_test, history)

if __name__ == "__main__":
    main()
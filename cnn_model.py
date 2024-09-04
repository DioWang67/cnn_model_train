import os
import numpy as np
import logging
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from data_utils import DataUtils
import cv2
import datetime
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.callbacks import TensorBoard
logging.basicConfig(level=logging.INFO)

class CNNModel:
    def __init__(self, image_size=(64, 64), batch_size=32, epochs=50, input_dir='./augmented_images'):
        """
        初始化 CNN 模型的參數。
        """
        self.image_size = image_size
        self.batch_size = batch_size
        self.epochs = epochs
        self.input_dir = input_dir
        self.model = None
        self.labels = []
        self.history = None

    def build_model(self, input_shape, num_classes):
        self.model = Sequential([
            Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
            MaxPooling2D((2, 2)),
            Conv2D(64, (3, 3), activation='relu'),
            MaxPooling2D((2, 2)),
            Conv2D(128, (3, 3), activation='relu'),
            MaxPooling2D((2, 2)),
            Flatten(),
            Dense(128, activation='relu'),
            Dropout(0.5),
            Dense(num_classes, activation='softmax')
        ])
        self.model.compile(optimizer=Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])


    def load_data(self):
        """
        從輸入目錄加載圖像及其對應的標籤。
        """
        data = []
        targets = []
        for filename in os.listdir(self.input_dir):
            if filename.endswith(".jpg"):
                label = DataUtils.extract_label(filename)
                if label and label not in self.labels:
                    self.labels.append(label)
                if label:
                    image_path = os.path.join(self.input_dir, filename)
                    image = cv2.imread(image_path)
                    if image is not None:
                        data.append(DataUtils.preprocess_image(image, self.image_size))
                        targets.append(self.labels.index(label))
                    else:
                        logging.warning(f"無法讀取圖像 {filename}。跳過此文件。")
        return np.array(data), to_categorical(targets, num_classes=len(self.labels))

    def train(self, callbacks=None):
        """
        使用加載的數據訓練 CNN 模型。支持傳遞回調函數。
        """
        # 加載數據
        X, y = self.load_data()
        if len(X) == 0:
            raise ValueError("未找到有效的圖像文件，請檢查輸入目錄和文件名格式。")

        # 將數據集劃分為訓練集和測試集
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # 設置數據增強參數
        datagen = ImageDataGenerator(
            rotation_range=20,
            width_shift_range=0.2,
            height_shift_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True,
            vertical_flip=True
        )
        datagen.fit(X_train)  # 使用增強參數擴充訓練數據

        # 設定模型輸入形狀和類別數量
        input_shape = (*self.image_size, 1)
        num_classes = len(self.labels)
        self.build_model(input_shape, num_classes)
        checkpoint_callback = ModelCheckpoint(
            filepath='./models/best_model.keras',
            monitor='val_accuracy',
            save_best_only=True,
            verbose=1
        )
        early_stopping = EarlyStopping(monitor='val_loss', patience=5, verbose=1, restore_best_weights=True)
        lr_reduction = ReduceLROnPlateau(monitor='val_loss', patience=3, verbose=1, factor=0.5, min_lr=0.00001)
        #tensorboard_callback = TensorBoard(log_dir="./logs", histogram_freq=1)
        # 訓練模型，使用回調函數
        self.history = self.model.fit(
            datagen.flow(X_train, y_train, batch_size=self.batch_size),  # 增強後的訓練數據
            epochs=self.epochs,
            validation_data=(X_test, y_test),  # 直接使用未增強的測試數據
            callbacks=[early_stopping, lr_reduction, checkpoint_callback]
        )

        # 評估模型性能
        loss, accuracy = self.model.evaluate(X_test, y_test)
        logging.info(f"模型準確率: {accuracy * 100:.2f}%")


    def save_model(self, model_base_path='./models/', labels_base_path='./labels/'):
        """
        保存訓練好的模型和標籤到指定路徑，使用時間戳來標記版本。
        """
        timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        model_path = os.path.join(model_base_path, f'cnn_model_{timestamp}.keras')
        labels_path = os.path.join(labels_base_path, f'labels_{timestamp}.npy')

        os.makedirs(model_base_path, exist_ok=True)
        os.makedirs(labels_base_path, exist_ok=True)

        if self.model:
            self.model.save(model_path)
            np.save(labels_path, self.labels)
            logging.info(f"模型保存到 {model_path}，標籤保存到 {labels_path}。")

    def plot_training_history(self):
        """
        繪製訓練過程中的準確率和驗證準確率。
        """
        if self.history:
            plt.plot(self.history.history['accuracy'], label='accuracy')
            plt.plot(self.history.history['val_accuracy'], label='val_accuracy')
            plt.xlabel('Epoch')
            plt.ylabel('Accuracy')
            plt.ylim([0, 1])
            plt.legend(loc='lower right')
            plt.show()
        else:
            logging.warning("沒有訓練歷史可供繪製。")

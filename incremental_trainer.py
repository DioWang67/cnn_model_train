import os
import cv2
import numpy as np
import logging
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping
from tensorflow.keras.utils import to_categorical
from sklearn.utils import shuffle
from data_utils import DataUtils
import datetime

logging.basicConfig(level=logging.INFO)

class IncrementalTrainer:
    def __init__(self, model_path, labels_path, new_data_dir, replay_buffer_dir=None, image_size=(64, 64)):
        self.model_path = model_path
        self.labels_path = labels_path
        self.new_data_dir = new_data_dir
        self.replay_buffer_dir = replay_buffer_dir
        self.image_size = image_size
        self.model = self.load_model()
        self.labels = self.load_labels()

    def load_model(self):
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"模型文件未找到：{self.model_path}")
        return load_model(self.model_path)

    def load_labels(self):
        if not os.path.exists(self.labels_path):
            raise FileNotFoundError(f"標籤文件未找到：{self.labels_path}")
        return np.load(self.labels_path, allow_pickle=True).tolist()

    def load_new_data(self, data_dir=None):
        if data_dir is None:
            data_dir = self.new_data_dir

        new_data = []
        new_targets = []
        new_labels = self.labels.copy()

        for filename in os.listdir(data_dir):
            if filename.endswith(".jpg"):
                label = DataUtils.extract_label(filename)
                if label:
                    if label not in new_labels:
                        new_labels.append(label)
                    image_path = os.path.join(data_dir, filename)
                    image = cv2.imread(image_path)
                    if image is not None:
                        new_data.append(DataUtils.preprocess_image(image, self.image_size))
                        new_targets.append(new_labels.index(label))
                    else:
                        logging.warning(f"無法讀取圖像文件：{image_path}")
        
        self.labels = new_labels  # 更新標籤列表
        return np.array(new_data), to_categorical(new_targets, num_classes=len(new_labels)), new_labels

    def load_replay_data(self):
        if self.replay_buffer_dir:
            replay_data, replay_targets, _ = self.load_new_data(self.replay_buffer_dir)
            return replay_data, replay_targets
        return np.array([]), np.array([])

    def train(self, epochs=50, batch_size=32, callbacks=None):
        new_X, new_y, new_labels = self.load_new_data()
        replay_X, replay_y = self.load_replay_data()

        combined_X = np.concatenate((new_X, replay_X), axis=0) if replay_X.size > 0 else new_X
        combined_y = np.concatenate((new_y, replay_y), axis=0) if replay_y.size > 0 else new_y

        if combined_X.size == 0 or combined_y.size == 0:
            raise ValueError("無有效數據進行訓練。請檢查新數據和回放數據目錄。")

        combined_X, combined_y = shuffle(combined_X, combined_y, random_state=42)

        # 檢查數據形狀
        logging.info(f"訓練集形狀: {combined_X.shape}")
        logging.info(f"訓練標籤形狀: {combined_y.shape}")

        # 修改手動劃分訓練和驗證集的邏輯
        if len(combined_X) < 2:
            # 樣本數不足時，全部作為訓練集
            train_X, train_y = combined_X, combined_y
            val_X, val_y = combined_X, combined_y
        else:
            split_index = int(0.8 * len(combined_X))
            train_X, val_X = combined_X[:split_index], combined_X[split_index:]
            train_y, val_y = combined_y[:split_index], combined_y[split_index:]

        logging.info(f"訓練集形狀: {train_X.shape}, 驗證集形狀: {val_X.shape}")
        logging.info(f"訓練標籤形狀: {train_y.shape}, 驗證標籤形狀: {val_y.shape}")


        logging.info(f"訓練集形狀: {train_X.shape}, 驗證集形狀: {val_X.shape}")
        logging.info(f"訓練標籤形狀: {train_y.shape}, 驗證標籤形狀: {val_y.shape}")

        datagen = ImageDataGenerator(
            rotation_range=20,
            width_shift_range=0.2,
            height_shift_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True,
            vertical_flip=True
        )

        train_generator = datagen.flow(train_X, train_y, batch_size=batch_size)
        validation_generator = datagen.flow(val_X, val_y, batch_size=batch_size)

        history = self.model.fit(
            train_generator,
            epochs=epochs,
            validation_data=validation_generator,
            callbacks=callbacks
        )

        timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        self.model.save(f'cnn_model.keras')
        np.save(f'labels.npy', self.labels)

        loss, accuracy = self.model.evaluate(validation_generator)
        logging.info(f"更新后的模型准确率: {accuracy * 100:.2f}%")

        return history

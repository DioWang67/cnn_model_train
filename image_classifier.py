
import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
tf.config.run_functions_eagerly(True)
class ImageClassifier:
    def __init__(self):
        model_path = './cnn_model.keras'
        labels_path = './labels.npy'

        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
        if not os.path.exists(labels_path):
            raise FileNotFoundError(f"Labels file not found: {labels_path}")

        self.model = load_model(model_path)
        self.labels = np.load(labels_path, allow_pickle=True)

        # 使用 @tf.function 裝飾器，並設置輸入的形狀
        self.predict_function = tf.function(
            self.model.predict,
            input_signature=[tf.TensorSpec(shape=(1, 64, 64, 1), dtype=tf.float32)]
        )

    def preprocess_and_extract_edges(self, image, image_size=(64, 64)):
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        resized_image = cv2.resize(gray_image, image_size)
        reshaped_image = resized_image.reshape((64, 64, 1)) / 255.0
        return reshaped_image

    def predict(self, image):
        new_image_features = self.preprocess_and_extract_edges(image).reshape((1, 64, 64, 1))
        prediction = self.predict_function(new_image_features)
        
        # 打印模型的輸出
        print("Prediction output:", prediction)
        
        max_proba_index = np.argmax(prediction)
        predicted_label = str(self.labels[max_proba_index])
        max_proba_score = np.float64(prediction[0][max_proba_index])
        
        # 打印置信度分數
        print(f"Predicted label: {predicted_label}, Max probability score: {max_proba_score}")
        
        return predicted_label, predicted_label, max_proba_score


    def extract_field_from_filename(self, filename):
        # 獲取文件的基礎名稱，不包括路徑
        base_name = os.path.basename(filename)
        
        # 將文件名分割成多個部分，使用 '-' 作為分隔符
        parts = base_name.split('-')
        
        # 檢查分割後的部分數量，並返回第二部分（索引為1），即 `J1`
        if len(parts) > 1:
            return parts[1]  # 返回標籤部分 'J1'
        
        # 如果分割後的部分不足兩個，則返回 None
        return None


    def compare_with_ccd_field(self, image, ccd_filename):
        predicted_label, max_proba_label, max_proba_score = self.predict(image)
        ccd_field = self.extract_field_from_filename(ccd_filename)
        
        # 調試輸出
        print(f"Predicted Label: {predicted_label}, CCD Field: {ccd_field}")
        
        match = predicted_label == ccd_field
        if match:
            return ccd_field, max_proba_label, max_proba_score
        else:
            return ccd_field, max_proba_label, 0.0

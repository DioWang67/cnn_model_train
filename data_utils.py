import cv2
import numpy as np
import re

class DataUtils:
    """
    一個用於處理數據的工具類別，包含預處理圖像和提取標籤的方法。
    """
    @staticmethod
    def preprocess_image(image, image_size):
        """
        將圖像轉換為灰度，調整大小並進行歸一化。
        """
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        resized_image = cv2.resize(gray_image, image_size)
        return resized_image.reshape((*image_size, 1)) / 255.0

    @staticmethod
    def extract_label(filename):
        """
        從文件名中提取標籤。
        """
        match = re.search(r'J\w+', filename)
        return match.group(0) if match else None

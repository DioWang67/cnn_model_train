from PIL import Image, ImageEnhance
import random
import os
import numpy as np

class ImageAugmentor:
    def __init__(self, input_dir, output_dir, num_augmented_images,
                 rotation_range, scale_range, brightness_range):
        """
        初始化影像增強器的屬性
        :param input_dir: 原始圖片的資料夾
        :param output_dir: 增強後圖片的保存資料夾
        :param num_augmented_images: 每張圖片要生成的增強版本數量
        :param rotation_range: 隨機旋轉的角度範圍
        :param scale_range: 隨機縮放的比例範圍
        :param brightness_range: 隨機亮度調整的範圍
        """
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.num_augmented_images = num_augmented_images
        self.rotation_range = rotation_range
        self.scale_range = scale_range
        self.brightness_range = brightness_range

        # 確保輸出目錄存在
        os.makedirs(self.output_dir, exist_ok=True)

    def augment_image(self, image_path, base_name):
        """
        增強單張圖片
        :param image_path: 圖片的完整路徑
        :param base_name: 圖片的基本名稱（不含擴展名）
        """
        original_image = Image.open(image_path)

        for i in range(1, self.num_augmented_images + 1):
            # 設定檔案的編號
            a = i + 0

            # 隨機旋轉
            angle = random.uniform(*self.rotation_range)
            augmented_image = original_image.rotate(angle)

            # 隨機縮放
            scale_factor = random.uniform(*self.scale_range)
            new_size = (int(augmented_image.width * scale_factor), int(augmented_image.height * scale_factor))
            augmented_image = augmented_image.resize(new_size, Image.LANCZOS)

            # 隨機調整亮度
            enhancer = ImageEnhance.Brightness(augmented_image)
            augmented_image = enhancer.enhance(random.uniform(*self.brightness_range))

            # 隨機水平翻轉
            if random.choice([True, False]):
                augmented_image = augmented_image.transpose(Image.FLIP_LEFT_RIGHT)
                
            augmented_image = self.add_random_noise(augmented_image, noise_factor=10)
            # 保存增強的圖像，命名格式為 <前綴><編號>.jpg
            output_filename = f'{base_name}{a}.jpg'
            augmented_image.save(os.path.join(self.output_dir, output_filename))

    def add_random_noise(self,image, noise_factor=0.5):
        np_image = np.array(image)
        noise = np.random.normal(0, noise_factor, np_image.shape)
        noisy_image = np_image + noise
        noisy_image = np.clip(noisy_image, 0, 255).astype(np.uint8)
        return Image.fromarray(noisy_image)

    def process_directory(self):
        """
        處理輸入資料夾內的所有圖片
        """
        image_files = [f for f in os.listdir(self.input_dir) if os.path.isfile(os.path.join(self.input_dir, f))]

        for image_file in image_files:
            image_path = os.path.join(self.input_dir, image_file)
            base_name = os.path.splitext(image_file)[0]  # 去除文件擴展名
            self.augment_image(image_path, base_name)


# 使用範例
if __name__ == "__main__":
    # 創建一個 ImageAugmentor 實例，並設定增強參數
    augmentor = ImageAugmentor(
        input_dir=r'D:\Git\cnn_model_train\A41402237001S-data\camera0',  # 原始圖片的資料夾
        output_dir=r'D:\Git\cnn_model_train\augmented_images',  # 增強後圖片的保存資料夾
        num_augmented_images=500,  # 每張圖片要生成的增強版本數量
        rotation_range=(-30, 30),  # 隨機旋轉的角度範圍
        scale_range=(0.5, 1.5),  # 隨機縮放的比例範圍
        brightness_range=(0.6, 1.4)  # 隨機亮度調整的範圍
    )

    # 開始處理目錄中的圖片
    augmentor.process_directory()

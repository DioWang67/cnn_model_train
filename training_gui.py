# training_app.py

import tkinter as tk
from tkinter import messagebox, filedialog
from cnn_model import CNNModel
from incremental_trainer import IncrementalTrainer
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from image_classifier import ImageClassifier
import cv2

class TrainingApp:
    def __init__(self, root):
        self.root = root
        self.root.title("CNN 模型訓練與測試介面")
        self.root.geometry("400x400")

        self.create_widgets()

    def create_widgets(self):
        # 模式選擇
        self.mode_label = tk.Label(self.root, text="選擇訓練模式:")
        self.mode_label.pack(pady=10)

        self.mode_var = tk.StringVar(value="initial")
        self.initial_radio = tk.Radiobutton(self.root, text="初始訓練", variable=self.mode_var, value="initial")
        self.incremental_radio = tk.Radiobutton(self.root, text="增量訓練", variable=self.mode_var, value="incremental")
        self.initial_radio.pack()
        self.incremental_radio.pack()

        # 訓練參數
        self.epochs_label = tk.Label(self.root, text="訓練次數 (最大 epochs):")
        self.epochs_label.pack(pady=5)
        self.epochs_entry = tk.Entry(self.root)
        self.epochs_entry.insert(0, "100")
        self.epochs_entry.pack()

        self.batch_size_label = tk.Label(self.root, text="批次大小 (batch size):")
        self.batch_size_label.pack(pady=5)
        self.batch_size_entry = tk.Entry(self.root)
        self.batch_size_entry.insert(0, "32")
        self.batch_size_entry.pack()

        # 開始訓練按鈕
        self.train_button = tk.Button(self.root, text="開始訓練", command=self.start_training)
        self.train_button.pack(pady=20)

        # 測試影像分類按鈕
        self.test_button = tk.Button(self.root, text="測試影像", command=self.test_image_classification)
        self.test_button.pack(pady=20)

    def start_training(self):
        mode = self.mode_var.get()
        epochs = int(self.epochs_entry.get())
        batch_size = int(self.batch_size_entry.get())

        if mode == "initial":
            self.run_initial_training(epochs, batch_size)
        elif mode == "incremental":
            self.run_incremental_training(epochs, batch_size)

    def run_initial_training(self, epochs, batch_size):
        input_dir = filedialog.askdirectory(title="選擇初始訓練數據目錄")
        if not input_dir:
            messagebox.showwarning("警告", "必須選擇初始訓練數據目錄")
            return

        cnn_model = CNNModel(image_size=(64, 64), batch_size=batch_size, epochs=epochs, input_dir=input_dir)
        
        early_stopping = EarlyStopping(monitor='val_loss', patience=5, verbose=1, restore_best_weights=True)
        lr_reduction = ReduceLROnPlateau(monitor='val_loss', patience=3, verbose=1, factor=0.5, min_lr=0.00001)

        cnn_model.train(callbacks=[early_stopping, lr_reduction])
        cnn_model.save_model()
        cnn_model.plot_training_history()

        messagebox.showinfo("完成", "初始訓練完成！")

    def run_incremental_training(self, epochs, batch_size):
        model_path = filedialog.askopenfilename(title="選擇已訓練好的模型文件", filetypes=[("Keras 模型", "*.keras")])
        labels_path = filedialog.askopenfilename(title="選擇標籤文件", filetypes=[("Numpy 文件", "*.npy")])
        new_data_dir = filedialog.askdirectory(title="選擇增量訓練數據目錄")

        if not model_path or not labels_path or not new_data_dir:
            messagebox.showwarning("警告", "必須選擇模型文件、標籤文件和新數據目錄")
            return

        replay_buffer_dir = filedialog.askdirectory(title="選擇回放緩衝區目錄 (可選)")
        if replay_buffer_dir == '':
            replay_buffer_dir = None

        incremental_trainer = IncrementalTrainer(
            model_path=model_path,
            labels_path=labels_path,
            new_data_dir=new_data_dir,
            replay_buffer_dir=replay_buffer_dir
        )
        
        early_stopping = EarlyStopping(monitor='val_loss', patience=5, verbose=1, restore_best_weights=True)
        lr_reduction = ReduceLROnPlateau(monitor='val_loss', patience=3, verbose=1, factor=0.5, min_lr=0.00001)

        incremental_trainer.train(epochs=epochs, batch_size=batch_size, callbacks=[early_stopping, lr_reduction])

        messagebox.showinfo("完成", "增量訓練完成！")

    def test_image_classification(self):
        # 選擇要測試的圖片
        image_path = filedialog.askopenfilename(title="選擇測試圖片", filetypes=[("Image files", "*.jpg *.png *.jpeg")])
        if not image_path:
            messagebox.showwarning("警告", "必須選擇測試圖片")
            return

        ccd_filename = filedialog.askopenfilename(title="選擇 CCD 文件", filetypes=[("Image files", "*.jpg *.png *.jpeg")])
        if not ccd_filename:
            messagebox.showwarning("警告", "必須選擇 CCD 文件")
            return

        image = cv2.imread(image_path)
        if image is None:
            messagebox.showwarning("警告", f"無法讀取圖片: {image_path}")
            return

        classifier = ImageClassifier()
        predicted_label, max_proba_label, max_proba_score = classifier.compare_with_ccd_field(image, ccd_filename)

        result_message = f"預測結果: {predicted_label}\n" \
                         f"信心最高的類別: {max_proba_label}\n" \
                         f"信心分數: {max_proba_score:.2f}"

        messagebox.showinfo("測試結果", result_message)


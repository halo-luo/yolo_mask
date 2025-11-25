# 文件名: sparky_zero.py
import tkinter as tk
from tkinter import ttk, messagebox
from PIL import Image, ImageTk
import cv2
import threading
import numpy as np
from ultralytics import YOLO
import os


class SparkyZero:
    def __init__(self, root):
        self.root = root
        self.root.title("Sparky-Zero：零样本魔法检测器 🔥")
        self.root.geometry("1000x720")
        self.root.configure(bg="#1a1a1a")
        self.root.resizable(False, False)

        # 核心：YOLO-World（开放词汇神器）
        print("正在加载 YOLO-World 模型（首次稍慢）...")
        self.model = YOLO("yolov8s-world.pt")  # 推荐 s 或 m，速度和精度平衡最好
        # 也可用 yolov8s-world.yaml（更轻更快，但首次推理慢一点）
        # self.model = YOLO("yolov8s-world.yaml")

        self.current_classes = ["person", "cat", "phone"]
        self.model.set_classes(self.current_classes)

        self.cap = None
        self.running = False

        self.create_widgets()

    def create_widgets(self):
        # === 标题 ===
        title = tk.Label(self.root, text="Sparky-Zero", font=("微软雅黑", 28, "bold"), fg="#00ff99", bg="#1a1a1a")
        title.pack(pady=10)

        subtitle = tk.Label(self.root, text="输入任何你想检测的东西 → 回车即可识别！", font=("微软雅黑", 12),
                            fg="#aaaaaa", bg="#1a1a1a")
        subtitle.pack(pady=5)

        # === 输入框 + 按钮 ===
        input_frame = tk.Frame(self.root, bg="#1a1a1a")
        input_frame.pack(pady=10)

        tk.Label(input_frame, text="想检测什么？", font=("微软雅黑", 14), fg="white", bg="#1a1a1a").pack(side="left",
                                                                                                        padx=5)
        self.entry = tk.Entry(input_frame, font=("微软雅黑", 14), width=40, relief="flat", bg="#333333", fg="white",
                              insertbackground="white")
        self.entry.pack(side="left", padx=5)
        self.entry.bind("<Return>", self.add_class)

        add_btn = tk.Button(input_frame, text="添加", command=self.add_class, bg="#00ff99", fg="black",
                            font=("微软雅黑", 10, "bold"), relief="flat")
        add_btn.pack(side="left", padx=5)

        clear_btn = tk.Button(input_frame, text="清空", command=self.clear_classes, bg="#ff3366", fg="white",
                              font=("微软雅黑", 10, "bold"), relief="flat")
        clear_btn.pack(side="left", padx=5)

        # === 当前类别显示 ===
        self.class_label = tk.Label(self.root, text=f"当前检测：{', '.join(self.current_classes)}",
                                    font=("微软雅黑", 11), fg="#00ff99", bg="#1a1a1a")
        self.class_label.pack(pady=5)

        # === 摄像头开关 ===
        self.cam_btn = tk.Button(self.root, text="打开摄像头", command=self.toggle_camera,
                                 bg="#3366ff", fg="white", font=("微软雅黑", 14, "bold"), width=20, height=2)
        self.cam_btn.pack(pady=10)

        # === 图片显示区域 ===
        self.canvas = tk.Label(self.root, bg="#000000", width=860, height=480)
        self.canvas.pack(pady=10)

        # === 提示 ===
        tip = tk.Label(self.root, text="支持示例：cat wearing hat / person with phone / red apple / blue backpack",
                       font=("微软雅黑", 10), fg="#888888", bg="#1a1a1a")
        tip.pack(pady=5)

    def add_class(self, event=None):
        text = self.entry.get().strip()
        if not text:
            return
        if text not in self.current_classes:
            self.current_classes.append(text)
            self.model.set_classes(self.current_classes)
            self.class_label.config(text=f"当前检测：{', '.join(self.current_classes)}")
        self.entry.delete(0, tk.END)

    def clear_classes(self):
        self.current_classes = []
        self.model.set_classes(self.current_classes)
        self.class_label.config(text="当前检测：无")

    def toggle_camera(self):
        if not self.running:
            self.cap = cv2.VideoCapture(0)
            if not self.cap.isOpened():
                messagebox.showerror("错误", "无法打开摄像头！")
                return
            self.running = True
            self.cam_btn.config(text="关闭摄像头", bg="#ff3366")
            self.detect_loop()
        else:
            self.running = False
            self.cam_btn.config(text="打开摄像头", bg="#3366ff")
            if self.cap:
                self.cap.release()

    def detect_loop(self):
        if not self.running:
            return

        ret, frame = self.cap.read()
        if not ret:
            return

        # YOLO-World 推理
        results = self.model(frame, conf=0.25, verbose=False)[0]
        annotated = results.plot()

        # 显示到界面
        rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(rgb)
        img = img.resize((860, 480), Image.Resampling.LANCZOS)
        photo = ImageTk.PhotoImage(img)
        self.canvas.config(image=photo)
        self.canvas.image = photo

        # 下一帧
        self.root.after(10, self.detect_loop)


if __name__ == '__main__':
    print("Sparky-Zero 启动中...")
    root = tk.Tk()
    app = SparkyZero(root)
    root.mainloop()

# UPDATE1225
import os

import cv2
import threading
import subprocess
import numpy as np
import FastApi.getWindowIdFFmpeg as gwID


def generate_frames_camera():
    # 获取摄像头的画面
    camera = cv2.VideoCapture(0)
    while True:
        success, frame = camera.read()
        if not success:
            break
        else:
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

def generate_frames(window_name: str):
    # vs = VideoStream("微信")
    # vs = VideoStream("heihuhu@heihuhu-Ubuntu: ~")
    # vs = VideoStream("CarlaUE4")
    vs = VideoStream(window_name)

    # 不断生成帧图像
    while True:
        frame = vs.get_frame()
        if frame is None:
            print("Frame is None, skipping encoding")
            continue

        ret, buffer = cv2.imencode('.jpg', frame)  # 将图像编码为JPEG格式
        if not ret:
            print("Failed to encode frame")
            continue

        frame_bytes = buffer.tobytes()  # 将编码后的图像转换为字节
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')  # 构建MJPEG流的一部分


class VideoStream:
    def __init__(self, window_name):
        # 初始化帧存储变量和锁
        self.frame = None
        self.lock = threading.Lock()

        # 获取窗口ID
        self.window_id = gwID.get_window_id(window_name)
        print(f"Window ID for '{window_name}': {self.window_id}")  # 调试信息

        # 动态获取窗口分辨率和位置
        self.width, self.height, self.x, self.y = gwID.get_window_geometry(self.window_id)
        print(f"Window resolution: {self.width}x{self.height}, Position: ({self.x}, {self.y})")  # 调试信息

        # 设置 DISPLAY 环境变量
        os.environ['DISPLAY'] = ':1'
        # 使用FFmpeg捕获指定窗口的视频流
        # -f x11grab: 指定输入格式为X11抓屏
        # -s 1920x1080: 设置分辨率为1920x1080
        # -i :0.0+0,0: 指定显示设备和偏移量
        # -window_id <window_id>: 指定窗口ID
        # -pix_fmt bgr24: 设置像素格式为BGR24
        # -r 15: 设置帧率为15fps
        # -f rawvideo: 输出原始视频数据
        # -: 将输出发送到标准输出
        # '-s', '1920x1080',
        self.process = subprocess.Popen(
            ['ffmpeg', '-f', 'x11grab', '-s', f'{self.width}x{self.height}', '-i', f':1+{self.x},{self.y}',
             '-window_id', self.window_id.lstrip("0x"), '-pix_fmt',
             'bgr24', '-r', '15', '-f', 'rawvideo', '-'],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=os.environ
        )
        print(
            f"Started FFmpeg process with command: {' '.join(['ffmpeg', '-f', 'x11grab', '-s', f'{self.width}x{self.height}', '-i', f':1+{self.x},{self.y}', '-window_id', self.window_id.lstrip('0x'), '-pix_fmt', 'bgr24', '-r', '15', '-f', 'rawvideo', '-'])}")  # 调试信息

        # 启动线程来更新帧
        self.thread = threading.Thread(target=self.update_frame)
        self.thread.daemon = True
        self.thread.start()

    def update_frame(self):
        # 不断读取FFmpeg进程的标准输出
        while True:
            # 读取一帧图像的数据（width * height * 3字节）
            raw_image = self.process.stdout.read(self.width * self.height * 3)
            if not raw_image:
                print("No data received from FFmpeg process")
                break

            try:
                # 将字节数据转换为NumPy数组并重塑为图像形状
                image = np.frombuffer(raw_image, dtype=np.uint8).reshape((self.height, self.width, 3))
                print(f"Received frame of size in update_frame in liveStreamFFmpeg: {image.shape}")  # 调试信息

                # 使用锁保护共享资源，防止多线程冲突
                with self.lock:
                    self.frame = image.copy()
            except ValueError as e:
                print(f"Error reshaping image: {e}")

    def get_frame(self):
        # 获取最新的帧图像
        with self.lock:
            return self.frame

    def get_frame(self):
        # 获取最新的帧图像
        with self.lock:
            return self.frame

    def stop(self):
        # 终止FFmpeg进程
        print("Stopping FFmpeg process")
        self.process.terminate()
        self.process.wait()



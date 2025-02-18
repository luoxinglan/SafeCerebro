import asyncio
import cv2
import numpy as np
from PIL import Image
import subprocess
import os
import time

# 打印 DISPLAY 和 XAUTHORITY 环境变量
print(f"DISPLAY environment variable: {os.environ.get('DISPLAY')}")
print(f"XAUTHORITY environment variable: {os.environ.get('XAUTHORITY')}")

# 使用 xdotool 获取窗口ID
def get_window_id(window_name):
    try:
        # 使用 xdotool 查找窗口ID
        result = subprocess.run(
            ['xdotool', 'search', '--onlyvisible', '--name', window_name],
            capture_output=True,
            text=True,
            check=True
        )
        window_ids = result.stdout.strip().split('\n')

        if not window_ids:
            print("没有找到匹配的窗口")
            return None

        # 返回第一个匹配的窗口ID
        window_id = window_ids[0]  # 他本来就是十进制的整数
        print(f"找到窗口ID: {window_id}")
        return window_id

    except subprocess.CalledProcessError as e:
        print(f"使用 xdotool 时出错: {e.stderr}")
        return None


# 激活窗口
def activate_window(window_id):
    try:
        subprocess.run(['xdotool', 'windowactivate', f"{window_id}"], check=True)
        print(f"已激活窗口ID: {window_id}")
    except subprocess.CalledProcessError as e:
        print(f"激活窗口时出错: {e.stderr}")


# 异步生成器，用于捕获特定窗口的帧
async def capture_screen(window_id):
    while True:
        try:
            # 激活窗口
            activate_window(window_id)

            # 增加等待时间以确保窗口准备好
            time.sleep(2)
            print("window_id=", window_id)
            # 使用 import 抓取窗口内容
            output_file = "/tmp/window_capture.png"
            command = ["import", "-window", window_id, output_file]
            print(f"执行命令: {' '.join(command)}")
            subprocess.run(command, check=True)

            # 读取 PNG 文件
            img = Image.open(output_file)
            frame = np.array(img)
            # 将图像从RGB格式转换为OpenCV所需的BGR格式
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            yield frame

            # 删除临时文件
            os.remove(output_file)

        except Exception as e:
            print(f"捕获屏幕时出错: {e}")
            break


# 生成MJPEG流
def generate_frames(window_name):
    window_id = get_window_id(window_name)
    if not window_id:
        print("窗口未找到")
        return

    cap_gen = capture_screen(window_id)

    async def inner():
        nonlocal cap_gen
        while True:
            try:
                frame = await cap_gen.__aiter__().__anext__()
                ret, buffer = cv2.imencode('.jpg', frame)  # 将图像编码为JPEG格式
                if not ret:
                    print("编码帧失败")
                    continue

                frame_bytes = buffer.tobytes()  # 将编码后的图像转换为字节
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')  # 构建MJPEG流的一部分
            except StopAsyncIteration:
                break
            except Exception as e:
                print(f"生成帧时出错: {e}")
                break

    return inner()


# 测试函数，手动验证窗口ID和导入命令
def test_import_command(window_name):
    window_id = get_window_id(window_name)
    if not window_id:
        print("窗口未找到")
        return

    output_file = "/tmp/test_capture.png"
    command = ["import", "-window", window_id, output_file]
    print(f"测试命令: {' '.join(command)}")

    try:
        subprocess.run(command, check=True)
        print(f"成功捕获窗口到 {output_file}")
    except subprocess.CalledProcessError as e:
        print(f"测试命令出错: {e.stderr}")

if __name__ == "__main__":
    test_import_command("飞书")
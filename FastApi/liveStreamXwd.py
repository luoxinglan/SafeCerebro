import asyncio
import cv2
import numpy as np
from PIL import Image
import subprocess
import os
import time

"""
在这个版本中，我们在 xwd 命令中添加了 -silent 选项，以便即使窗口被遮挡或最小化，也能继续捕获其图像。
同时，在 xdotool search 中移除了 --onlyvisible 选项，以确保能找到所有匹配的窗口，而不仅仅是可见的窗口。
"""

# 打印 DISPLAY 和 XAUTHORITY 环境变量
print(f"DISPLAY environment variable: {os.environ.get('DISPLAY')}")
print(f"XAUTHORITY environment variable: {os.environ.get('XAUTHORITY')}")

# 使用 xdotool 获取窗口ID
def get_window_id(window_name):
    try:
        # 使用 xdotool 查找窗口ID，不限制为仅查找可见窗口
        result = subprocess.run(
            ['xdotool', 'search', '--name', window_name],
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


# 异步生成器，用于捕获特定窗口的帧
async def capture_screen(window_id):
    while True:
        try:
            # 增加等待时间以确保窗口准备好
            time.sleep(2)
            print("window_id=", window_id)
            # 使用 xwd 抓取窗口内容，使用 -silent 忽略窗口是否可见
            output_file_xwd = "/tmp/window_capture.xwd"
            command_xwd = ["xwd", "-id", window_id, "-out", output_file_xwd, "-silent"]
            print(f"执行命令: {' '.join(command_xwd)}")
            subprocess.run(command_xwd, check=True)

            # 将 XWD 文件转换为 JPEG 文件
            output_file_jpeg = "/tmp/window_capture.jpg"
            command_convert = ["convert", output_file_xwd, output_file_jpeg]
            print(f"执行命令: {' '.join(command_convert)}")
            subprocess.run(command_convert, check=True)

            # 读取 JPEG 文件并转换为图像
            img = Image.open(output_file_jpeg)
            frame = np.array(img)
            # 将图像从RGB格式转换为OpenCV所需的BGR格式
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            yield frame

            # 删除临时文件
            os.remove(output_file_xwd)
            os.remove(output_file_jpeg)

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

    output_file_xwd = "/tmp/test_capture.xwd"
    command_xwd = ["xwd", "-id", window_id, "-out", output_file_xwd, "-silent"]
    print(f"测试命令: {' '.join(command_xwd)}")

    try:
        subprocess.run(command_xwd, check=True)
        print(f"成功捕获窗口到 {output_file_xwd}")

        # 将 XWD 文件转换为 JPEG 文件
        output_file_jpeg = "/tmp/test_capture.jpg"
        command_convert = ["convert", output_file_xwd, output_file_jpeg]
        print(f"执行命令: {' '.join(command_convert)}")
        subprocess.run(command_convert, check=True)

        print(f"成功转换为 JPEG 文件 {output_file_jpeg}")
    except subprocess.CalledProcessError as e:
        print(f"测试命令出错: {e.stderr}")


if __name__ == "__main__":
    test_import_command("飞书")




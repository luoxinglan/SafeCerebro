import subprocess
import os

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
        window_id = int(window_ids[0], 16)  # 将十六进制字符串转换为整数
        print(f"找到窗口ID: {window_id} (0x{window_id:X})")
        return window_id

    except subprocess.CalledProcessError as e:
        print(f"使用 xdotool 时出错: {e.stderr}")
        return None


# 测试函数，手动验证窗口ID和导入命令
def test_import_command(window_name):
    window_id = get_window_id(window_name)
    if not window_id:
        print("窗口未找到")
        return

    output_file = "/tmp/test_capture.png"
    command = ["import", "-window", f"0x{window_id:X}", output_file]
    print(f"测试命令: {' '.join(command)}")

    try:
        subprocess.run(command, check=True)
        print(f"成功捕获窗口到 {output_file}")
    except subprocess.CalledProcessError as e:
        print(f"测试命令出错: {e.stderr}")

if __name__ == "__main__":
    test_import_command("飞书")




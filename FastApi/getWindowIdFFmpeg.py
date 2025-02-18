# UPDATE1224
import subprocess


def get_window_id(window_name):
    # 使用xwininfo和grep命令来获取窗口ID
    cmd = f"xwininfo -root -tree | grep '{window_name}'"
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)

    if result.returncode != 0:
        raise Exception(f"Failed to find window with name {window_name}: {result.stderr}")

    lines = result.stdout.split('\n')
    for line in lines:
        parts = line.strip().split()
        if len(parts) > 0 and parts[0].startswith('0x'):
            return parts[0]

    raise Exception(f"No window found with name {window_name}")

def get_window_geometry(window_id):
    # 使用 xwininfo 获取窗口的几何信息
    cmd = f"xwininfo -id {window_id}"
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)

    if result.returncode != 0:
        raise Exception(f"Failed to get geometry for window ID {window_id}: {result.stderr}")

    lines = result.stdout.split('\n')
    width = None
    height = None
    x = None
    y = None

    for line in lines:
        if "Width:" in line:
            width = int(line.split(":")[1].strip())
        elif "Height:" in line:
            height = int(line.split(":")[1].strip())
        elif "Absolute upper-left X:" in line:
            x = int(line.split(":")[1].strip())
        elif "Absolute upper-left Y:" in line:
            y = int(line.split(":")[1].strip())

    if width is None or height is None or x is None or y is None:
        raise Exception(f"Could not determine dimensions or position for window ID {window_id}")

    return width, height, x, y

# 使用示例：
if __name__ == "__main__":
    # 示例：获取名为 "WeChat" 的窗口ID
    window_name = "微信"
    window_id = get_window_id(window_name)
    window = get_window_geometry(window_id)
    print(f"Window ID for {window_name}: {window_id}")
    print(f"Window ID for {window_name}: {window}")
import os, asyncio

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

from fastapi import FastAPI, Request
import json

from FastApi.metrics_inserter import MetricsInserter

# 获取当前文件所在目录
current_dir = os.path.dirname(os.path.abspath(__file__))
app = FastAPI()

# 添加CORS中间件
origins = [
    "http://localhost",
    "http://localhost:8080",
    "http://localhost:8081"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 存储 WebSocket 连接
active_connections = []


# 处理 WebSocket 连接
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    active_connections.append(websocket)
    try:
        while True:
            # 保持连接活跃
            await asyncio.sleep(0.1)
    except WebSocketDisconnect:
        active_connections.remove(websocket)


# 发送消息给所有连接的 WebSocket 客户端
async def send_message(message: str):
    for connection in active_connections:
        await connection.send_text(message)


import re

# 处理按钮点击请求
# ANSI escape code 正则表达式
ansi_escape = re.compile(r'\x1B[@-_][0-?]*[ -/]*[@-~]')
progress_pattern = re.compile(r'\[(\d+)/(\d+)\]')


# 处理按钮“启动”点击请求，执行脚本 run.py
@app.post("/run-script")
async def run_script(request: Request):
    try:
        # 获取前端传递的参数
        form_data = await request.json()
        print("Received form data:", form_data)

        # 构建命令行参数
        script_path = os.path.join(current_dir, "../scripts/run.py")
        args = ["python", "-u", script_path]
        for key, value in form_data.items():
            args.append(f"--{key}")
            args.append(str(value))

        # 执行脚本
        process = await asyncio.create_subprocess_exec(
            *args,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )

        # 异步读取标准输出和标准错误
        async def read_stream(stream, label):
            while True:
                data = await stream.read(1024)  # 使用read替代readline
                if not data:
                    break
                lines = data.decode('utf-8').split('\n')
                for line in lines:
                    if line:
                        # 去除 ANSI 控制序列
                        clean_line = ansi_escape.sub('', line)
                        await send_message(f"{label}{clean_line}")

                        # 查找进度信息并单独发送12.16
                        progress_match = progress_pattern.search(clean_line)
                        if progress_match:
                            progress_info = f"Progress: [{progress_match.group(1)}/{progress_match.group(2)}]"
                            await send_message(progress_info)

        # 并发读取输出
        await asyncio.gather(
            read_stream(process.stdout, ""),
            read_stream(process.stderr, "")
        )
        if process.returncode != 0:
            await send_message(f"Script failed with return code {process.returncode}")
        else:
            await send_message("Script succeeded.")

    except Exception as e:
        await send_message(f"An unexpected error occurred: {e}")


from fastapi import FastAPI, HTTPException
import pymysql
from pydantic import BaseModel
from typing import List, Optional


class Metric(BaseModel):
    id: int
    collision_rate: float
    avg_red_light_freq: float
    avg_stop_sign_freq: float
    out_of_road_length: float
    route_following_stability: float
    route_completion: float
    avg_time_spent: float
    avg_acceleration: float
    avg_yaw_velocity: float
    avg_lane_invasion_freq: float
    safety_os: float
    task_os: float
    comfort_os: float
    final_score: float


# TODO：UPATE12.16：扩展 Metric 模型以包含序号字段
class MetricWithIndex(Metric):
    index: int


def get_db_connection():
    return pymysql.connect(
        host='localhost',
        user='heihuhu',
        password='1234',  # 填写您的MySQL密码
        database='db',
        charset='utf8mb4',
        cursorclass=pymysql.cursors.DictCursor
    )


# TODO：UPATE12.16：获取带有下标（index）的metrics（解决Mysql的id自动递增的问题）
@app.get("/metrics", response_model=List[MetricWithIndex])
async def read_metrics():
    connection = get_db_connection()
    try:
        with connection.cursor() as cursor:
            sql = "SELECT * FROM metrics"
            cursor.execute(sql)
            result = cursor.fetchall()
            if not result:
                raise HTTPException(status_code=404, detail="No records found")

            # 使用 enumerate 为每条记录添加索引
            indexed_result = [{"index": idx + 1, **record} for idx, record in enumerate(result)]
            return [MetricWithIndex(**r) for r in indexed_result]
    finally:
        connection.close()


# TODO：UPATE12.20：后端fastapi创建一个a接口，能够获取前端的数据的id（是metrics的id），根据id查询metrics_TTC的metrics_id属性，获得一条数据，返回给前端。
@app.get("/ttc/{metric_id}", response_model=dict)
async def get_metric_by_id(metric_id: int):
    connection = get_db_connection()
    try:
        with connection.cursor() as cursor:
            sql = "SELECT * FROM metrics_TTC WHERE metrics_id=%s"
            cursor.execute(sql, (metric_id,))
            result = cursor.fetchone()
            if not result:
                raise HTTPException(status_code=404, detail=f"No record found for metrics_id={metric_id}")
            return result
    finally:
        connection.close()


# TODO：UPATE12.16：新增路由：检查并插入 metrics 数据
import logging

# 设置日志记录
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@app.post("/check-and-insert-metrics")
async def check_and_insert_metrics():
    try:
        inserter = MetricsInserter(host='localhost', user='heihuhu', password='1234', db='db')
        inserter.insert_metrics_from_folder('./log/exp/')
        inserter.close()
        return {"status": "success", "message": "Metrics checked and inserted successfully"}
    except Exception as e:
        logger.error(f"Error checking and inserting metrics: {e}", exc_info=True)
        return {"status": "error", "message": str(e)}


# TODO:UPDATE12.16插入TTC
from FastApi.metrics_ttc_inserter import MetricsTTCInserter


# 新增路由：检查并插入 TTC 数据
@app.post("/check-and-insert-ttc")
async def check_and_insert_ttc():
    try:
        ttc_inserter = MetricsTTCInserter(host='localhost', user='heihuhu', password='1234', db='db')
        ttc_inserter.connect()
        ttc_inserter.insert_metrics_from_folder('./log/exp/')
        ttc_inserter.close()
        return {"status": "success", "message": "TTC metrics checked and inserted successfully"}
    except Exception as e:
        logger.error(f"Error checking and inserting TTC metrics: {e}", exc_info=True)
        return {"status": "error", "message": str(e)}


# TODO:UPDATE1223：新增WebSocket路由用于视频流
from fastapi.responses import StreamingResponse
from FastApi.liveStreamFFmpeg import generate_frames_camera


@app.get('/video_feed')
async def video_feed():
    return StreamingResponse(generate_frames_camera(), media_type='multipart/x-mixed-replace; boundary=frame')


# TODO：UPDATE1225：新增WebSocket路由用于监视窗口
from FastApi.liveStreamFFmpeg import generate_frames as frame_ff


@app.get('/stream/{window_name}')
def stream(window_name: str):
    # 返回MJPEG流响应
    response = StreamingResponse(content=frame_ff(window_name),
                                 media_type='multipart/x-mixed-replace; boundary=frame')
    return response


@app.post("/vehicle-data/")
async def receive_vehicle_data(vehicle_data: dict):
    try:
        print(f"Received vehicle data: {vehicle_data}")
        await send_message(json.dumps(vehicle_data), active_connections)
        return {"status": "success", "data": vehicle_data}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# UPDATE0110：Carla画面传输与显示
# from FastApi.liveStreamXlibNew import generate_frames as frame_Xlib # N
from FastApi.liveStreamFFmpeg import generate_frames as frame_Xlib  # TODO


# from FastApi.liveStreamImportActivate import generate_frames as frame_Xlib # Y
# from FastApi.liveStreamXwd import generate_frames as frame_Xlib # Y
# from FastApi.liveStreamImportNoneActi import generate_frames as frame_Xlib # Y
@app.get('/stream_Carla/{window_name}')
def stream_Carla(window_name: str):
    # 返回MJPEG流响应
    response = StreamingResponse(content=frame_Xlib(window_name),
                                 media_type='multipart/x-mixed-replace; boundary=frame')
    return response


# UPDATE0113：
from fastapi import UploadFile, File
from fastapi.responses import JSONResponse
from collections import deque
import threading

image_queues = {
    "camera": deque(maxlen=10),
    "lidar": deque(maxlen=10),
    "semantic": deque(maxlen=10),
    "multicamera": [deque(maxlen=10) for _ in range(4)],
    "stitched": deque(maxlen=10)
}

async def frame_generator(img_queue):
    while True:
        if img_queue:
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + img_queue.popleft() + b'\r\n')
        else:
            await asyncio.sleep(0.03)  # 等待新图像

from PIL import Image
from io import BytesIO
async def combine_images():
    while True:
        images = []
        for q in image_queues["multicamera"]:
            try:
                images.append(q.popleft())
            except IndexError:
                await asyncio.sleep(0.03)  # 等待更多图像
                continue

        if len(images) != 4:
            continue

        img1 = Image.open(BytesIO(images[0]))
        img2 = Image.open(BytesIO(images[1]))
        img3 = Image.open(BytesIO(images[2]))
        img4 = Image.open(BytesIO(images[3]))

        width, height = img1.size
        new_width = 2 * width
        new_height = 2 * height

        combined_img = Image.new('RGB', (new_width, new_height))

        combined_img.paste(img1, (0, 0))
        combined_img.paste(img2, (width, 0))
        combined_img.paste(img3, (0, height))
        combined_img.paste(img4, (width, height))

        img_bytes = BytesIO()
        combined_img.save(img_bytes, format="JPEG")
        img_str = img_bytes.getvalue()

        image_queues["stitched"].append(img_str)
        print("组合图像已放入队列")


# UPDARE0117：图像的传输时间性能优化
@app.post("/upload-image/{sensor_type}/{index}")
async def upload_image(sensor_type: str, index: int, file: UploadFile = File(...)):
    contents = await file.read()
    if sensor_type == "multicamera":
        if 0 <= index < len(image_queues["multicamera"]):
            image_queues["multicamera"][index].append(contents)
            return JSONResponse(content={"message": f"multicamera{index}图像已接收", "filename": file.filename})
        else:
            return JSONResponse(content={"error": "Invalid multicamera index"}, status_code=400)
    elif sensor_type in image_queues:
        image_queues[sensor_type].append(contents)
        return JSONResponse(content={"message": f"{sensor_type}图像已接收", "filename": file.filename})
    else:
        return JSONResponse(content={"error": "Invalid sensor type"}, status_code=400)


# UPDARE0117：图像的传输时间性能优化
@app.get('/stream_Carla_new/lidar')
async def stream_lidar():
    return StreamingResponse(frame_generator(image_queues["lidar"]),
                             media_type='multipart/x-mixed-replace; boundary=frame')


@app.get('/stream_Carla_new/semantic')
async def stream_semantic():
    return StreamingResponse(frame_generator(image_queues["semantic"]),
                             media_type='multipart/x-mixed-replace; boundary=frame')


@app.get('/stream_Carla_new/camera')
async def stream_camera():
    return StreamingResponse(frame_generator(image_queues["camera"]),
                             media_type='multipart/x-mixed-replace; boundary=frame')


@app.get('/stream_Carla_new/stitched')
async def stream_stitched():
    loop = asyncio.get_event_loop()
    loop.create_task(combine_images())
    return StreamingResponse(frame_generator(image_queues["stitched"]),
                             media_type='multipart/x-mixed-replace; boundary=frame')

@app.get('/stream_Carla_new/multi-camera/{index}')
async def stream_multicamera(index: int):
    return StreamingResponse(frame_generator(image_queues["multicamera"][index]),
                             media_type='multipart/x-mixed-replace; boundary=frame')


class VehicleData(BaseModel):
    timestamp: float
    velocity_modulus: float
    acceleration_modulus: float
    ttc: Optional[float]  # 允许 ttc 为 None


# 存储最近十秒的数据
data_queue = deque(maxlen=100)  # 假设每秒发送10条数据


@app.post("/upload-vehicle-data/")
async def upload_vehicle_data(request: Request, data: VehicleData):
    # 添加数据到队列
    data_queue.append(data.dict())
    return {"message": "Vehicle data received successfully"}


@app.get("/get-vehicle-data/")
async def get_vehicle_data():
    # 返回当前存储的所有数据
    return list(data_queue)


if __name__ == "__main__":
    import uvicorn

    # 打印当前环境变量中的 DISPLAY
    print(f"Running with DISPLAY: {os.environ.get('DISPLAY')}")

    uvicorn.run(app, host='0.0.0.0', port=8001)  # 启动FastAPI应用，监听所有接口的8001端口

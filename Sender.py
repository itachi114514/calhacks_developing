import websockets
import struct
from math import pi
import asyncio
import struct

class WebSocketServer:
    """
    一个最小的 WebSocket 服务端实现类。
    当客户端连接时，会发送一条消息给它。
    """

    def __init__(self, host, port):
        """
        初始化服务器地址和端口。
        :param host: 服务器主机名 (例如 'localhost' 或 '0.0.0.0')
        :param port: 服务器端口号 (例如 8765)
        """
        self.host = host
        self.port = port
        self.servo_amount = 16
        self.message_queue = asyncio.Queue(3)
        self.clients = set()  # 用于存储所有连接的客户端
        self.servo_mapping_info = ServoMappingInfo()
        self.index_npy = 0
        # self.visualizer = RealtimeMatrixVisualizer(z_max=90, colormap='viridis')
        print(f"WebSocket 服务器准备在 ws://{self.host}:{self.port} 启动")

    async def _handler(self, websocket):
        """
        处理单个 WebSocket 连接的内部方法。
        :param websocket: 代表客户端连接的 WebSocket 对象。
        """
        print(f"客户端已连接: {websocket.remote_address}")
        self.clients.add(websocket)
        try:
            await websocket.wait_closed()
        except websockets.exceptions.ConnectionClosed as e:
            print(f"客户端连接已关闭: {websocket.remote_address}，原因: {e}")
        finally:
            self.clients.remove(websocket)
            print(f"与 {websocket.remote_address} 的连接已终止")

    async def _queue_processor(self):
        """
        核心方法：持续等待队列直到有消息出现，然后出队并广播。
        这是一个永久运行的后台任务。
        """
        print("队列处理器已启动，等待消息...")
        while True:
            # 等待队列中有新消息。这行代码会在此暂停，直到有消息为止。
            message = await self.message_queue.get()
            print(message)
            try:
                processed_message = self.process_message(message)
            except ValueError as e:
                print(f"处理消息时出错: {e}")
                self.message_queue.task_done()
                continue
            print(processed_message)
            if self.clients:
                print(self.clients)
                tasks = [client.send(processed_message) for client in self.clients]
                await asyncio.gather(*tasks, return_exceptions=True)
                print("消息已广播给所有客户端")
            # 标记任务已完成
            self.message_queue.task_done()

    def process_message(self, message):
        servo_commands = [0] * self.servo_amount
        for obj in message:
            servo_ids, angle = self.servo_mapping_info.process_object(obj)
            for servo_id in servo_ids:
                servo_commands[servo_id] = min(servo_commands[servo_id], int(angle)) if servo_commands[servo_id] !=0 else int(angle)
        np.save(f"./servo_npy/servos.npy{self.index_npy:06d}", np.array(servo_commands))
        self.index_npy += 1
        return struct.pack(f'!{self.servo_amount}d', *servo_commands)

    async def put_message(self, message: list[tuple]):
        """
        将消息放入队列以便发送给所有连接的客户端。
        :param message: 要发送的消息，可以是任何可序列化为 JSON 的 Python 对象。
        """
        await self.message_queue.put(message)
        # print(f"消息已放入队列: {message}")

    async def start(self):
        """
        启动 WebSocket 服务器并持续运行。
        """
        # 使用 async with 语句来确保服务器在完成后能被正确关闭
        # async with websockets.serve(self._handler, self.host, self.port):
        #     print(f"服务器已成功启动，正在监听 ws://{self.host}:{self.port}")
        #     # 使用 await asyncio.Future() 使服务器永久运行
        #     await asyncio.Future()
        # 将队列处理器作为后台任务启动
        queue_processor_task = asyncio.create_task(self._queue_processor())

        async with websockets.serve(self._handler, self.host, self.port):
            print(f"服务器已成功启动，正在监听 ws://{self.host}:{self.port}")
            # 使用 await asyncio.Future() 使服务器永久运行，直到被外部中断
            await asyncio.Future()

    def run(self):
        """
        运行服务器的同步入口点。
        """
        try:
            asyncio.run(self.start())
        except KeyboardInterrupt:
            print("\n服务器正在关闭...")

class ServoMappingInfo:
    def __init__(self, num_sectors=4, start_angle=-pi/4, end_angle=pi/4):
        self.object_map = {
            0: [1, 1, 1, 1],    # 人
            1: [1, 1, 1, 0],    # 自行车
            2: [1, 1, 0, 1],    # 汽车
            3: [1, 0, 1, 1],    # 摩托车
            5: [0, 1, 1, 1],    # 公交车
            7: [1, 1, 0, 0],    # 卡车
            60: [1, 0, 1, 0],   # 桌子
            56: [0, 1, 0, 1],   # 椅子
            63: [1, 0, 0, 1],   # 笔记本电脑
            67: [0, 1, 1, 0],   # 手机
            47: [0, 0, 1, 1],   # 苹果
        }
        # --- 动态配置方向和扇区 ---
        if num_sectors <= 0:
            raise ValueError("扇区数量 (num_sectors) 必须是正整数。")
        self.num_sectors = num_sectors
        self.start_angle = start_angle
        self.end_angle = end_angle
        self.total_angle_range = self.end_angle - self.start_angle
        self.angle_interval = self.total_angle_range / self.num_sectors

        self.min_distance = 0
        self.max_distance = 250
        self.min_servo_angle = 0
        self.max_servo_angle = 60

        self.angle_history = [[] for _ in range(self.num_sectors)] # 每个扇区的历史角度列表

    def which_direction(self, theta: float) -> int:
        """根据 theta 确定使用哪个方向的扇区索引。"""
        # 将 theta 角度平移，使其从0开始
        theta_offset = theta - self.start_angle

        # 根据平移后的角度计算扇区索引
        index = theta_offset // self.angle_interval

        # 边界检查，确保索引在有效范围内 [0, num_sectors-1]
        index = max(0, min(index, self.num_sectors - 1))
        return int(index)

    def which_in_four(self, cls_id):
        # 根据类别 ID 确定在四个舵机中的哪几个
        return self.object_map.get(cls_id)

    def which_servos(self, theta):
        index = self.which_direction(theta)
        servo_id = index * 4
        return servo_id

    def get_servo_ids(self, class_id, theta):
        index = self.which_direction(theta)
        in_four = self.which_in_four(class_id)
        servo_ids = []
        for i in range(4):
            if in_four[i]:
                servo_ids.append(index * 4 + i)
        print(servo_ids)
        return servo_ids, index

    def map_distance_to_servo_angle(self, distance):
        """将距离线性映射到舵机角度（反向映射）。"""
        min_input = self.min_distance
        max_input = self.max_distance
        min_angle = self.min_servo_angle
        max_angle = self.max_servo_angle

        distance_clamped = max(min_input, min(distance, max_input))

        # --- 核心修改：反转映射逻辑 ---
        # 之前: (distance_clamped - min_input) -> 距离越小，值越小
        # 现在: (max_input - distance_clamped) -> 距离越小，值越大
        angle_range = max_angle - min_angle
        distance_range = max_input - min_input

        angle = ((max_input - distance_clamped) / distance_range) * angle_range + min_angle

        return angle

    def process_object(self, object):
        class_id = object[0]
        distance = object[1]
        theta = object[2]
        servo_ids, index = self.get_servo_ids(class_id, theta)
        angle = self.map_distance_to_servo_angle(distance)
        # if self.angle_history[index] != []:
        #     angle = self.smooth_filter(self.angle_history[index][-1], angle)
        # angle = self.history_filter(self.angle_history[index], angle)
        return servo_ids, angle

    def smooth_filter(self, previous_angle, current_angle, alpha=0.7):
        """应用指数平滑滤波器来平滑舵机角度变化。"""
        smoothed_angle = alpha * previous_angle + (1 - alpha) * current_angle
        return smoothed_angle

    def history_filter(self, angle_history, current_angle):
        """使用历史角度列表来计算平滑后的舵机角度。"""
        angle_history.append(current_angle)
        if len(angle_history) > 5:
            angle_history.pop(0)
        smoothed_angle = sum(angle_history) / len(angle_history)
        return smoothed_angle


import matplotlib

# *** 关键改动 1: 显式设置后端 ***
# 必须在导入 pyplot 之前调用此行
# TkAgg 是一个可靠的选择，因为它通常随 Python 一起安装
# 如果您安装了 PyQt 或 PySide, 也可以使用 'Qt5Agg'
matplotlib.use('TkAgg')

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.colors as colors
import matplotlib.cm as cm
import time
import random


class RealtimeMatrixVisualizer:
    """
    一个用于实时可视化4x4矩阵数据的类（健壮版本）。
    主程序可以创建一个该类的实例，并反复调用 update() 方法来推送新数据。
    """

    def __init__(self, z_max=90, colormap='plasma'):
        """
        初始化可视化窗口和设置。
        """
        # 1. 设置图表和3D坐标系
        self.fig = plt.figure(figsize=(10, 8))
        self.ax = self.fig.add_subplot(111, projection='3d')

        # 开启交互模式
        plt.ion()

        # 2. 保存参数和设置
        self.z_max = z_max
        self.frame_count = 0

        # 设置颜色映射
        self.norm = colors.Normalize(vmin=0, vmax=z_max)
        self.cmap = cm.get_cmap(colormap)

        # 3. 预先计算好柱状图的网格位置
        _x = np.arange(4)
        _y = np.arange(4)
        _xx, _yy = np.meshgrid(_x, _y)
        self.x, self.y = _xx.ravel(), _yy.ravel()
        self.bottom = np.zeros_like(self.x)
        self.width = self.depth = 0.8

        # 4. 显示一个空的初始窗口
        self.fig.show()
        # 立即处理一次事件，确保窗口出现
        plt.pause(0.1)

    def update(self, data_list):
        """
        接收新的数据列表并更新图表。

        参数:
        data_list (list): 一个长度为16的列表，包含0到z_max的数值。
        """
        # 1. 数据验证
        if not isinstance(data_list, list) or len(data_list) != 16:
            print(f"错误: 输入数据必须是一个长度为16的列表，但收到了长度为 {len(data_list)} 的数据。")
            return

        # 2. 清除当前坐标轴上的所有图形
        self.ax.cla()

        # 3. 准备绘图数据
        dz = np.array(data_list)
        bar_colors = self.cmap(self.norm(dz))

        # 4. 绘制新的3D柱状图
        self.ax.bar3d(self.x, self.y, self.bottom, self.width, self.depth, dz, color=bar_colors, shade=True)
        self.frame_count += 1

        # 5. 重新设置坐标轴属性
        self.ax.set_title(f'4x4 Realtime Matrix Bar Graph (Frame {self.frame_count})')
        self.ax.set_xlabel('Column')
        self.ax.set_ylabel('Row')
        self.ax.set_zlabel('Value')
        self.ax.set_zlim(0, self.z_max)
        self.ax.set_xticks(np.arange(4))
        self.ax.set_yticks(np.arange(4))

        # *** 关键改动 2: 使用更可靠的刷新方法 ***
        # 命令画布重绘图形
        self.fig.canvas.draw()
        # 处理所有挂起的GUI事件，使更新在屏幕上可见
        self.fig.canvas.flush_events()


# --- 使用示例 ---
if __name__ == '__main__':
    # 1. 创建可视化类的实例
    visualizer = RealtimeMatrixVisualizer(z_max=90, colormap='viridis')

    # 2. 你的主循环
    try:
        # 模拟运行100次更新
        for i in range(100):
            # 从你的数据源获取新的列表
            new_data = [random.uniform(0, 90) for _ in range(16)]

            print(f"正在更新第 {i + 1} 帧...")

            # 调用update方法，将新数据推送给可视化对象
            visualizer.update(new_data)

            # 你的主循环可以执行其他任务
            # 注意：这里的 time.sleep() 是模拟数据生成间隔，它与绘图刷新无关
            time.sleep(0.2)

    except Exception as e:
        # 捕获可能的窗口关闭等异常
        print(f"程序结束或发生错误: {e}")
    finally:
        print("主循环结束。")
        # 让窗口保持打开状态，直到用户手动关闭
        plt.ioff()
        plt.show()
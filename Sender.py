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
        self.message_queue = asyncio.Queue()
        self.clients = set()  # 用于存储所有连接的客户端
        self.servo_mapping_info = ServoMappingInfo()
        print(f"WebSocket 服务器准备在 ws://{self.host}:{self.port} 启动")

    async def _handler(self, websocket):
        """
        处理单个 WebSocket 连接的内部方法。
        :param websocket: 代表客户端连接的 WebSocket 对象。
        :param path: 客户端请求的路径。
        """
        print(f"客户端已连接: {websocket.remote_address}")
        self.clients.add(websocket)
        try:
            await websocket.wait_closed()
        except websockets.exceptions.ConnectionClosed as e:
            print(f"客户端连接已关闭: {websocket.remote_address}，原因: {e}")
        finally:
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

            processed_message = self.process_message(message)
            print(processed_message)
            await asyncio.gather(*[client.send(processed_message) for client in self.clients])

            # 标记任务已完成
            self.message_queue.task_done()

    def process_message(self, message):
        servo_commands = [0] * 20
        for obj in message:
            servo_ids, angle = self.servo_mapping_info.process_object(obj)
            for servo_id in servo_ids:
                servo_commands[servo_id] = min(servo_commands[servo_id], int(angle)) if servo_commands[servo_id] !=0 else int(angle)
        return struct.pack('!20d', *servo_commands)

    async def put_message(self, message: list[tuple]):
        """
        将消息放入队列以便发送给所有连接的客户端。
        :param message: 要发送的消息，可以是任何可序列化为 JSON 的 Python 对象。
        """
        await self.message_queue.put(message)
        print(f"消息已放入队列: {message}")

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

        # 启动 WebSocket 服务器
        server = await websockets.serve(self._handler, self.host, self.port)
        print(f"服务器已成功启动，正在监听 ws://{self.host}:{self.port}")

        # 等待服务器和队列处理器任务完成（实际上会永久运行）
        await asyncio.gather(server.wait_closed(), queue_processor_task)

    def run(self):
        """
        运行服务器的同步入口点。
        """
        try:
            asyncio.run(self.start())
        except KeyboardInterrupt:
            print("\n服务器正在关闭...")

class ServoMappingInfo:
    def __init__(self):
        self.angle_interval = 180/5 # 每个舵机覆盖的角度范围
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

    def which_direction(self, theta):
        # 根据 theta 确定使用哪个方向的舵机
        index = theta  // self.angle_interval
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
        return servo_ids

    def map_distance_to_servo_angle(self, distance):
        # 将距离映射到舵机角度
        min_input = 0
        max_input = 1000
        min_angle = 0
        max_angle = 90
        if distance < min_input or distance > max_input:
            raise ValueError(f"Value {distance} out of range ({min_input}-{max_input})")
        angle = (distance - min_input) / (max_input - min_input) * (max_angle - min_angle) + min_angle
        return angle

    def process_object(self, object):
        class_id = object[0]
        distance = object[1]
        theta = object[2]
        servo_ids = self.get_servo_ids(class_id, theta)
        angle = self.map_distance_to_servo_angle(distance)
        return servo_ids, angle
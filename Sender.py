import websockets
import struct
from math import pi
import asyncio
import json


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

            print(f"从队列中取出消息: {message}，准备广播给 {len(self.clients)} 个客户端。")
            # 将 Python 对象序列化为 JSON 字符串以便传输
            json_message = json.dumps(message)
            await asyncio.gather(*[client.send(json_message) for client in self.clients])

            # 标记任务已完成
            self.message_queue.task_done()

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


# --- 如何使用这个类 ---
if __name__ == "__main__":
    # 创建服务器实例，监听本地所有网络接口的 8765 端口
    server = WebSocketServer('0.0.0.0', 8765)
    # 启动服务器
    server.run()

class Sender:
    def __init__(self, host='localhost', port=12345):
        asyncio.create_task(self.start_server())


    def send_message(self, message):
        try:
            self.sock.sendto(message.encode(), (self.host, self.port))
            print(f"Message sent to {self.host}:{self.port}")
        except Exception as e:
            print(f"Error sending message: {e}")

    def close(self):
        self.sock.close()

    def convert_to_servo_angle(self, distance, min_input=0, max_input=10000, min_angle=0, max_angle=90):
        # 将输入值线性映射到舵机角度范围
        if distance < min_input or distance > max_input:
            raise ValueError(f"Value {distance} out of range ({min_input}-{max_input})")
        angle = (distance - min_input) / (max_input - min_input) * (max_angle - min_angle) + min_angle

        return angle

    def convert_to_servo_id(self, id, min_id=0, max_id=15):
        if id <= 15:
            board = 0 # Board A
        else:
            board = 1 # Board B
            id -= 16
        return board << 4 | id

    def convert_class_id(self, class_id):
        classes = [0, 1, 2, 3, 5, 7, 60, 56, 63, 67]
        assert class_id in classes, f"Class id {class_id} not in {classes}"
        # return classes.index(class_id)<<5
        return classes.index(class_id)

    def convert_theta_to_servo_ids(self, theta):
        # mock logic
        id0 = theta//45
        return [id0+i for i in range(4)]

    def encode_message(self, message):
        for obj in message:
            class_id = self.convert_class_id(obj[0])
            angle = self.convert_to_servo_angle(obj[1])
            theta = angle * 180.0 / pi
            servo_ids = self.convert_theta_to_servo_ids(theta)
        # Some logic here
        encoded_message = ["something"]
        struct.pack_into("<I", encoded_message, *servo_ids)

    def send(self, message):
        self.send_message(message)

    def encode_and_send(self, message):
        byte_array = bytearray()
        for obj in message:
            id_and_channel = self.convert_to_servo_id(obj['id'])
            class_id = self.convert_class_id(obj['class_id'])
            angle = int(self.convert_to_servo_angle(obj['distance']))
            byte_array.extend(struct.pack('BBB', id_and_channel, class_id, angle))
        self.send_message(byte_array.decode('latin1'))
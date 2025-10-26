import asyncio
import websockets
import struct

class WebSocketClient:
    """
    一个最小的 WebSocket 客户端实现类。
    它会连接到指定的服务器，并持续接收和打印消息。
    """

    def __init__(self, uri: str, callback: callable):
        """
        初始化客户端。
        :param uri: 要连接的 WebSocket 服务器的 URI (例如 'ws://localhost:8765')
        """
        self.callback = callback
        self.uri = uri
        print(f"准备连接到 WebSocket 服务器: {self.uri}")

    async def _listen(self):
        """
        连接到服务器并进入主监听循环。
        这个方法会一直运行，直到连接关闭。
        """
        # 使用 async with 语句来自动管理连接的建立和关闭
        while True:
            try:
                async with websockets.connect(self.uri, ping_interval=5, ping_timeout=1) as websocket:
                    print(f"已成功连接到 {self.uri}")
                    # 使用 async for 循环来优雅地处理接收到的消息
                    # 这个循环会持续等待，直到有新消息到达
                    while True:
                        try:
                            async for message in websocket:
                                message = self.unpack_msg(message)
                                print(f"收到消息: {message}")
                                self.callback(message)
                        except websockets.exceptions.ConnectionClosedOK:
                            print("连接已被服务器正常关闭。")
                            break
                        except Exception as e:
                            print(f"处理消息时发生错误: {e}")

            except ConnectionRefusedError:
                print(f"连接被拒绝。请确保服务端正在 {self.uri} 上运行。")
            except websockets.exceptions.ConnectionClosed as e:
                # 当服务器关闭或连接意外断开时，会进入这里
                print(f"与服务器的连接已关闭。代码: {e.code}, 原因: {e.reason}")
            except Exception as e:
                print(f"发生了一个未知错误: {e}")
            await asyncio.sleep(1)

    def unpack_msg(self, msg):
        return struct.unpack("!16d", msg)

    def run(self):
        """
        运行客户端的同步入口点。
        """
        try:
            # 启动 asyncio 事件循环并运行 _listen 协程
            asyncio.run(self._listen())
        except KeyboardInterrupt:
            self.callback([0]*16)
            # 允许用户通过 Ctrl+C 来优雅地停止客户端
            print("\n客户端正在关闭...")


# --- 如何使用这个类 ---
if __name__ == "__main__":
    # 设置你要连接的服务器地址
    # 如果你的服务端运行在另一台机器上，请将 'localhost' 替换为相应的 IP 地址
    SERVER_URI = "ws://localhost:8765"

    # from servo import MultiBoardServoController
    # controller = MultiBoardServoController()
    def func(msg):
        msg = struct.unpack("!20d", msg)
        print(f"处理消息: {msg}")
    # 创建客户端实例
    client = WebSocketClient(SERVER_URI, callback=func)

    # 启动客户端
    client.run()
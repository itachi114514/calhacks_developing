import time
from typing import Tuple, Dict

from board import SCL, SDA
import busio
from adafruit_pca9685 import PCA9685
from adafruit_motor import servo

class MultiBoardServoController:
    """
    管理多个 PCA9685 板与多路舵机。
        - boards: 形如 {"A": 0x40, "B": 0x41}
        - channels: 为每块板需要控制的通道列表，如 {"A": [0,1], "B": [0,1]}
        - min_pulse/max_pulse: 舵机脉宽微调（us），默认 500~2500 覆盖常见舵机
    """
    def __init__(
            self,
            boards: Dict[str, int] = None,
            channels: Dict[str, list] = None,
            frequency:int=50,
            min_pulse: int = 500, max_pulse: int = 2500
    ):
        self.boards = {
            "A": 0x40,
            "B": 0x41,
        } if not boards else boards

        self.channels = {
            "A": list(range(16)),
            "B": list(range(16)),
        } if not channels else channels
        #Initialize I2C
        self.i2c = busio.I2C(SCL, SDA)

        #Initialize PCA9685 boards
        self.pcas:Dict[str, PCA9685] = {}
        for name, addr in self.boards.items():
            p = PCA9685(self.i2c, address=addr)
            p.frequency = frequency
            self.pcas[name] = p
            print(f"[INFO] PCA9685 board '{name}' at 0x{addr:02X} initialized (freq={frequency}Hz)")

        #Initialize servo channels
        self.servos:Dict[Tuple[str,int], servo.Servo] = {}
        for board_name, channel_list in self.channels.items():
            if board_name not in self.pcas:
                raise ValueError(f"[ERROR] Board '{board_name}' not found in boards dict")
            for channel in channel_list:
                s = servo.Servo(self.pcas[board_name].channels[channel], min_pulse=min_pulse, max_pulse=max_pulse)
                self.servos[(board_name, channel)] = s
                print(f"[INFO] Servo created on board '{board_name}' channel {channel} "
                      f"(min_pulse={min_pulse}us, max_pulse={max_pulse}us)")


    def set_angle(self, board_name:str, channel:int, angle:float):
        """Set the servo angle (0-180 degrees)."""
        key = (board_name, channel)
        if key not in self.servos:
            raise ValueError(f"[ERROR] Servo on board '{board_name}' channel {channel} not found in servos dict")
        if not (0 <= angle <= 180):
            raise ValueError(f"[ERROR] Angle must be in 0-180 degrees, got {angle}")
        self.servos[key].angle = angle
        print(f"[OK] ({board_name}, ch{channel}) -> angle = {angle:.1f}°")

    def update_angles(self, angles: list):
        """批量更新多个舵机角度，angles 为 [angle1, angle2, ...] 列表，元素为浮点数角度"""
        print("angles:", angles)
        print("A", angles[:8])
        print("B", angles[8:])
        for i, angle in enumerate(angles[:8]): # 前8个舵机，A板
            self.set_angle("A", i, angle)
        for i, angle in enumerate(angles[8:]): # 后12个舵机，B板
            self.set_angle("B", i, angle)

    def release(self, board_name:str, channel:int):
        """释放某路舵机（停止维持力）"""
        key = (board_name, channel)
        if key in self.servos:
            self.servos[key].angle = None
            print(f"[OK] ({board_name}, ch{channel}) released")


    def sweep(self, board_name:str, channel:int, step:int=10, delay:float=0.05):
        """单路舵机来回扫描（测试用）"""
        print(f"[TEST] Sweep ({board_name}, ch{channel})")
        for a in range(0, 181, step):
            self.set_angle(board_name, channel, a)
            time.sleep(delay)
        for a in range(180, -1, -step):
            self.set_angle(board_name, channel, a)
            time.sleep(delay)

    def stop_all(self):
        """释放所有舵机"""
        for (board_name, channel), s in self.servos.items():
            s.angle = None
        print("[OK] All servos released.")

    def close(self):
        """释放所有资源"""
        self.stop_all()
        for name, p in self.pcas.items():
            p.deinit()
            print(f"[INFO] PCA9685 '{name}' deinitialized.")
        print("[INFO] Controller closed.")

# Example usage:
if __name__ == "__main__":

    ctrl = MultiBoardServoController()

    try:
        # 简单自检：四路舵机依次到 0°、90°、180°、再回 90°
        sequence = [0, 90, 180, 90, 0]
        for angle in sequence:
            for board_name in ["A", "B"]:
                for channel in range(16):
                    try:
                        ctrl.set_angle(board_name, channel, angle)
                    except:
                        print("empty")
            time.sleep(0.8)
        # 释放所有舵机（不再维持力）
        ctrl.stop_all()

    except KeyboardInterrupt:
        print("\n[INTERRUPT] Stopped by user.")
    except Exception as e:
        print(f"\n[ERROR] {e}")
    finally:
        ctrl.close()


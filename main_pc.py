import cv2

from video_streamer import VideoStreamer
from detector import Detector
import asyncio
from math import sqrt
from numpy import arctan2
from Sender import WebSocketServer

import matplotlib

# 同样，显式设置后端以确保兼容性
matplotlib.use('TkAgg')

import numpy as np
import matplotlib.pyplot as plt
import time
import random
from collections import deque


class LineChartVisualizer:
    """
    一个用于实时可视化6个值（来自两个3元素元组）变化的类。
    """

    def __init__(self, labels=None, max_points=100, y_range=None):
        """
        初始化可视化窗口和设置。

        参数:
        labels (list of str, optional): 长度为6的字符串列表，用于图例。
                                        如果为None，则使用默认标签。
        max_points (int): 图表上显示的最大数据点数量，用于创建滚动窗口效果。
        y_range (tuple, optional): 一个 (min, max) 元组，用于固定Y轴的范围。
                                   如果为None，则Y轴会自动缩放。
        """
        # 1. 初始化图表和2D坐标系
        self.fig, self.ax = plt.subplots(figsize=(12, 7))
        plt.ion()  # 开启交互模式

        # 2. 数据存储
        self.max_points = max_points
        # 使用 collections.deque 可以高效地在列表两端添加和删除元素
        self.data_history = [deque(maxlen=max_points) for _ in range(6)]
        self.timesteps = deque(maxlen=max_points)
        self.current_step = 0

        # 3. 设置图表美学
        # 如果没有提供标签，则创建默认标签
        if labels is None or len(labels) != 6:
            self.labels = [f'Tuple1_Val{i + 1}' for i in range(3)] + \
                          [f'Tuple2_Val{i + 1}' for i in range(3)]
        else:
            self.labels = labels

        # 4. 创建6个空的折线对象
        # ax.plot返回一个列表，我们取第一个元素
        self.lines = [self.ax.plot([], [], label=self.labels[i])[0] for i in range(6)]

        # 5. 设置静态图表属性
        self.ax.legend()
        self.ax.grid(True)
        self.ax.set_title("Realtime Accel Gyro 6-Value Line Graph")
        self.ax.set_xlabel("Time Step")
        self.ax.set_ylabel("Value")

        # 如果指定了Y轴范围，则设置它
        self.y_range = y_range
        if self.y_range:
            self.ax.set_ylim(self.y_range[0], self.y_range[1])

        # 6. 显示窗口
        self.fig.show()
        # plt.pause(0.1)

    def update(self, tuple1, tuple2):
        """
        接收两个新的3元素元组并更新折线图。

        参数:
        tuple1 (tuple or list): 包含3个数值的第一个元组。
        tuple2 (tuple or list): 包含3个数值的第二个元组。
        """
        # 1. 数据验证
        if len(tuple1) != 3 or len(tuple2) != 3:
            print("错误: 输入数据必须是两个长度为3的元组或列表。")
            return

        # 2. 合并数据并更新历史记录
        new_values = list(tuple1) + list(tuple2)
        self.timesteps.append(self.current_step)
        for i in range(6):
            self.data_history[i].append(new_values[i])

        self.current_step += 1

        # 3. 更新6条折线的数据
        for i in range(6):
            self.lines[i].set_data(self.timesteps, self.data_history[i])

        # 4. 重新计算并调整坐标轴范围
        # self.ax.relim() 会根据所有线条的数据重新计算限制
        self.ax.relim()
        # self.ax.autoscale_view() 会应用新的限制
        self.ax.autoscale_view()

        # 如果Y轴范围是固定的，则重新应用它 (autoscale可能会覆盖它)
        if self.y_range:
            self.ax.set_ylim(self.y_range[0], self.y_range[1])
        # 确保X轴范围与数据同步
        self.ax.set_xlim(min(self.timesteps), max(self.timesteps) if self.timesteps else 1)

        # 5. 强制刷新画布
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()


# # --- 使用示例 ---
# if __name__ == '__main__':
#     # 假设我们正在监控一个IMU传感器（加速度计和陀螺仪）
#     # 创建自定义标签
#     sensor_labels = ['Accel_X', 'Accel_Y', 'Accel_Z', 'Gyro_X', 'Gyro_Y', 'Gyro_Z']
#
#     # 创建可视化类的实例
#     # y_range=(-100, 100) 可以固定Y轴范围，让变化更直观
#     line_visualizer = LineChartVisualizer(labels=sensor_labels, max_points=100, y_range=(-100, 100))
#
#     # 你的主循环
#     try:
#         # 模拟运行200次更新
#         for i in range(200):
#             # 模拟从传感器获取数据
#             # 加速度数据在-50到50之间波动
#             accel_data = (random.uniform(-50, 50), random.uniform(-50, 50), random.uniform(-50, 50))
#             # 陀螺仪数据在-90到90之间波动
#             gyro_data = (random.uniform(-90, 90), random.uniform(-90, 90), random.uniform(-90, 90))
#
#             print(f"正在更新第 {i + 1} 帧...")
#
#             # 调用update方法，推送新数据
#             line_visualizer.update(accel_data, gyro_data)
#
#             # 模拟数据更新的间隔
#             time.sleep(0.1)
#
#     except Exception as e:
#         print(f"程序结束或发生错误: {e}")
#     finally:
#         print("主循环结束。")
#         plt.ioff()
#         plt.show()  # 保持窗口打开直到用户手动关闭

def to_polar(x, z):
    r = sqrt(x**2 + z**2)
    theta = arctan2(x, z)
    return r, theta

def get_point(points, cX, cY):
    try:
        point = points[cY*1280 + cX]
    except IndexError as e:
        print(f"IndexError: {e} for cX={cX}, cY={cY}")
        # return None, None
        exit(1)
    x, y, z = point
    return x, y, z

def random_point_in_mask(mask, center_x, center_y):
    # 填充mask轮廓内的区域
    h, w = mask.shape
    filled_mask = np.zeros((h, w), dtype=np.uint8)
    contours = [np.array(mask, dtype=np.int32)]
    cv2.fillPoly(filled_mask, pts=contours, color=1)
    mask = filled_mask
    # 获取mask中所有为1的点的坐标
    ys, xs = np.where(mask == 1)
    if len(xs) == 0 or len(ys) == 0:
        return None
    # 当选出的点不在mask内时，重新选点
    while True:
        # 计算每个点到中心点的距离
        distances = np.sqrt((xs - center_x) ** 2 + (ys - center_y) ** 2)
        # 使用距离的倒数作为权重，距离越近权重越大
        weights = 1 / (distances + 1e-6)  # 避免除以零
        weights /= np.sum(weights)  # 归一化权重
        # 根据权重随机选择一个点
        chosen_index = np.random.choice(len(xs), p=weights)
        if mask[ys[chosen_index], xs[chosen_index]] == 1:
            break
    return xs[chosen_index], ys[chosen_index]

async def main():
    isAnotate = True
    isSaving = True
    if isSaving:
        frame_index = 0
        import os
        import numpy as np
        SAVE_DIRECTORY = "point_cloud_data"
        fps = 30
        w = 1280
        h = 720
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter('output.mp4', fourcc, fps, (int(w), int(h)))
        out_depth = cv2.VideoWriter('depth.mp4', fourcc, fps, (int(w), int(h)))
    sensor_labels = ['Accel_X', 'Accel_Y', 'Accel_Z', 'Gyro_X', 'Gyro_Y', 'Gyro_Z']

    # 创建可视化类的实例
    # y_range=(-100, 100) 可以固定Y轴范围，让变化更直观
    # line_visualizer = LineChartVisualizer(labels=sensor_labels, max_points=100, y_range=(-100, 100))

    Server = WebSocketServer('0.0.0.0', 8765)
    asyncio.create_task(Server.start())
    isCallback = True
    streamer = VideoStreamer(isCallback=isCallback)
    detector = Detector(model_path="yolo11x-seg.pt", isTracked=True, isAnnotate=isAnotate)
    if isSaving:
        index = 0
    try:
        while True:
            color_image, depth_image, points, accel, gyro = await streamer.get_frame()
            # line_visualizer.update(accel, gyro)
            if isSaving:
                np.save(f"./imu_npy/accelGyro.npy{index:06d}", np.array([*accel, *gyro]))
                index+=1
            if color_image is not None and depth_image is not None:
                detector.points = points
                img = detector.detect(color_image)
                cls_id = detector.get_class_id()
                if not cls_id:
                    continue
                points_2d = detector.calculate_centroid()
                points_2d_box_center = detector.get_box_center()
                assert len(points_2d_box_center) == len(points_2d)
                # print("cls_id=", cls_id)
                # print("points_2d=", points_2d)
                polars = []
                cls_id_new = []
                for i, (cX, cY) in enumerate(points_2d):
                    print("cx,cy=", cX, cY)
                    if cY * 1280 + cX >= 921600 or cY * 1280 + cX < 0:
                        continue
                    point = get_point(points, cX, cY)
                    if point == (0.0, 0.0, 0.0):
                        # 使用盒子中心点尝试获取更可靠的深度信息
                        cX_box, cY_box = points_2d_box_center[i]
                        point = get_point(points, int(cX_box), int(cY_box))
                        if point == (0.0, 0.0, 0.0):
                            # 从mask中随机选取，正态分布靠近中心，直到找到非零点
                            cX_rand, cY_rand = random_point_in_mask(detector.result.masks.xy[i], cX, cY)
                            point = get_point(points, int(cX_rand), int(cY_rand))
                            assert point != (0.0, 0.0, 0.0)

                    print("point=", point)
                    if point is None:
                        continue
                    x, _, z = point
                    r, theta = to_polar(float(x), float(z))
                    if r>250:
                        continue
                    cls_id_new.append(cls_id[i])
                    polars.append((r, theta))
                assert len(cls_id_new) == len(polars)
                zipped = [tuple([a, *b]) for a, b in zip(cls_id_new, polars)]
                await Server.put_message(zipped)
                if isAnotate:
                    cv2.imshow("Video Streamer", img)
                if points is None:
                    continue
                if isSaving:
                    # 保存视频
                    out.write(img)
                    out_depth.write(depth_image)
                    # 使用 zero-padding 格式化文件名，确保后续能正确排序
                    filename = f"frame_{frame_index:06d}.npy"
                    filepath = os.path.join(SAVE_DIRECTORY, filename)
                    np.save(filepath, points)
                    print(f"[{frame_index}] 已保存点云: {filepath}")
                    frame_index += 1
                if cv2.waitKey(1) & 0xFF == 27:
                    break
                # Sender.encode_and_send(zipped)

                # print(points.shape)
    finally:
        streamer.release()
asyncio.run(main())
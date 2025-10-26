import asyncio

from pyorbbecsdk import *
from utils import frame_to_bgr_image
import cv2
import numpy as np
from asyncio import Queue

MAX_QUEUE_SIZE = 2
frames_queue = Queue(MAX_QUEUE_SIZE)

video_sensor_types = [
    OBSensorType.DEPTH_SENSOR,
    OBSensorType.LEFT_IR_SENSOR,
    OBSensorType.RIGHT_IR_SENSOR,
    OBSensorType.IR_SENSOR,
    OBSensorType.COLOR_SENSOR
]

class VideoStreamer:
    def __init__(self, isCallback=True, isPointcloud=True, depthFilter=False, isAccel=True):
        self.window_width = 1280
        self.window_height = 720
        self.MIN_DEPTH = 20  # 20mm
        self.MAX_DEPTH = 10000  # 10000mm
        self.config = Config()
        self.pipeline = Pipeline()
        if isPointcloud:
            self.point_cloud_filter = PointCloudFilter()
            self.point_cloud_filter.set_create_point_format(OBFormat.POINT)
            self.point_cloud_filter.set_position_data_scaled(0.1)
            self.align_filter = AlignFilter(align_to_stream=OBStreamType.COLOR_STREAM)
            self.isPointCloud = True
        else:
            self.isPointCloud = False
        if not(isCallback):
            self.pipeline.start()
            self.get_frame = self.non_callback_get_frame
            return

        if isAccel:
            self.isAccel = True
            self.config.enable_accel_stream()
            self.config.enable_gyro_stream()
            self.config.set_frame_aggregate_output_mode(OBFrameAggregateOutputMode.FULL_FRAME_REQUIRE)
        else:
            self.isAccel = False
        self.loop = asyncio.get_running_loop()
        device = self.pipeline.get_device()
        sensor_list = device.get_sensor_list()
        for sensor in range(len(sensor_list)):
            sensor_type = sensor_list[sensor].get_type()
            if sensor_type in video_sensor_types:
                try:
                    print(f"Enabling sensor type: {sensor_type}")
                    self.config.enable_stream(sensor_type)
                except:
                    print(f"Failed to enable sensor type: {sensor_type}")
                    continue
        if depthFilter:
            self.depthFilter = True
            depth_sensor = device.get_sensor(OBSensorType.DEPTH_SENSOR)
            # 4.Get a list of post-processing filtering recommendations
            self.filter_list = depth_sensor.get_recommended_filters()
        else:
            self.depthFilter = False
        profiles = self.pipeline.get_stream_profile_list(OBSensorType.DEPTH_SENSOR)
        if profiles is None:
            raise RuntimeError("No proper depth profile, cannot start streaming")
        depth_profile = profiles.get_default_video_stream_profile()
        self.config.enable_stream(depth_profile)

        preset_list = device.get_available_preset_list()
        for i in range(len(preset_list)):
            print(f"Available preset: {preset_list[i]}")
        preset_name = preset_list[0]
        print(f"Using preset: {preset_name}")
        # Set preset
        device.load_preset(preset_name)

        # Find the depth work mode name through the list
        depth_work_mode_list = device.get_depth_work_mode_list()
        select_depth_work_mode = depth_work_mode_list.get_depth_work_mode_by_index(0)
        for i in range(len(depth_work_mode_list)):
            mode = depth_work_mode_list.get_depth_work_mode_by_index(i)
            print(f"Available depth work mode: {mode.name}, index: {i}")
            if mode.name == select_depth_work_mode.name:
                select_depth_work_mode = mode
        # Set depth work mode
        device.set_depth_work_mode(select_depth_work_mode.name)

        self.pipeline.start(self.config, self.on_new_frame_callback)

    def non_callback_get_frame(self):
        frames = self.pipeline.wait_for_frames(100)
        if frames is None:
            return None, None, None
        color_image, depth_image, points = self.process_frame(frames)
        return color_image, depth_image, points

    async def get_frame(self):
        frames = await frames_queue.get()
        color_image, depth_image, points = self.process_frame(frames)
        accel_data, gyro_data = self.get_accel_frame(frames)
        if self.isAccel:
            if accel_data is not None and gyro_data is not None:
                return color_image, depth_image, points, accel_data, gyro_data
            else:
                return color_image, depth_image, points, None, None
        return color_image, depth_image, points

    def get_accel_frame(self, frames):
        accel_frame = frames.get_frame(OBFrameType.ACCEL_FRAME)
        gyro_frame = frames.get_frame(OBFrameType.GYRO_FRAME)
        accel_frame = accel_frame.as_accel_frame() if accel_frame else None
        gyro_frame = gyro_frame.as_gyro_frame() if gyro_frame else None
        if not accel_frame or not gyro_frame:
            return None, None
        accel_data = (accel_frame.get_x(), accel_frame.get_y(), accel_frame.get_z())
        gyro_data = (gyro_frame.get_x(), gyro_frame.get_y(), gyro_frame.get_z())
        return accel_data, gyro_data

    def process_frame(self, frames):
        color_frame = frames.get_color_frame()
        depth_frame = frames.get_depth_frame()
        color_image = frame_to_bgr_image(color_frame) if color_frame else None
        depth_image = self.process_depth(depth_frame) if depth_frame else None
        color_image = cv2.resize(color_image, (self.window_width, self.window_height)) if color_image is not None else None
        depth_image = cv2.resize(depth_image, (self.window_width, self.window_height)) if depth_image is not None else None
        if self.depthFilter:
            for post_filter in self.filter_list:
                if post_filter.is_enabled():
                    # Only apply enabled filters
                    depth_frame = post_filter.process(depth_frame)
        if self.isPointCloud:
            align_frame = self.align_filter.process(frames)
            point_cloud_frame = self.point_cloud_filter.process(align_frame)
            points = self.point_cloud_filter.calculate(point_cloud_frame)
        else:
            points = None
        return color_image, depth_image, points

    def on_new_frame_callback(self, frame: FrameSet):
        """
        由 Orbbec SDK 的内部线程调用。
        使用 call_soon_threadsafe 将帧安全地传递给 asyncio 事件循环。
        """
        # 3. 这是核心修复：使用 call_soon_threadsafe
        # 它安排 _put_frame_in_queue 在事件循环线程中尽快被调用
        self.loop.call_soon_threadsafe(self._put_frame_in_queue, frame)

    def _put_frame_in_queue(self, frame: FrameSet):
        """这个方法将由 asyncio 事件循环在主线程中执行"""
        if frame is None:
            return
        # 因为这个方法在事件循环中运行，所以可以安全地操作 asyncio.Queue
        if frames_queue.full():
            # 丢弃最旧的帧以腾出空间
            frames_queue.get_nowait()
        frames_queue.put_nowait(frame)

    def process_depth(self, depth_frame):
        """Process depth frame to colorized depth image"""
        if not depth_frame:
            return None
        try:
            width = depth_frame.get_width()
            height = depth_frame.get_height()
            scale = depth_frame.get_depth_scale()
            depth_data = np.frombuffer(depth_frame.get_data(), dtype=np.uint16).reshape((height, width))
            depth_data = depth_data.astype(np.float32) * scale
            depth_data = np.where((depth_data > self.MIN_DEPTH) & (depth_data < self.MAX_DEPTH), depth_data, 0).astype(np.uint16)
            depth_image = cv2.normalize(depth_data, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
            depth_image = cv2.applyColorMap(depth_image, cv2.COLORMAP_JET)
            return depth_image
        except ValueError:
            print("Error processing depth frame")
            return None

    def release(self):
        self.pipeline.stop()

if __name__ == '__main__':
    async def main():
        isCallback = True
        isSaving = False
        streamer = VideoStreamer(isCallback=isCallback)
        if isSaving:
            frame_index = 0
            import os
            SAVE_DIRECTORY = "point_cloud_data"  # 用于存放点云文件的文件夹
        while True:
            if isCallback:
                color_image, depth_image, points = await streamer.get_frame()
            else:
                color_image, depth_image, points = streamer.get_frame()
            if color_image is not None and depth_image is not None:
                combined_image = np.hstack((color_image, depth_image))
                print(points.shape)
                cv2.imshow("Video Streamer", combined_image)
                if isSaving:
                    # 2. 保存点云数据 (新增的简短逻辑)
                    if points is not None and points.shape == (407040, 3):
                        # 使用 zero-padding 格式化文件名，确保后续能正确排序
                        filename = f"frame_{frame_index:06d}.npy"
                        filepath = os.path.join(SAVE_DIRECTORY, filename)
                        np.save(filepath, points)
                        print(f"[{frame_index}] 已保存点云: {filepath}")
                        frame_index += 1
            if cv2.waitKey(1) & 0xFF == 27:  # ESC key to exit
                break
        streamer.release()
        cv2.destroyAllWindows()
    asyncio.run(main())
import asyncio

from pyorbbecsdk import *
from utils import frame_to_bgr_image
import cv2
import numpy as np
from asyncio import Queue

frames_queue = Queue()
MAX_QUEUE_SIZE = 1


class VideoStreamer:
    def __init__(self):
        self.config = Config()
        self.pipeline = Pipeline()
        self.pipeline.start(Config(), self.on_new_frame_callback)
        self.window_width = 1280
        self.window_height = 720

    async def get_frame(self):
        frames = await frames_queue.get()
        color_frame = frames.get_color_frame()
        depth_frame = frames.get_depth_frame()
        color_image = frame_to_bgr_image(color_frame) if color_frame else None
        depth_image = self.process_depth(depth_frame) if depth_frame else None
        color_image = cv2.resize(color_image, (self.window_width // 2, self.window_height)) if color_image is not None else None
        depth_image = cv2.resize(depth_image, (self.window_width // 2, self.window_height)) if depth_image is not None else None
        return color_image, depth_image

    @staticmethod
    def on_new_frame_callback(frame: FrameSet):
        """Callback function to handle new frames"""
        global frames_queue
        if frame is None:
            return
        if frames_queue.qsize() >= MAX_QUEUE_SIZE:
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
        streamer = VideoStreamer()
        while True:
            color_image, depth_image = await streamer.get_frame()
            if color_image is not None and depth_image is not None:
                combined_image = np.hstack((color_image, depth_image))
                cv2.imshow("Video Streamer", combined_image)
            if cv2.waitKey(1) & 0xFF == 27:  # ESC key to exit
                break
        streamer.release()
        cv2.destroyAllWindows()
    asyncio.run(main())
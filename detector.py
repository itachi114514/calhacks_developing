from numpy.ma.core import arctan
from ultralytics import YOLO
from collections import defaultdict

from ultralytics import YOLO
import cv2
import numpy as np
import time


# 封装为类
class Detector:
    def __init__(self, model_path="yolo11x-seg.pt", isTracked=True, isAnnotate=True):
        self.model = YOLO(model_path)
        self.model.fuse()
        self.isTracked = isTracked
        self.isAnnotate = isAnnotate
        self.img = None  # 外部注入的图像，每次annotate调用后清空
        if isTracked:
            self.track_history = defaultdict(lambda: [])
        # 人，自行车，汽车，摩托车，公交车，卡车，桌子，椅子，笔记本电脑，手机，苹果
        self.interested_classes = [0, 1, 2, 3, 5, 7, 60, 56, 63, 67, 48]

    def annotate(self):
        frame = self.result.plot(img=self.img)
        self.img = None
        fps = 1 / (self.end_time - self.start_time)
        cv2.putText(frame, f'FPS: {fps:.2f}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        boxes = self.result.boxes.xywh.cpu()
        track_ids = self.result.boxes.id.int().cpu().tolist()
        for box, track_id in zip(boxes, track_ids):
            track = self.get_track_points(track_id)
            points = np.hstack(track).astype(np.int32).reshape((-1, 1, 2))
            cv2.polylines(frame, [points], isClosed=False, color=(0, 0, 255), thickness=2)
        return frame

    def update_track_history(self):
        if self.result.boxes and self.result.boxes.is_track:
            boxes = self.result.boxes.xywh.cpu()
            track_ids = self.result.boxes.id.int().cpu().tolist()
            for box, track_id in zip(boxes, track_ids):
                x, y, w, h = box
                track = self.track_history[track_id]
                track.append((float(x), float(y)))
                if len(track) > 50:
                    track.pop(0)

    def get_track_points(self, track_id):
        return self.track_history[track_id]

    def detect(self, frame):
        self.start_time = time.time()
        if self.isTracked:
            results = self.model.track(frame, persist=True, tracker="bytetrack.yaml", classes=self.interested_classes)
        else:
            results = self.model(frame, classes=self.interested_classes)
        self.end_time = time.time()

        self.result = results[0]
        self.class_ids = self.result.boxes.cls.int().cpu().tolist()
        self.update_track_history()
        if self.isAnnotate:
            annotated_frame = self.annotate()
            return annotated_frame
        else:
            return None

    def calculate_centroid(self):
        # 计算检测到的每个对象的质心，根据掩码
        centroid = []
        counters = self.result.masks.xy
        for i in counters:
            M = cv2.moments(np.array(i, dtype=np.int32))
            if M["m00"] != 0:
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])
                centroid.append((cX, cY))
            else:
                centroid.append((None, None))
        self.centorid = centroid
        return centroid

    def get_class_id(self):
        return self.class_ids

if __name__ == '__main__':
    from video_streamer import VideoStreamer
    import cv2
    import numpy as np
    import asyncio
    async def main():
        isCallback = True
        isSaving = False
        streamer = VideoStreamer(isCallback=isCallback)
        detector = Detector(model_path="yolo11x-seg.pt", isTracked=True)
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
                detector.img = depth_image.copy()
                annotated_frame = detector.detect(color_image)
                points_2d = detector.calculate_centroid()

                from math import sqrt
                from numpy import arctan2
                for (cX, cY) in points_2d:
                    point = points[cY*640 + cX]
                    x, y, z = point
                    r = sqrt(x**2 + z**2)
                    theta = arctan2(x, z)
                    print(r, theta)

                combined_image = np.hstack((annotated_frame, depth_image))
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
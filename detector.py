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
        self.points = None  # 外部注入的点云，每次detect调用后更新
        if isTracked:
            self.track_history = defaultdict(lambda: [])
        # 人，自行车，汽车，摩托车，公交车，卡车，桌子，椅子，笔记本电脑，手机，苹果
        self.interested_classes = [0, 1, 2, 3, 5, 7, 60, 56, 63, 67, 48]
        self.interested_classes = [0]
        self.boxes = None

    def annotate(self):
        frame = self.result.plot(img=self.img)
        self.img = None
        fps = 1 / (self.end_time - self.start_time)
        cv2.putText(frame, f'FPS: {fps:.2f}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        boxes = self.result.boxes.xywh.cpu()
        # 计算质心
        centroid = self.calculate_centroid()
        for box, track_id, (cx, cy) in zip(boxes, self.track_ids, centroid):
            track = self.get_track_points(track_id)
            points = np.hstack(track).astype(np.int32).reshape((-1, 1, 2))
            cv2.polylines(frame, [points], isClosed=False, color=(0, 0, 255), thickness=2)
            # 用绿色标注质心
            if cx is not None and cy is not None:
                cv2.circle(frame, (cx, cy), 5, (0, 255, 0), -1)
            # 计算r, theta并标注
            if cy is None:
                continue
            if cy*1280 + cx >= 921600 or cy*1280 + cx < 0:
                continue
            if self.points is not None and cx is not None and cy is not None:
                point = self.points[cy*1280 + cx]
                x, y, z = point
                r = np.sqrt(x**2 + z**2)
                theta = np.degrees(np.arctan2(x, z))
                cv2.putText(frame, f'ID:{track_id} R:{r:.1f} Theta:{theta:.1f}', (int(box[0]), int(box[1]) - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        return frame

    def update_track_history(self):
        if self.result.boxes and self.result.boxes.is_track:
            boxes = self.result.boxes.xywh.cpu()
            for box, track_id in zip(boxes, self.track_ids):
                x, y, w, h = box
                track = self.track_history[track_id]
                track.append((float(x), float(y)))
                if len(track) > 50:
                    track.pop(0)

    def get_box_center(self):
        centers = []
        boxes = self.result.boxes.xywh.cpu()
        for box in boxes:
            x, y, w, h = box
            centers.append((x, y))
        self.box_centers = centers
        return centers

    def get_track_points(self, track_id):
        return self.track_history[track_id]

    def get_track_id(self):
        return self.track_ids

    # 每次调用次函数会清除result中所有track id不为1的对象，只保留track id为1的对象
    def filter_track_id_one(self):
        if self.result.boxes and self.result.boxes.is_track:
            mask = [tid == 1 for tid in self.track_ids]
            self.result.boxes = self.result.boxes[mask]
            self.track_ids = [tid for tid in self.track_ids if tid == 1]
            self.class_ids = [cid for cid, m in zip(self.class_ids, mask) if m]
            # 更新track history，只保留track id为1的对象
            new_history = defaultdict(lambda: [])
            if 1 in self.track_history:
                new_history[1] = self.track_history[1]
            self.track_history = new_history

    def detect(self, frame):
        self.start_time = time.time()
        if self.isTracked:
            results = self.model.track(frame, persist=True, tracker="bytetrack.yaml", classes=self.interested_classes, verbose=False)
        else:
            results = self.model(frame, classes=self.interested_classes, verbose=False)
        self.end_time = time.time()

        if results[0].boxes.id is None:
            self.track_ids = None
            self.class_ids = None
            return None
        self.result = results[0]
        self.track_ids = self.result.boxes.id.int().cpu().tolist()
        self.class_ids = self.result.boxes.cls.int().cpu().tolist()
        self.update_track_history()
        # self.filter_track_id_one()
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
                detector.points = points
                annotated_frame = detector.detect(color_image)
                points_2d = detector.calculate_centroid()

                from math import sqrt
                from numpy import arctan2
                for (cX, cY) in points_2d:
                    if cX is None or cY is None:
                        continue
                    if cY * 1280 + cX >= 921600 or cY * 1280 + cX < 0:
                        continue
                    point = points[cY*1280 + cX]
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
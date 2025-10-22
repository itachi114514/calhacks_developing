from video_streamer import VideoStreamer
from detector import Detector
import asyncio
from math import sqrt
from numpy import arctan2
from Sender import WebSocketServer

def to_polar(x, z):
    r = sqrt(x**2 + z**2)
    theta = arctan2(x, z)
    return r, theta

def get_point(points, cX, cY):
    point = points[cY*640 + cX]
    x, y, z = point
    return x, y, z

async def main():
    Server = WebSocketServer('0.0.0.0', 8765)
    asyncio.create_task(Server.start())
    isCallback = True
    streamer = VideoStreamer(isCallback=isCallback)
    detector = Detector(model_path="yolo11x-seg.pt", isTracked=True, isAnnotate=False)
    try:
        while True:
            color_image, depth_image, points = await streamer.get_frame()
            if color_image is not None and depth_image is not None:
                detector.detect(color_image)
                cls_id = detector.get_class_id()
                if not cls_id:
                    continue
                points_2d = detector.calculate_centroid()
                print("cls_id=", cls_id)
                print("points_2d=", points_2d)
                polars = []
                for (cX, cY) in points_2d:
                    point = get_point(points, cX, cY)
                    x, _, z = point
                    r, theta = to_polar(float(x), float(z))
                    polars.append((r, theta))
                    print(r, theta)
                print("polars=", polars)
                zipped = [tuple([a, *b]) for a, b in zip(cls_id, polars)]
                await Server.put_message(zipped)
                # Sender.encode_and_send(zipped)

                print(points.shape)
    finally:
        streamer.release()
asyncio.run(main())
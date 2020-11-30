# # Create a black image
# img = np.zeros((512,512,3), np.uint8)
# # Draw a diagonal blue line with thickness of 5 px
# cv.line(img,(0,0),(511,511),(255,0,0),5)
# cv.rectangle(img,(384,0),(510,128),(0,255,0),3)
# cv.circle(img,(447,63), 63, (0,0,255), -1)
# cv.ellipse(img,(256,256),(100,50),0,0,180,255,-1)
# pts = np.array([[10,5],[20,30],[70,20],[50,10]], np.int32)
# pts = pts.reshape((-1,1,2))
# cv.polylines(img,[pts],True,(0,255,255))
# font = cv.FONT_HERSHEY_SIMPLEX
# cv.putText(img,'opencv',(10,500), font, 4,(255,255,255),2,cv.LINE_AA)


import time

import cv2
import numpy as np
from PIL import ImageFont, ImageDraw, Image


width = 1280
height = 720
FPS = 24
time_interval = 1.0 / FPS
video_filename = 'lab1.mp4'
preview_window_name = 'lab1'
video = cv2.VideoWriter(video_filename, cv2.VideoWriter_fourcc(*'mp4v'), float(FPS), (width, height))


class VideoGeneratorBase:
    def __init__(self, frame_cnt: int, frame_gen_func):
        self.frame_count = frame_cnt
        self.frame_gen_func = frame_gen_func

    def draw(self):
        for i in range(self.frame_count):
            frame_start_time = time.time()
            frame = self.frame_gen_func(i)
            cv2.imshow(preview_window_name, frame)
            video.write(frame)
            wait_time = time_interval - (time.time() - frame_start_time)
            if wait_time > 0 and cv2.waitKey(int(wait_time * 1000)) == 32:
                while cv2.waitKey(-1) != 32:
                    pass


class CircleGen(VideoGeneratorBase):
    def __init__(self):
        self.radius = 150
        self.paint_h = int(height / 2)
        super().__init__(int((width + 2 * self.radius) / 6), self.circle_draw_frame)

    def circle_draw_frame(self, frame_cnt):
        paint_x = -self.radius + 6 * frame_cnt
        frame = np.random.randint(0, 256,
                                  (height, width, 3),
                                  dtype=np.uint8)
        cv2.circle(frame, (paint_x, self.paint_h), self.radius, (0, 0, 0), -1)
        return frame


class TextGen(VideoGeneratorBase):
    def __init__(self):
        self.step = 4
        super().__init__(int(256 / self.step), self.text_draw_frame)
        self.fontpath = 'simsun.ttc'
        self.text = '曾充 3180106183'

    def text_draw_frame(self, frame_cnt):
        font = ImageFont.truetype(self.fontpath, 32)
        img_pil = Image.fromarray(np.zeros((height, width, 3), np.uint8))
        draw = ImageDraw.Draw(img_pil)
        draw.text((0, 0),  self.text, font=font, fill=(frame_cnt * self.step, frame_cnt * self.step, frame_cnt * self.step, 255))
        return np.array(img_pil)


if __name__ == '__main__':
    video_gen = CircleGen()
    text_gen = TextGen()
    text_gen.draw()
    video_gen.draw()
    cv2.destroyAllWindows()
    video.release()

# # Create a black image
#
# # Draw a diagonal blue line with thickness of 5 px
# cv.line(img,(0,0),(511,511),(255,0,0),5)
# cv.rectangle(img,(384,0),(510,128),(0,255,0),3)
# cv.circle(img,(447,63), 63, (0,0,255), -1)
# cv.ellipse(img,(256,256),(100,50),0,0,180,255,-1)
# pts = np.array([[10,5],[20,30],[70,20],[50,10]], np.int32)
# pts = pts.reshape((-1,1,2))
# cv.polylines(img,[pts],True,(0,255,255))



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
    def __init__(self):
        self.frame_count = None

    def frame_gen_func(self, frame_cnt):
        raise NotImplementedError()

    def draw(self):
        cv2.namedWindow(preview_window_name)
        cv2.moveWindow(preview_window_name, 0, 0)
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
    def __init__(self, frames):
        super().__init__()
        self.frame_count = frames

    def frame_gen_func(self, frame_cnt):
        frame = np.random.randint(0, 256,
                                  (height, width, 3),
                                  dtype=np.uint8)
        return frame


class TextGen(VideoGeneratorBase):
    def __init__(self, text, frames, font_file):
        super().__init__()
        self.frame_count = frames
        self.font_path = font_file
        self.text = text

    def frame_gen_func(self, frame_cnt):
        font = ImageFont.truetype(self.font_path, 32)
        img_pil = Image.fromarray(np.ones((height, width, 3), np.uint8) * 255)
        draw = ImageDraw.Draw(img_pil)
        w, h = draw.textsize(self.text, font=font)
        temp_val = 255 - int(255 * frame_cnt / self.frame_count)
        draw.text(((width - w) / 2, (height - h) / 2), self.text, font=font, fill=(temp_val, temp_val, temp_val, 255))
        return np.array(img_pil)


class ImageGen(VideoGeneratorBase):
    def __init__(self, img_file, frames):
        super().__init__()
        self.frame_count = frames
        self.img_file = img_file

    def frame_gen_func(self, frame_cnt):
        canvas = np.ones((height, width, 3), np.uint8) * 255
        image = cv2.imread(self.img_file)
        img_h, img_w, _ = image.shape
        canvas[int(height / 2) - int(img_h / 2):int(height / 2) - int(img_h / 2) + img_h, int(width / 2) -
               int(img_w / 2):int(width / 2) - int(img_w / 2) + img_w, :] = image[:, :, :] / self.frame_count * \
               frame_cnt + np.ones((img_h, img_w, 3), np.uint8) / self.frame_count * 255 * (self.frame_count -
                                                                                            frame_cnt)
        return canvas


class EndGen(VideoGeneratorBase):
    def __init__(self, text, frames):
        super().__init__()
        self.frame_count = frames
        self.text = text

    def frame_gen_func(self, frame_cnt):
        img = np.ones((height, width, 3), np.uint8) * 255
        (w, h), _ = cv2.getTextSize(self.text, cv2.FONT_HERSHEY_SIMPLEX, 4, 2)
        cv2.putText(img, self.text, (int((width - w) / 2), int((-h) * frame_cnt / self.frame_count + (height + h) *
                                     (self.frame_count - frame_cnt) / self.frame_count)), cv2.FONT_HERSHEY_SIMPLEX,
                    4, (0, 0, 0), 2, cv2.LINE_AA)
        return img


if __name__ == '__main__':
    ImageGen('assets/zju.png', 48).draw()
    ImageGen('assets/me.jpg', 24).draw()
    TextGen('曾充 3180106183', 24, 'assets/simsun.ttc').draw()
    TextGen('儿童画时间到！', 24, 'assets/simsun.ttc').draw()
    CircleGen(12).draw()
    EndGen("The End", 96).draw()
    EndGen("A ZC's Film", 96).draw()
    EndGen("THX", 96).draw()
    cv2.destroyAllWindows()
    video.release()

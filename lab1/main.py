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
                print('Paused')
                while cv2.waitKey(-1) != 32:
                    pass
                print('Continue')


class RandomPixel(VideoGeneratorBase):
    def __init__(self, frames):
        super().__init__()
        self.frame_count = frames

    def frame_gen_func(self, frame_cnt):
        frame = np.random.randint(0, 256, (height, width, 3), dtype=np.uint8)
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
                                                                                          int(img_w / 2):int(
            width / 2) - int(img_w / 2) + img_w, :] = image[:, :, :] / self.frame_count * frame_cnt \
                                                      + np.ones((img_h, img_w, 3),
                                                                np.uint8) / self.frame_count * 255 * (
                                                                  self.frame_count - frame_cnt)
        return canvas


class ASCIIGen(VideoGeneratorBase):
    def __init__(self, text, frames):
        super().__init__()
        self.frame_count = frames
        self.text = text

    def frame_gen_func(self, frame_cnt):
        img = np.ones((height, width, 3), np.uint8) * 255
        (w, h), _ = cv2.getTextSize(self.text, cv2.FONT_HERSHEY_SIMPLEX, 4, 2)
        cv2.putText(img, self.text, (int((width - w) / 2), int((-h) * frame_cnt / self.frame_count + (height + h) *
                                                               (self.frame_count - frame_cnt) / self.frame_count)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    4, (0, 0, 0), 2, cv2.LINE_AA)
        return img


class ChildrenPaint1(VideoGeneratorBase):
    def __init__(self):
        super().__init__()
        self.frame_count = 96 * 2

    def frame_gen_func(self, frame_cnt):
        img = np.ones((height, width, 3), np.uint8) * 255
        cv2.line(img, (0, 480), (int(1280 * frame_cnt / self.frame_count), 480), (0, 0, 0), 2)
        cv2.rectangle(img, (300, 480), (380, 400), (255 * (self.frame_count - frame_cnt) / self.frame_count, 255, 255),
                      3)
        cv2.rectangle(img, (350, 480), (370, 440), (255 * (self.frame_count - frame_cnt) / self.frame_count, 255, 255),
                      3)
        cv2.circle(img, (
        1080, int(93 * (frame_cnt / self.frame_count) - 63 * (self.frame_count - frame_cnt) / self.frame_count)), 63,
                   (0, 0, 255), -1)
        cv2.ellipse(img, (340, 400), (80, 50), 0, 180, 180 * (1 + frame_cnt / self.frame_count), (255, 0, 0), -1)
        return img


class ChildrenPaint2(VideoGeneratorBase):
    def __init__(self):
        super().__init__()
        self.frame_count = 48

    def frame_gen_func(self, frame_cnt):
        img = np.ones((height, width, 3), np.uint8) * 255
        cv2.ellipse(img, (int(width / 2), int(height / 2)), (300, 300), 0, 0, 360 * frame_cnt / self.frame_count,
                    (255, 0, 0), 10)
        cv2.circle(img, (
        510, int(290 * (frame_cnt / self.frame_count) - 63 * (self.frame_count - frame_cnt) / self.frame_count)), 63,
                   (0, 0, 255), -1)
        cv2.circle(img, (
        770, int(290 * (frame_cnt / self.frame_count) - 63 * (self.frame_count - frame_cnt) / self.frame_count)), 63,
                   (0, 0, 255), -1)
        cv2.ellipse(img, (640, 430), (100, 60), 0, 0, 180 * frame_cnt / self.frame_count, (0, 0, 0), -1)
        return img


class ChildrenPaint3(VideoGeneratorBase):
    def __init__(self):
        super().__init__()
        self.frame_count = 48

    def frame_gen_func(self, frame_cnt):
        img = np.ones((height, width, 3), np.uint8) * 255
        cv2.ellipse(img, (int(width / 2), int(height / 2)), (300, 300), 0, 0, 360 * frame_cnt / self.frame_count,
                    (255, 0, 0), 10)
        cv2.circle(img, (
        510, int(290 * (frame_cnt / self.frame_count) - 63 * (self.frame_count - frame_cnt) / self.frame_count)), 63,
                   (0, 0, 255), -1)
        cv2.circle(img, (
        770, int(290 * (frame_cnt / self.frame_count) - 63 * (self.frame_count - frame_cnt) / self.frame_count)), 63,
                   (0, 0, 255), -1)
        cv2.ellipse(img, (640, 430), (100, 60), 0, 0, 180 * frame_cnt / self.frame_count, (0, 0, 0), 5)
        return img


if __name__ == '__main__':
    ImageGen('assets/zju.png', 48).draw()
    ImageGen('assets/me.jpg', 24).draw()
    TextGen('曾充 3180106183', 24, 'assets/simsun.ttc').draw()
    TextGen('儿童画时间到！', 24, 'assets/simsun.ttc').draw()
    TextGen('太阳升起来了', 24, 'assets/simsun.ttc').draw()
    ChildrenPaint1().draw()
    TextGen('让我看看谁有早八？', 24, 'assets/simsun.ttc').draw()
    ChildrenPaint2().draw()
    TextGen('哦，原来是我啊（', 24, 'assets/simsun.ttc').draw()
    ChildrenPaint3().draw()
    TextGen('太丑了，不画了，不画了……', 24, 'assets/simsun.ttc').draw()
    RandomPixel(12).draw()
    ASCIIGen("The End", 96).draw()
    ASCIIGen("A ZC's Film", 96).draw()
    ASCIIGen("THX", 96).draw()
    cv2.destroyAllWindows()
    video.release()

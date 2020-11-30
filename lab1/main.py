# import numpy as np
# import cv2 as cv
# from PIL import ImageFont, ImageDraw, Image
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
# fontpath = "./simsun.ttc"
# font = ImageFont.truetype(fontpath, 32)
# img_pil = Image.fromarray(img)
# draw = ImageDraw.Draw(img_pil)
# draw.text((0, 0),  "国庆节/中秋节 快乐!", font=font, fill = (0, 255, 255, 155))
# img = np.array(img_pil)
#
# cv.imshow('233', img)
# cv.waitKey(-1)
# cv.destroyAllWindows()
import time

import cv2
import numpy as np

width = 1280
height = 720
FPS = 24
seconds = 10
radius = 150
paint_h = int(height / 2)
time_interval = 1.0 / FPS

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
video = cv2.VideoWriter('./circle_noise.mp4', fourcc, float(FPS), (width, height))


def drawing():
    for paint_x in range(-radius, width + radius, 6):
        frame_start_time = time.time()
        frame = np.random.randint(0, 256,
                                  (height, width, 3),
                                  dtype=np.uint8)
        cv2.circle(frame, (paint_x, paint_h), radius, (0, 0, 0), -1)
        cv2.imshow('lab1', frame)
        video.write(frame)
        wait_time = time_interval - (time.time() - frame_start_time)
        if wait_time > 0 and cv2.waitKey(int(wait_time * 1000)) == 32:
            while cv2.waitKey(-1) != 32:
                pass


drawing()
cv2.destroyAllWindows()
video.release()

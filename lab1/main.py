import numpy as np
import cv2 as cv
from PIL import ImageFont, ImageDraw, Image
# Create a black image
img = np.zeros((512,512,3), np.uint8)
# Draw a diagonal blue line with thickness of 5 px
cv.line(img,(0,0),(511,511),(255,0,0),5)
cv.rectangle(img,(384,0),(510,128),(0,255,0),3)
cv.circle(img,(447,63), 63, (0,0,255), -1)
cv.ellipse(img,(256,256),(100,50),0,0,180,255,-1)
pts = np.array([[10,5],[20,30],[70,20],[50,10]], np.int32)
pts = pts.reshape((-1,1,2))
cv.polylines(img,[pts],True,(0,255,255))
font = cv.FONT_HERSHEY_SIMPLEX
cv.putText(img,'opencv',(10,500), font, 4,(255,255,255),2,cv.LINE_AA)
fontpath = "./simsun.ttc"
font = ImageFont.truetype(fontpath, 32)
img_pil = Image.fromarray(img)
draw = ImageDraw.Draw(img_pil)
draw.text((0, 0),  "国庆节/中秋节 快乐!", font=font, fill = (0, 255, 255, 155))
img = np.array(img_pil)

cv.imshow('233', img)
cv.waitKey(-1)
cv.destroyAllWindows()

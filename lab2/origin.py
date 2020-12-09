# import cv2
# import numpy as np
#
# img = cv2.imread('assets/sample-2.jpg')
# gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
# edges = cv2.Canny(gray,50,150,apertureSize = 3)
#
# lines = cv2.HoughLines(edges,1,np.pi/180,50)
# for rho,theta in lines[0]:
#     a = np.cos(theta)
#     b = np.sin(theta)
#     x0 = a*rho
#     y0 = b*rho
#     x1 = int(x0 + 1000*(-b))
#     y1 = int(y0 + 1000*(a))
#     x2 = int(x0 - 1000*(-b))
#     y2 = int(y0 - 1000*(a))
#
#     cv2.line(img,(x1,y1),(x2,y2),(0,0,255),2)
#
# cv2.imshow('res',img)
# cv2.waitKey(-1)

import cv2 as cv
#拉普拉斯算子

src = cv.imread('assets/sample-1.jpg')
cv.namedWindow('input_image', cv.WINDOW_NORMAL) #设置为WINDOW_NORMAL可以任意缩放
# cv.imshow('input_image', src)
# dx, dy = cv.spatialGradient(src)
def _debug():
    pass
    # cv.imshow('res', edge_image)
    # cv.waitKey(-1)

edge_image = cv.cvtColor(src, cv.COLOR_BGR2GRAY)
_debug()
edge_image = cv.bilateralFilter(edge_image, 90, 75, 150)
_debug()
edge_image = cv.Canny(edge_image, 50, 150, apertureSize=3, L2gradient=True)
_debug()
edge_image = cv.dilate(edge_image, cv.getStructuringElement(cv.MORPH_RECT, (5, 5)), iterations=3)
_debug()
edge_image = cv.erode(edge_image, cv.getStructuringElement(cv.MORPH_RECT, (5, 5)), iterations=3)
_debug()
edge_image = cv.Canny(edge_image, 50, 150, apertureSize=3, L2gradient=True)
_debug()
edge_image = cv.dilate(edge_image, cv.getStructuringElement(cv.MORPH_RECT, (3, 3)), iterations=1)
edge_image = cv.erode(edge_image, cv.getStructuringElement(cv.MORPH_RECT, (3, 3)), iterations=1)
_debug()
grad_x = cv.Scharr(edge_image, cv.CV_32F, 1, 0)
grad_y = cv.Scharr(edge_image, cv.CV_32F, 0, 1)
magnitude, angle = cv.cartToPolar(grad_x, grad_y)
cv.imshow("gradient", angle)
cv.waitKey(-1)
cv.destroyAllWindows()

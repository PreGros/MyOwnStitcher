import numpy as np
import cv2
import math
from shapely.geometry import Polygon

# p1 = Polygon([(0,0), (1,1), (1,0)])
# p2 = Polygon([(0,1), (1,0), (1,1)])
# print(p1.intersects(p2))




# p = Polygon([(1,1),(2,2),(4,2),(3,1)])
# q = Polygon([(1.5,2),(3,5),(5,4),(3.5,1)])
# print(p.intersects(q))  # True
# print(p.intersection(q).area)  # 1.0
# x = p.intersection(q)
# print(x)



img = cv2.imread("dataset/photoshopTest/dji_fly_20230410_171522_290_1681154690765_photo.jpg")
# img = cv2.imread("dataset/photoshopTest/dji_fly_20230410_171530_291_1681154689203_photo.jpg")

height, width = img.shape[:2]

heightCut = int(height*0.25*0.5)
widthCut = int(width*0.25*0.5)


centerCropped = img[heightCut:height-heightCut, :]

cv2.imwrite("dataset/photoshopOutput/tes.jpg", centerCropped)
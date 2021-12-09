import os
import cv2

path = "C:\\Users\\Robotics3\\PycharmProjects\\FreightFrenzyCV\\dataset\\raw\\left"
newpath = "C:\\Users\\Robotics3\\PycharmProjects\\FreightFrenzyCV\\dataset\\downscaled\\left"

for impath in os.listdir(path):
    img = cv2.imread(os.path.join(path, impath))
    img = cv2.resize(img, (640, 360), fx=0, fy=0, interpolation=cv2.INTER_LINEAR)
    cv2.imwrite(os.path.join(newpath, impath), img)


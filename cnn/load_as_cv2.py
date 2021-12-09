import cv2
import numpy as np
import matplotlib.pyplot as plt

cv2_net = cv2.dnn.readNetFromTensorflow("C:/Users/Robotics3/PycharmProjects/FreightFrenzyCV/frozen_models/v1.0_frozen_graph.pb")

img = cv2.imread("C:\\Users\\Robotics3\\PycharmProjects\\FreightFrenzyCV\\dataset\\split\\testing\\center\\b1c_frame0005.png")


print(cv2_net.getLayerNames())

img_blob = cv2.dnn.blobFromImage(img, size=(360, 640), swapRB=True, crop=True, ddepth=cv2.CV_8U)

cv2_net.setInput(img_blob)
out = cv2_net.forward()
print("Predictions: " + str(out))
print("Guess: " + str(np.argmax(out)))

cv2.imshow('img', img)
cv2.waitKey(0)

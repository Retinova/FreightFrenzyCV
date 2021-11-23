import cv2
import numpy as np
# from tensorflow.keras.preprocessing import image
import matplotlib.pyplot as plt

cv2_net = cv2.dnn.readNetFromTensorflow("C:/Users/Robotics3/PycharmProjects/pythonProject/frozen_models/v1.1_frozen_graph.pb")

img = cv2.imread("C:\\Users\\Robotics3\\PycharmProjects\\pythonProject\\dataset\\raw_split\\testing\\c\\IMG_20191001_051512536_BURST029.jpg")
crop = img[408:2856]

# cv2.imshow(img)
# cv2.waitKey(0)

# img = image.load_img("C:\\Users\\Robotics3\\PycharmProjects\\pythonProject\\dataset\\raw_split\\testing\\a\\IMG_20191001_051441042_BURST000_COVER.jpg")
# img = np.asarray(img)
# plt.imshow(img)
# plt.show()

# img_blob = cv2.dnn.blobFromImage(img)

print(cv2_net.getLayerNames())
# print(crop.shape)

# inputs = np.empty((1, 2448, 2448, 3), dtype='uint8')
# inputs[0] = crop

img_blob = cv2.dnn.blobFromImage(img, size=(2448, 2448), swapRB=True, crop=True, ddepth=cv2.CV_8U)

cv2_net.setInput(img_blob)
out = cv2_net.forward()
print("Predictions: " + str(out))
print("Guess: " + str(np.argmax(out)))

resized = cv2.resize(crop, None, fx=0.2, fy=0.2)
cv2.imshow('img', resized)
cv2.waitKey(0)

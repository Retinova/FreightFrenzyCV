import os
import random
import shutil

path = "C:\\Users\\Robotics3\\PycharmProjects\\FreightFrenzyCV\\dataset\\split\\training"
path2 = "C:\\Users\\Robotics3\\PycharmProjects\\FreightFrenzyCV\\dataset\\split\\testing"
d = ["b1c_frame", "b2c_frame", "r1c_frame", "r2c_frame", "b1l_frame", "b2l_frame", "r1l_frame", "r2l_frame", "b1r_frame", "b2r_frame", "r1r_frame", "r2r_frame"]

nums = []
count = 0

for fold in ["center", "left", "right"]:
    for x in range(0, 4):
        nums = random.sample(range(1, 600), 50)
        for num in nums:
            src = os.path.join(path, fold, d[count+x] + format(num, "04d") + ".png")
            dst = os.path.join(path2, fold, d[count+x] + format(num, "04d") + ".png")
            shutil.move(src, dst)
    count += 4

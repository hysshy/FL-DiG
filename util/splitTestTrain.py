import os
import shutil
import random

if __name__ == '__main__':
    imgPath = '/home/chase/shy/DGDA/data/NEU-CLS/orimg'
    trainPath = '/home/chase/shy/DGDA/data/NEU-CLS/train2'
    testPath = '/home/chase/shy/DGDA/data/NEU-CLS/test2'
    thr = 0.5
    for cls in os.listdir(imgPath):
        if not os.path.exists(trainPath+'/'+cls):
            os.makedirs(trainPath+'/'+cls)
        if not os.path.exists(testPath+'/'+cls):
            os.makedirs(testPath+'/'+cls)
        for imgName in os.listdir(imgPath+'/'+cls):
            if random.random() < thr:
                shutil.copy(imgPath+'/'+cls+'/'+imgName, testPath+'/'+cls)
            else:
                shutil.copy(imgPath+'/'+cls+'/'+imgName, trainPath+'/'+cls)



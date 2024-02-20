import os
import shutil
import random

if __name__ == '__main__':
    imgPath = '/home/chase/shy/DenoisingDiffusionProbabilityModel-ddpm-/Miner'
    trainPath = '/home/chase/shy/DDPM4MINER/Miner_train'
    testPath = '/home/chase/shy/DDPM4MINER/Miner_test'

    for cls in os.listdir(imgPath):
        if not os.path.exists(trainPath+'/'+cls):
            os.makedirs(trainPath+'/'+cls)
        if not os.path.exists(testPath+'/'+cls):
            os.makedirs(testPath+'/'+cls)
        for imgName in os.listdir(imgPath+'/'+cls):
            if random.random() < 0.3:
                shutil.copy(imgPath+'/'+cls+'/'+imgName, testPath+'/'+cls)
            else:
                shutil.copy(imgPath+'/'+cls+'/'+imgName, trainPath+'/'+cls)



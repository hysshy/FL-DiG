import os
import shutil
import random

if __name__ == '__main__':
    imgPath = '/home/chase/shy/DDPM4MINER/data/classes'
    trainPath = '/home/chase/shy/DDPM4MINER/data/Miner_train'
    testPath = '/home/chase/shy/DDPM4MINER/data/Miner_test'

    for cls in os.listdir(imgPath):
        for shape in os.listdir(imgPath+'/'+cls):
            if not os.path.exists(trainPath+'/'+cls +'/'+shape):
                os.makedirs(trainPath+'/'+cls +'/'+shape)
            if not os.path.exists(testPath+'/'+cls+'/'+shape):
                os.makedirs(testPath+'/'+cls+'/'+shape)
            for imgName in os.listdir(imgPath+'/'+cls+'/'+shape):
                if random.random() < 0.3:
                    shutil.copy(imgPath+'/'+cls+'/'+shape+'/'+imgName, testPath+'/'+cls+'/'+shape)
                else:
                    shutil.copy(imgPath+'/'+cls+'/'+shape+'/'+imgName, trainPath+'/'+cls+'/'+shape)



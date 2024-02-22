import os

import cv2

# if __name__ == '__main__':
#     imgPath = '/home/chase/shy/DenoisingDiffusionProbabilityModel-ddpm-/Miner/cluster'
#     cpnum = 10
#     for subPath in os.listdir(imgPath):
#         for imgName in os.listdir(imgPath+'/'+subPath):
#             img = cv2.imread(imgPath+'/'+subPath+'/'+imgName)
#             for i in range(cpnum):
#                 cv2.imwrite(imgPath+'/'+subPath+'/'+imgName.split('.')[0]+'_'+str(i+1)+'.jpg', img)

if __name__ == '__main__':
    imgPath = '/home/chase/shy/DDPM4MINER/data/ddpm_miner'
    MaxNum = 60
    for subPath in os.listdir(imgPath):
        for subsubPath in os.listdir(imgPath+'/'+subPath):
            imgNum = len(os.listdir(imgPath+'/'+subPath+'/'+subsubPath))
            cpnum = int((MaxNum-imgNum)/imgNum)
            for imgName in os.listdir(imgPath+'/'+subPath+'/'+subsubPath):
                img = cv2.imread(imgPath+'/'+subPath+'/'+subsubPath+'/'+imgName)
                for i in range(cpnum):
                    cv2.imwrite(imgPath+'/'+subPath+'/'+subsubPath+'/'+imgName.split('.')[0]+'_'+str(i+1)+'.jpg', img)
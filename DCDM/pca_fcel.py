
# 导入skfuzzy库和其他必要的库
import skfuzzy as fuzz
import numpy as np
import os
from PIL import Image
from sklearn.decomposition import PCA

# 定义文件夹路径，聚类数目，模糊指数
folder = "data/MT/train"
savePath = 'data/MT/F-train'# 存放图片的文件夹

def embeding(folder, savePath):
    c = 5 # 聚类数目，根据图片的特征和数量选择
    m = 1.1 # 模糊指数，一般取2

    for cls in os.listdir(folder):
        # 读取文件夹下的所有图片文件，并转换为灰度值矩阵
        images = []  # 存放图片对象的列表
        data = []  # 存放图片灰度值的列表
        imageNames = []
        for imgName in os.listdir(folder+'/'+cls): # 假设你有10张图片，你可以根据你的实际情况修改
            img = Image.open(folder+'/'+cls+'/'+imgName)
            img = img.resize((64,64), Image.ANTIALIAS)
            img = img.convert("L") # 转换为灰度图
            images.append(img) # 将图片从BGR格式转换为RGB格式
            imageNames.append(imgName)
            data.append(np.array(img).flatten())
        data = np.array(data)
        pca = PCA(n_components=20)
        data = pca.fit_transform(data)

        # 将图片灰度值列表转换为numpy数组
        data = np.array(data).T # 转置为skfuzzy要求的格式

        # 使用skfuzzy库的cmeans函数进行模糊聚类
        cntr, u, u0, d, jm, p, fpc = fuzz.cluster.cmeans(
            data, c, m, error=0.0005, maxiter=10000, init=None)
        print(fpc)
        # 根据聚类结果，将图片分组并保存到不同的文件夹
        cluster_membership = np.argmax(u, axis=0) # 获取每张图片的聚类标签
        for i in range(c):
            os.makedirs(savePath+'/'+cls+'/'+f"cluster_{i}", exist_ok=True) # 创建聚类文件夹
            for j in range(len(images)):
                if cluster_membership[j] == i: # 如果图片属于该聚类
                    images[j].save(savePath+'/'+cls+'/'+f"cluster_{i}/"+imageNames[j]) # 保存图片到对应的文件夹
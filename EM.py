import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm
import numpy as np
from scipy import stats
# 定义高斯分布的参数
mean1, std1 = 164, 3
mean2, std2 = 176, 5

# 从两个高斯分布中生成各50个样本
data1 = np.random.normal(mean1, std1, 500)
data2 = np.random.normal(mean2, std2, 1500)
data = np.concatenate((data1, data2), axis=0)

# 将数据写入 CSV 文件
df = pd.DataFrame(data, columns=['height'])
df.to_csv('height_data.csv', index=False)

# 绘制数据的直方图
plt.hist(data, bins=20)
plt.xlabel('Height (cm)')
plt.ylabel('Count')
plt.title('Distribution of Heights')
plt.show()

# EM算法的实现

def em(h, mu1, sigma1, w1, mu2, sigma2, w2):
    d = 1
    n = len(h)  # 样本长度
    p1 = w1 * stats.norm(mu1, sigma1).pdf(h)
    p2 = w2 * stats.norm(mu2, sigma2).pdf(h)
    # p1, p2权重 * 男女生的后验概率
    R1i = p1 / (p1 + p2)
    R2i = p2 / (p1 + p2)
    # M-step
    # mu的更新
    mu1 = np.sum(R1i * h) / np.sum(R1i)
    mu2 = np.sum(R2i * h) / np.sum(R2i)
    # sigma1的更新
    sigma1 = np.sqrt(np.sum(R1i * np.square(h - mu1)) / (d * np.sum(R1i)))
    sigma2 = np.sqrt(np.sum(R2i * np.square(h - mu2)) / (d * np.sum(R2i)))
    # w的更新
    w1 = np.sum(R1i) / n
    w2 = np.sum(R2i) / n

    return mu1, sigma1, w1, mu2, sigma2, w2

#数据集
h=list(data1)# 转化为list
h.extend(data2)
h=np.array(h)# 再转成numpy格式的数据
#初始化
mu1 = 180; sigma1 = 5; w1 = 0.9
mu2 = 170; sigma2 = 4; w2 = 0.1
for iteration in range(1000):
    mu1,sigma1,w1,mu2,sigma2,w2=em(h,mu1,sigma1,w1,mu2,sigma2,w2)
print("mu1: ",mu1)
print("sigma1: ",sigma1)
print("w1: ",w1)
print("mu2: ",mu2)
print("sigma2: ",sigma2)
print("w2: ",w2)

#预测
while(1):
 height =int(input("输入身高："))
 prob1 = [w1 * norm.pdf(height, mu1, sigma1)]
 prob2 = [w2 * norm.pdf(height, mu2, sigma2)]
 if prob1 < prob2:
    print("可能是女性")
 else:
    print("可能是男性")

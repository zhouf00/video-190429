# 分类

import numpy as np  # 处理矩阵
import pandas as pd  # numpy 加强版，并且可以帮我们读写数据文件
import matplotlib.pylab as plt  # 帮我们数据可视化
from sklearn.preprocessing import StandardScaler

dataset = pd.read_csv("Social_Network_Ads.csv")


x = dataset.iloc[:, [2, 3]].values  # 取出年龄、性别
y = dataset.iloc[:, -1].values  # 取最后一行
#x = dataset[["Age", "EstimatedSalary"]]
#y = dataset["Purchased"]


# 数据预处理
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size= 0.25)

# 特征处理
std = StandardScaler()
x_train = std.fit_transform(x_train)
x_test = std.fit_transform(x_test)

# 建立模型
from sklearn.linear_model import LogisticRegression
classifler = LogisticRegression()
classifler.fit(x_train, y_train)

# 进行预测
y_pred = classifler.predict(x_test)
print(y_pred)
y_score = classifler.score(x_test, y_test)
print(y_score)

from matplotlib.colors import ListedColormap

x_set, y_set = x_train, y_train  # 把两个训练数据拿来画图
x1, x2 = np.meshgrid(np.arange(start=x_set[:, 0].min()-1, stop=x_set[:, 0].max()+1, step=0.01),
                     np.arange(start=x_set[:, 1].min() - 1, stop=x_set[:, 1].max() + 1, step=0.01))
plt.contourf(x1, x2, classifler.predict(np.array([x1.ravel(), x2.ravel()]).T).reshape(x1.shape),
             alpha=0.75, cmap=ListedColormap(("red", "green")))
plt.xlim(x1.min(), x1.max())
plt.xlim(x2.min(), x2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(x_set[y_set == j, 0], x_set[y_set == j, 1],
                c=ListedColormap(("yellow", "blue"))(i), label=j)
plt.show()
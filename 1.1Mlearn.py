# 回归

import numpy as np  # 处理矩阵
import pandas as pd  # numpy 加强版，并且可以帮我们读写数据文件
import matplotlib.pylab as plt  # 帮我们数据可视化

dataset = pd.read_csv("Salary_Data.csv")

y = dataset.iloc[:, 1].values
x = dataset.iloc[:, :-1].values

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=1/3)

from sklearn.linear_model import LinearRegression  # 调用线性回归模型

regressor = LinearRegression()  # 把包拿来用，用来做模型
regressor.fit(x_train, y_train)

y_pred = regressor.predict(x_test)
print(y_pred)
#y_score = regressor.score(x_test, y_test)
#print(y_score)

plt.scatter(x_train, y_train, color="red")
plt.plot(x_test, y_pred, color = "blue")
plt.show()

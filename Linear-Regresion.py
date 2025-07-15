【线性回归（Linear Regression）】
简单来说就是通过线性模型学习数据特征与目标值之间的关系，可以计算出各个特征的权重、整体的偏置，或者进行预测
就比如数学中想要得到一条直线的表达式，根据直线上的两个以上确定点的坐标，就可以计算出来

1.用一个简单的例子来展示一下基础线性回归模型：

假设：房价=2.5x面积-1.8x距市中心距离+50

from sklearn.linear_model import LinearRegression
import numpy as np

x = np.array([
    [5,5],
    [10,30],
    [20,15],
    [50,80],
    [100,25]
]) # 输入二维特征数据，分别代表面积和距离
y = np.array([53.5,21,73,31,255])

model = LinearRegression()
model.fit(x,y)

print(model.predict([[40,55]]))
# 这里注意 model.predict函数里也要两层[] 表示是多维特征
print('权重 w 为',model.coef_,'\n','偏置 b 为',model.intercept_)

-->非常简单的表达式，样本数量虽然小但也足够，因此运行可以准确得到我们设定的两个权重和偏置：
[51.]
权重 w 为 [ 2.5 -1.8] 
 偏置 b 为 50.0


----------


2.还可以随机生成数据
让数据量变大，同时引入不同的噪声

np.random.seed(42) # 固定随机数生成器的种子，让每次运行生成的随机数相同
n_samples = 30
# 随机生成30个样本，模拟小数据过拟合场景
x1 = np.random.uniform(0, 200, size=n_samples)
x2 = np.random.uniform(0, 100, size=n_samples)
x = np.column_stack([x1, x2])
y_true = 2.5 * x1 - 1.8 * x2 + 50
y = y_true + np.random.normal(0, 40, size=n_samples)  # 加入随机噪声

# 添加无关特征，将原始矩阵与8个随机生成的噪声特征水平合并成一个高维特征矩阵，元素服从标准正态分布
x_highdim = np.column_stack([x, np.random.randn(n_samples, 8)])


----------


3.实际运用中经常会使用到【正则化】来防止过拟合

过拟合可能会由以下原因引起：
a.数据量太少（样本不足）
b.模型复杂度过高:模型参数过多（如高阶多项式、深层神经网络），灵活度过高
c.特征过多（高维数据）:特征数量多且部分特征无关或冗余，模型可能会学习到噪声
d.训练数据噪声大
e.训练时间过长（迭代次数过多）
这些情况下，模型容易机械地学习样本细节和噪声，相当于考前复习只背答案，一到测试集上就不行了

常见正则化方式有：
a.L1正则化（lasso）:对所有权重进行平方惩罚，倾向于让权重均匀减小，但不会完全为0，适合特征间存在共线性（相关性高）的情况
b.L2正则化（岭回归ridge）:对权重进行绝对值惩罚，倾向于让部分权重为0（稀疏解），适合特征选择，会自动剔除不重要的特征
c.弹性网 Elastic Net :结合L1和L2

超参数 λ：控制正则化强度

--->
以下模型实现从真实数据集中加载数据，适当添加噪声，最后对比无正则化/ridge/lasso的权重和偏置计算结果与MSE

import numpy as np
import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.metrics import mean_squared_error

data = fetch_california_housing()
X = data.data  # 20640 samples, 8 features
y = data.target  # 目标值 （房价）

# 添加高维特征（多项式特征 + 随机噪声特征）
poly = PolynomialFeatures(degree=2, include_bias=False)  # 生成二次多项式特征
X_poly = poly.fit_transform(X)  # 原始8维 -> 扩展到36维（8 + C(8,2) + 8^2）

# 添加20个随机噪声特征（模拟高维冗余特征）
np.random.seed(42)
noise_features = np.random.randn(X_poly.shape[0], 10)
X_highdim = np.hstack([X,X_poly, noise_features])  # 最终特征维度：8 + 36 + 10 = 54维

# 数据标准化（正则化对尺度敏感，必须先标准化！）
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_highdim)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

lr = LinearRegression()
lr.fit(X_train, y_train)

ridge = Ridge(alpha=100.0)  # 正则化强度调大
ridge.fit(X_train, y_train)

lasso = Lasso(alpha=0.001,max_iter=5000)  # 将最大迭代次数从默认的1000增加到5000，确保模型能收敛到最优解
lasso.fit(X_train, y_train)


def evaluate(model, X, y):
    y_pred = model.predict(X)
    mse = mean_squared_error(y, y_pred)
    print(f"{model.__class__.__name__:8s} MSE: {mse:.4f}")
    print(f"        前5个特征权重: {model.coef_[:5]}")
    print(f"        噪声特征权重均值: {np.mean(np.abs(model.coef_[36:])):.4f}")  # 统计噪声特征权重

print("=== 训练集性能 ===")
evaluate(lr, X_train, y_train)
evaluate(ridge, X_train, y_train)
evaluate(lasso, X_train, y_train)

print("\n=== 测试集性能 ===")
evaluate(lr, X_test, y_test)
evaluate(ridge, X_test, y_test)
evaluate(lasso, X_test, y_test)


-->运行结果：

=== 训练集性能 ===
LinearRegression MSE: 0.4206
        前5个特征权重: [-11.33841937  -5.29496991   9.77386445  -9.09419109  -0.11731072]
        噪声特征权重均值: 6.2743
Ridge    MSE: 0.4767
        前5个特征权重: [ 0.2966968  -0.05748412 -0.0819126   0.05746237 -0.15790859]
        噪声特征权重均值: 0.1094
Lasso    MSE: 0.4677
        前5个特征权重: [ 0.82663729 -0.         -0.08975537  0.         -0.54308037]
        噪声特征权重均值: 0.1782

=== 测试集性能 ===
LinearRegression MSE: 0.4646
        前5个特征权重: [-11.33841937  -5.29496991   9.77386445  -9.09419109  -0.11731072]
        噪声特征权重均值: 6.2743
Ridge    MSE: 0.4937
        前5个特征权重: [ 0.2966968  -0.05748412 -0.0819126   0.05746237 -0.15790859]
        噪声特征权重均值: 0.1094
Lasso    MSE: 0.4804
        前5个特征权重: [ 0.82663729 -0.         -0.08975537  0.         -0.54308037]
        噪声特征权重均值: 0.1782


可以看出：
a.无正则化模型噪声特征权重非常大，过度依赖某些特征，且从训练集到测试集明显表现变差（MSE变化大）
b.lasso和ridge噪声特征权重在0.2以内，有效抑制了噪声干扰
c.lasso将部分权重压为0，实现了自动特征选择





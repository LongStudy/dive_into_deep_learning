import random
import torch
from d2l import torch as d2l

'''

with torch.no_grad() 则主要是用于停止autograd模块的工作，以起到加速和节省显存的作用，具体行为就是停止gradient计算，从而节省了GPU算力和显存，但是并不会影响dropout和batchnorm层的行为。

mm只能进行矩阵乘法,也就是输入的两个tensor维度只能是( n × m )和( m × p )
bmm是两个三维张量相乘, 两个输入tensor维度是( b × n × m )和( b × m × p ), 第一维b代表batch size，输出为( b × n × p )
matmul可以进行张量乘法, 输入可以是高维.

python知识补充：
Python3 range() 函数返回的是一个可迭代对象（类型是对象），而不是列表类型， 所以打印的时候不会打印列表。
Python3 list() 函数是对象迭代器，可以把range()返回的可迭代对象转为一个列表，返回的变量类型为列表。
Python3 range(start, stop[, step])
Python3 shuffle() 方法将序列的所有元素随机排序。shuffle()是不能直接访问的，需要导入 random 模块。举例：random.shuffle (list)
Python3 yield是python中的生成器

'''

print('生成数据集')


# 人造数据集
def synthetic_data(w, b, nums_example):
    """生成 y = Xw + b + 噪声。"""
    X = torch.normal(0, 1, (nums_example, len(w)))
    y = torch.matmul(X, w) + b
    print("y_shape:", y.shape)
    y += torch.normal(0, 0.01, y.shape)  # 加入噪声
    return X, y.reshape(-1, 1)  # y从行向量转为列向量


true_w = torch.tensor([2, -3.4])
true_b = 4.2
# 生成1000个人造数据
features, labels = d2l.synthetic_data(true_w, true_b, 1000)
# 读第一个样本和标签
print('features:', features[0], '\nlabel:', labels[0])
# 生成散点图
d2l.set_figsize()
d2l.plt.scatter(features[:, 1].detach().numpy(), labels.detach().numpy(), 1)
# .detach()返回一个新的tensor，从当前计算图中分离下来的，但是仍指向原变量的存放位置,不同之处只是requires_grad为false，得到的这个tensor永远不需要计算其梯度，不具有grad
d2l.plt.show()

print('读取数据集')


# 读数据集
def data_iter(batch_size, features, lables):  # batch_size指定小批量读取
    nums_example = len(features)
    indices = list(range(nums_example))  # 生成0-999的元组，然后将range()返回的可迭代对象转为一个列表
    random.shuffle(indices)  # 将序列的所有元素随机排序。
    for i in range(0, nums_example, batch_size):
        # range(start, stop, step)， 从0-999，每次读batch_size个，由于yield记住返回的位置下次继续
        index_tensor = torch.tensor(indices[i: min(i + batch_size, nums_example)])
        # min(i + batch_size, nums_example)防止越界，例如数据有998个，最后一组只有8个时min返回8
        yield features[index_tensor], lables[index_tensor]  # 通过索引访问向量


batch_size = 10
for X, y in data_iter(batch_size, features, labels):
    print("X:", X, "\ny", y)
    break

print('初始化模型参数')

# 初始化参数
w = torch.normal(0, 0.01, size=(2, 1), requires_grad=True)  # 均值为0，方差为0.01，大小为2，1，需要计算梯度
b = torch.zeros(1, requires_grad=True)

print('定义模型')


# 定义模型
def linreg(X, w, b):
    return torch.matmul(X, w) + b


print('定义损失函数')


# 定义损失函数
def squared_loss(y_hat, y):
    # print("y_hat_shape:",y_hat.shape,"\ny_shape:",y.shape)
    return (y_hat - y.reshape(
        y_hat.shape)) ** 2 / 2  # 这里为什么要加 y_hat_shape: torch.Size([10, 1])  y_shape: torch.Size([10])，即形状转换


print('定义优化算法')


# 定义优化算法
def sgd(params, lr, batch_size):
    """小批量随机梯度下降。"""
    with torch.no_grad():  # with torch.no_grad() 则主要是用于停止autograd模块的工作，
        for param in params:
            param -= lr * param.grad / batch_size  #  这里用param = param - lr * param.grad / batch_size会导致导数丢失， zero_()函数报错
            param.grad.zero_()  # 导数如果丢失了，会报错‘NoneType’ object has no attribute ‘zero_’


print('训练模型')

# 训练模型
lr = 0.03
num_epochs = 3
net = d2l.linreg
loss = d2l.squared_loss

for epoch in range(0, num_epochs):
    for X, y in data_iter(batch_size, features, labels):
        l = loss(net(X, w, b), y) # `X`和`y`的小批量损失
        # 因为`f`形状是(`batch_size`, 1)，而不是一个标量。`f`中的所有元素被加到一起，
        # 并以此计算关于[`w`, `b`]的梯度
        l.sum().backward()
        d2l.sgd([w, b], lr, batch_size)  # 使用参数的梯度更新参数
    with torch.no_grad():
        train_l = loss(net(features, w, b), labels)
        # print("w {0} \nb {1} \nloss {2:f}".format(w, b, float(train_l.mean())))
        print(f'epoch {epoch + 1}, loss {float(train_l.mean()):f}')

# print("w误差 ", true_w - w, "\nb误差 ", true_b - b)
print(f'w的估计误差: {true_w - w.reshape(true_w.shape)}')
print(f'b的估计误差: {true_b - b}')
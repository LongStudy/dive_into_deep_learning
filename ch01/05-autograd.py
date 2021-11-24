import torch

print('1.自动梯度计算')
x = torch.arange(4.0, requires_grad=True)  # 1.将梯度附加到想要对其计算偏导数的变量
print('x:', x)
print('x.grad:', x.grad)
y = 2 * torch.dot(x, x)  # 2.记录目标值的计算
print('y:', y)
y.backward()  # 3.执行它的反向传播函数
print('x:', x)
print('x.grad:', x.grad)  # 4.访问得到的梯度
print('x.grad == 4*x:', x.grad == 4 * x)
print('')

## 计算另一个函数
x.grad.zero_() # 清零
y = x.sum()
print('y:', y)
y.backward()
print('x.grad:', x.grad)
print('')

# 非标量变量的反向传播
x.grad.zero_()
print('x:', x)
y = x * x
y.sum().backward() # 梯度只能为标量（即一个数）输出隐式地创建，此时y为矩阵，需要sum为标量
print('x.grad:', x.grad)
print('')

# 分离计算
x.grad.zero_()
y = x * x
u = y.detach()
z = u * x # u当成常量，所以x.grad = u
z.sum().backward()
print('x.grad == u', x.grad == u)
x.grad.zero_()
y.sum().backward()
print('x.grad == 2 * x', x.grad == 2 * x)
print('')
def f(a):
    b = a * 2
    print('b.norm:', b.norm())
    while b.norm() < 1000:  # 求L2范数：元素平方和的平方根
        b = b * 2
    if b.sum() > 0:
        c = b
    else:
        c = 100 * b
    return c


print('2.Python控制流的梯度计算')
a = torch.tensor(2.0)  # 初始化变量
a.requires_grad_(True)  # 1.将梯度赋给想要对其求偏导数的变量
print('a:', a)
d = f(a)  # 2.记录目标函数
print('d:', d)
d.backward()  # 3.执行目标函数的反向传播函数
print('a.grad:', a.grad)  # 4.获取梯度

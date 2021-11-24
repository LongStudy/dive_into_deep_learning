import torch
from torch.distributions import multinomial
from d2l import torch as d2l

fair_probs = torch.ones([6]) / 6
A = multinomial.Multinomial(1, fair_probs).sample()
print('模拟投掷1次骰子，各点数出现次数：', A)
B = multinomial.Multinomial(10, fair_probs).sample()
print('模拟投掷10次骰子，各点数出现次数：', B)
counts = multinomial.Multinomial(1000, fair_probs).sample()
print('模拟投掷1000次骰子，各点数出现概率：', counts / 1000)

counts = multinomial.Multinomial(10, fair_probs).sample((500,))
cum_counts = counts.cumsum(dim=0)
estimates = cum_counts / cum_counts.sum(dim=1, keepdims=True)

d2l.set_figsize((6, 4.5))
for i in range(6):
    d2l.plt.plot(estimates[:, i].numpy(),
                 label=("P(die=" + str(i + 1) + ")"))
d2l.plt.axhline(y=0.167, color='black', linestyle='dashed')
d2l.plt.gca().set_xlabel('实验次数')
d2l.plt.gca().set_ylabel('估算概率')
d2l.plt.legend()
d2l.plt.rcParams['font.sans-serif']=['SimHei'] # 解决中文无法显示
d2l.plt.show()
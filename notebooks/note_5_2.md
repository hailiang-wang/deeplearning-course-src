
对 w 之于 loss 进行求导

公式1：$\frac {\partial Loss}{\partial w} =\frac {\partial Loss}{\partial model} \frac {\partial model}{\partial w}  $

对 b 之于 loss 进行求导

公式2： $\frac {\partial Loss}{\partial b} =\frac {\partial Loss}{\partial model} \frac {\partial model}{\partial b}  $


梯度：由公式1和公式2计算结果，输出变成向量。


均方误差的表达式：

$$MSE = \ \sum_{m} \frac {(model(x) - y)^{2}}{2m} $$

对均方误差的求导：

$$ \frac {\partial Loss}{\partial model} = model(x) - y$$

model 对 w 求导

$$\frac {\partial model}{\partial w} = \frac {\partial (wx+b)}{\partial w}  = x$$

model 对 b 求导

$$\frac {\partial model}{\partial b} = \frac {\partial (wx+b)}{\partial b}  = 1.0$$




'''
了解 Tensor 张量的数据结构
'''
import torch

# 打印的样式
torch.set_printoptions(edgeitems=10, threshold=50)

x = torch.randn(3, 2, 2)
print(x)
print(x.is_contiguous())
# print(list(x.storage()))
# print(x.stride())

# y = x.view(1, 9)
y = x.view(6, 2)
z = y.transpose(dim0=1, dim1=0)
print(z.is_contiguous())
# w = z.view(3, 4) # 這麼些會出錯
w = z.reshape(3, 4)  # 這麼做不會出錯，但是會帶來開銷 <1>

# 思考
# 在 <1> 中，w 和 z 是同一個 storage 嗎？ id(w.storage()) 和 id(z.storage()) 是相同的嗎？答案：是相同的，都是临时的地址
# print(id(w.storage().data_ptr())) <3>
# print(id(z.storage().data_ptr())) <4>
# <3> 和 <4> 是不同的

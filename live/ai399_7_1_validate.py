from ai399_7_1 import *

from torchinfo import summary

model.load_state_dict(torch.load("ai399_7_1.pth"))

cifar2_x = [x for x, _ in cifar2_val]
cifar2_x = torch.stack(cifar2_x)
cifar2_y = torch.tensor([y for _, y in cifar2_val])

with torch.no_grad():
    outputs = model(cifar2_x.view(cifar2_x.shape[0], -1))
    # print(outputs.shape)
    # torch.Size([2000, 2])

    filter_result = torch.argmax(outputs, dim=-1)

    print("filter_result", filter_result)
    print("cifar2_y", cifar2_y)

    correct = (filter_result == cifar2_y).sum()
    print("准确率： %.4f" % (correct / len(cifar2_y)))

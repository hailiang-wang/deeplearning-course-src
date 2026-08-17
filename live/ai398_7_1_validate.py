import sys
from ai398_7_1 import *
from torchinfo import summary

model.load_state_dict(torch.load("sample_model.pth"))
summary(model)

val_dataloader = torch.utils.data.DataLoader(cifar2_val, batch_size=len(cifar2_val), shuffle=True)

correct = 0
total = len(cifar2_val)

model.eval()
for (imgs, labels) in val_dataloader:
    outputs = model(imgs.view(imgs.shape[0], -1))
    filter_outputs = torch.argmax(outputs, dim=-1)
    print("filter_outputs", filter_outputs, "filter_outputs shape", filter_outputs.shape)
    correct += (labels == filter_outputs).sum().item()
print("准确率： %.3f" % (correct / total))
print("识别准确： %s" % correct)

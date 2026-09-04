'''
对 7_1 的模型进行评估，验证
'''
import sys
from ai501_7_1 import *

torch.set_printoptions(edgeitems=2, threshold=30)

model.load_state_dict(torch.load("ai501_7_1.pth"))
model.to(device=default_device)
summary(model)

correct = 0
total = 0

val_loader = torch.utils.data.DataLoader(cifar2_val, batch_size=batch_size,
                                         shuffle=False,
                                         generator=torch.Generator(device=default_device))

model.eval()

for imgs, labels in val_loader:
    imgs = imgs.view(imgs.shape[0], -1).to(default_device)
    predicts = model(imgs)

    outputs = torch.argmax(predicts, dim=-1)

    print(outputs.shape)
    print(labels.shape)

    correct += (labels == outputs).sum()
    total += labels.shape[0]

print("Accuracy %.4f" % (correct / total))

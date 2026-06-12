import sys
from practice_7_1 import *
from torchinfo import summary

torch.set_printoptions(edgeitems=2, threshold=50)

model.load_state_dict(torch.load("sample_model.pth"))
model.to(default_device)

summary(model)

correct = 0
total = 0

val_loader = torch.utils.data.DataLoader(cifar2_val, batch_size=20, shuffle=True,
                                         generator=torch.Generator(device=default_device))


with torch.no_grad():
    for (imgs, labels) in val_loader:
        imgs = imgs.to(default_device).view(imgs.shape[0], -1)
        outputs = model(imgs)

        # print("*"*80)
        # print("outputs")
        # print(outputs)

        # print("outputs.shape")
        # print(outputs.shape)

        # print("labels")
        # print(labels)

        filter_result = torch.argmax(outputs, dim=-1)

        # print("filter_result")
        # print(filter_result)

        correct += (labels == filter_result).sum()
        total += imgs.shape[0]

print("Accuracy %f" % (correct / total))

import torch
from matplotlib import pyplot as plt
from torchvision import datasets
from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter

torch.manual_seed(100)

'''
加载数据集
'''
data_path = "data-unversioned/p1ch7/"
cifar10 = datasets.CIFAR10(data_path, train=True, download=False)
cifar10_val = datasets.CIFAR10(data_path, train=False, download=False)
class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']

cifar10 = datasets.CIFAR10(data_path, train=True, download=False, transform=transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4915, 0.4823, 0.4468),  # imagenet, 超過 100　万张图片上，做的统计后得到的 RGB 三个通道的均值和标准差
                         (0.2470, 0.2435, 0.2616))
]))

cifar10_val = datasets.CIFAR10(data_path, train=False, download=False, transform=transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4915, 0.4823, 0.4468),  # imagenet, 超過 100　万张图片上，做的统计后得到的 RGB 三个通道的均值和标准差
                         (0.2470, 0.2435, 0.2616))
]))

################################
# 训练神经网络
################################
print("Load dataset ...")
label_map = {0: 0, 2: 1}
class_names = ['airplane', 'bird']  # airplane 飞机，的索引是 0, bird 的索引是 1
# 训练数据
cifar2 = [(img, label_map[label]) for img, label in cifar10 if label in [0, 2]]
# 验证数据
cifar2_val = [(img, label_map[label])
              for img, label in cifar10_val if label in [0, 2]]

print("  Len of cifar2", len(cifar2))
print("  Len of cifar2_val", len(cifar2_val))

validate_inputs = torch.stack(
    tuple([img for (img, _) in cifar2_val]), dim=0)
validate_labels = torch.tensor([label for (_, label) in cifar2_val])

'''
搭建卷积神经网络
'''
import torch.nn as nn
import torch.optim as optimizer

batch_size = 20
n_epoches = 10
lr = 1e-3
n_out = 2

# 图片大小：3x32x32

model = nn.Sequential(
    nn.Conv2d(3,16, kernel_size=3, padding=1, stride=1),
    # 经过了一轮卷积后，含有 16 个输出，那么每个输出的矩阵是多大呢？
    # 宽度计算 32 + 2 - 3 + 1 = 32
    # 那么，卷积输出的shape 就是 16x32x32
    nn.Tanh(),
    nn.MaxPool2d(2),
    # 16x32x32 --> 16x16x16
    nn.Conv2d(16, 8, kernel_size=3, padding=1,stride=1),
    # 16x16x16 --> 8x16x16
    nn.Tanh(),
    nn.MaxPool2d(2),
    #  8x16x16 --> 8x8x8
    nn.Flatten(),
    nn.Linear(8*8*8, 32),
    nn.Tanh(),
    nn.Linear(32, n_out),
    nn.LogSoftmax(dim=-1)
)

opt = optimizer.Adam(params=model.parameters(), lr=lr)
loss_fn = nn.NLLLoss()

if __name__ == "__main__":
    train_loader = torch.utils.data.DataLoader(
        cifar2, batch_size=batch_size, shuffle=True)
    writer = SummaryWriter()

    total_step = 0

    print("Start to train neural network ...")
    for epoch in range(n_epoches):
        # 对训练的 Loss 进行记录
        train_loss = 0
        train_step = 0
        train_outputs = None
        train_labels = None

        for imgs, labels in train_loader:
            # 20x3x32x32 -> 20x3072
            # outputs = model(imgs.view(imgs.shape[0], -1))
            outputs = model(imgs)
            loss = loss_fn(outputs, labels)

            opt.zero_grad()
            loss.backward()

            with torch.no_grad():
                opt.step()
                train_loss += loss.item()
                train_step += 1
                total_step += 1

                if train_outputs is None:
                    train_outputs = outputs
                    train_labels = labels
                else:
                    train_outputs = torch.cat((train_outputs, outputs), dim=0)
                    train_labels = torch.cat((train_labels, labels), dim=0)
        
        '''
        每 1 个 Epoch 完成训练后，进行评测
        '''
        # 记录训练的损失函数值
        train_loss = train_loss / train_step

        # 记录训练的准确率
        predict_labels = torch.argmax(train_outputs, dim=-1)
        predict_correct = (predict_labels == train_labels).sum()
        train_accuracy = predict_correct / predict_labels.numel()

        # 进行验证集数据的预测
        with torch.no_grad():
            # 计算验证集上的损失
            validate_output = model(validate_inputs)

            validate_loss = loss_fn(validate_output, validate_labels)

            # 计算验证集上的准确率
            predict_labels = torch.argmax(validate_output, dim=-1)
            predict_correct = (predict_labels == validate_labels).sum()
            validate_accuracy = predict_correct / len(cifar2_val)

            writer.add_scalars("Loss", {
                "Train": train_loss,
                "validate": validate_loss.item()
            }, epoch)

            writer.add_scalars("Accuracy", {
                "Train": train_accuracy.item(),
                "validate": validate_accuracy.item()
            }, epoch)

            print("  Epoch %s, train (loss %.4f, accuracy %.4f), validate(loss %.4f, accuracy %.4f)" %
                  (epoch, train_loss, train_accuracy.item(), validate_loss.item(), validate_accuracy.item()))

    print("Train done, model saved to sample_model.pth, checkout log in ./runs with tensorboard.")
    torch.save(model.state_dict(), "sample_model.pth")
    writer.close()





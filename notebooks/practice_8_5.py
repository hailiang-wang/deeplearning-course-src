#############################################
# 在 practice_8_4.py 的基础上，进一步的扩大模型，并引入超参数自定计算
# 卷基层参数
#############################################
import torch
from matplotlib import pyplot as plt
from torchvision import datasets
from torch.utils.tensorboard import SummaryWriter
from torchinfo import summary

torch.manual_seed(100)
torch.cuda.manual_seed_all(100)

################################
# 设置默认的设备，有 GPU 的话，默认使用 GPU
################################
default_device = torch.device("cpu")
if torch.cuda.is_available():
    torch.set_default_device(torch.device("cuda"))
    default_device = torch.device("cuda")


print("default device", default_device)

################################
# 加载数据集
################################
data_path = "data-unversioned/p1ch7/"
cifar10 = datasets.CIFAR10(data_path, train=True, download=False)
cifar10_val = datasets.CIFAR10(data_path, train=False, download=False)
class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']

from torchvision import transforms

################################
# 数据的规范化
# 比如，使用 单位标准差 方法
# https://zhuanlan.zhihu.com/p/2028540638145062215
################################
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
    tuple([img.to(default_device) for (img, _) in cifar2_val]), dim=0)
validate_labels = torch.tensor([label for (_, label) in cifar2_val])

# 搭建神经网络
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

batch_size = 20
n_epoches = 10
n_channels = 128
learning_rate = 1e-4
n_out = 2  # 希望神经的输出，是一个含有两个元素的向量，
# 比如 [0.9, 0.1]，然后约定，数值较大的索引，就是分类标签，比如 0.9 的索引是 0, 0.1 的索引是 1，那么，前面的向量代表图片属于分类 0


class Net(nn.Module):
    '''
    A convolution neural network
    '''

    def __init__(self, n_channels):
        super().__init__()
        self.n_channels = n_channels
        # 输入的图片矩阵大小 3x32x32
        # 每个卷积层的卷积核大小是 3x3, 左右前后 padding 都是 1，stride 是 1
        # InputWidth + 2 - 3 + 1 = InputWidth; 高度与此类同
        self.conv1 = nn.Conv2d(3, n_channels, padding=1,
                               kernel_size=3, stride=1)
        self.conv2 = nn.Conv2d(n_channels, n_channels //
                               2, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(n_channels // 2, n_channels //
                               2, kernel_size=3, padding=1)
        self.fc1 = nn.Linear((n_channels // 2) * 4 * 4, 32)
        self.fc2 = nn.Linear(32, 2)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, x):
        out = F.max_pool2d(torch.tanh(self.conv1(x)), 2)
        out = F.max_pool2d(torch.tanh(self.conv2(out)), 2)
        out = F.max_pool2d(torch.tanh(self.conv3(out)), 2)
        # 使用 -1 自动计算 batch 大小
        out = out.view(-1, 4 * 4 * (self.n_channels // 2))
        out = torch.tanh(self.fc1(out))
        out = self.fc2(out)
        out = self.softmax(out)

        return out


model = Net(n_channels=n_channels)
summary(model=model)


# 10,2 --> (10/(10+2)), (2/(10+2))
# 将使用 softmax  = 1 / 1 + e^x
# opt = optim.SGD(params=model.parameters(), lr=learning_rate, momentum=0.9)
opt = optim.AdamW(params=model.parameters(), lr=learning_rate)
loss_fn = nn.NLLLoss()

if __name__ == "__main__":
    train_loader = torch.utils.data.DataLoader(
        cifar2, batch_size=batch_size, shuffle=True, generator=torch.Generator(device=default_device))
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
            imgs = imgs.to(default_device)
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

            # print(f'Step {total_step} epoch {epoch}, loss {loss}')

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

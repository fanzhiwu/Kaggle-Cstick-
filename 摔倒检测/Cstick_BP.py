import torch
import numpy as np
import os
import matplotlib.pyplot as plt
import torch.nn.functional as F  # 激活函数
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import torch.optim as optim
'''
数据集原地址：
https://www.kaggle.com/laavanya/elderly-fall-prediction-and-detection?select=readme.docx
'''

is_train = True
batch_size = 6
train_ratio = 0.7  #样本中训练数据的比例
feature_num =  6#输入样本特征的个数
lr = 0.001 #学习率
# 随机种子设置
random_state = 40
np.random.seed(random_state)
torch.manual_seed(random_state)  # 设置随机数种子，确保结果可重复
torch.cuda.manual_seed(random_state)
torch.cuda.manual_seed_all(random_state)
np.random.seed(random_state)
# 设置
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device = "cpu"

# 加载数据
class Dataset(Dataset):
    # 数据很小就都加载到内存
    def __init__(self,filepath):
        xy = np.loadtxt(filepath, delimiter=" ", dtype=np.float32)
        self.len=xy.shape[0]
        self.x_data = torch.from_numpy(xy[:, :-1])  # 所有行,最后一列不要
        self.y_data = torch.from_numpy(xy[:, -1])
        self.y_data = self.y_data.type(torch.LongTensor) # 分类为整数
        # 数据归一化处理
        self.mean_x = self.x_data.mean(axis=0).reshape(1, 6)  # axis = 0：压缩行，对各列求均值
        self.std_x = self.x_data.std(axis=0).reshape(1, 6)  # axis = 0,压缩行，对各列求标准差
        self.x_data = ((self.x_data - self.mean_x) / self.std_x)

    def __getitem__(self, item):
        return self.x_data[item],self.y_data[item]

    def __len__(self):
        return self.len

dataset = Dataset('Data/cStick.txt')
train_size = int(train_ratio*len(dataset))  #样本中训练数据的比例
test_size = len(dataset)-train_size


# 保存归一化参数
mean_x = np.array([i.item() for i in dataset.mean_x[0]])
std_x = np.array([i.item() for i in dataset.std_x[0]])
mean_x= torch.from_numpy(mean_x)
std_x= torch.from_numpy(std_x)




train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])
train_loader = DataLoader(dataset=train_dataset,shuffle=True,
                          # batch_size=batch_size,drop_last=True
                          )
test_loader = DataLoader(dataset=test_dataset,shuffle=False,
                         # batch_size=batch_size,drop_last=True
                         )

class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.Liner1 = torch.nn.Linear(6, 4)
        self.Liner2 = torch.nn.Linear(4, 3)
        # 归一化参数
        self.mean_x = mean_x
        self.std_x = std_x

    def forward(self, x):
        batch_size = x.size(0)
        x = F.relu(self.Liner1(x))
        x = F.relu(self.Liner2(x))
        x = x.view(batch_size, -1)
        return x


model = Model()
if(os.path.exists('Model/demo.pt')):
    model=torch.load('demo.pt')
model.to(device)

# 定义一个损失函数，来计算我们模型输出的值和标准值的差距
criterion = torch.nn.CrossEntropyLoss() #归一化---->整数化----->独热编码
# 定义一个优化器，训练模型咋训练的，就靠这个，他会反向的更改相应层的权重
optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.5)  # lr为学习率

def train(epoch):
    running_loss = 0.0
    for batch_idx, data in enumerate(train_loader, 0):  # 每次取一个样本
        inputs, targets = data
        inputs, targets = inputs.to(device), targets.to(device)
        # 优化器清零
        optimizer.zero_grad()
        inputs = inputs.view(-1,1,6)
        outs = model(inputs)

        loss = criterion(outs, targets)
        loss.backward()
        # 更新权重
        optimizer.step()
        running_loss += loss.item()


    print("第%d次训练, 训练集平均误差:%f"%(epoch,running_loss/train_size))
    return running_loss/train_size

'''
    test_data:测试集
    num:测试集数量
'''
def test(test_data,num):
    predict_y = [] #保存预测数据
    test_y = [] #保存测试集真实数据
    with torch.no_grad():  # 不用算梯度
        for batch_idx, data in enumerate(test_data, 0):
            inputs, targets = data
            inputs, targets = inputs.to(device), targets.to(device)
            inputs = inputs.view(-1, 1, 6)

            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, dim=1)

            print("测试值：",targets.item(),"真实值：",predicted.item())
            predict_y.append(predicted.item())
            test_y.append(targets.item())

    count=0
    for t1, t2 in zip(test_y, predict_y):
        if t1==t2:
            count+=1

    print("======================device:",device,"=======================")
    print("预测错误个数：[" + str(num-count)+"/"+str(num) + "]")
    print("准确率：" + str(count/num))



if __name__ == '__main__':
    train_loss=[]
    x=[]
    # 训练
    for epoch in range(100):
        train_loss.append(train(epoch))
        x.append(epoch)

    # 绘图
    plt.figure(figsize=(24, 8))
    plt.plot(x,train_loss,color='b', label='train_loss')
    torch.save(model, "Model/demo.pt")
    plt.legend(loc="upper right")
    plt.show()

    # 测试
    test_data=test_loader
    num=test_size
    test(test_data,num)






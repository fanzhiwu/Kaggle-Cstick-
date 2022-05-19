import torch
import numpy as np
import os
import torch.nn.functional as F  # 激活函数

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device = "cpu"

class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.Liner1 = torch.nn.Linear(6, 4)
        self.Liner2 = torch.nn.Linear(4, 3)

    def forward(self, x):
        batch_size = x.size(0)
        x = F.relu(self.Liner1(x))
        x = F.relu(self.Liner2(x))
        x = x.view(batch_size, -1)
        return x

model = Model()
if(os.path.exists('Model/demo.pt')):
    model=torch.load('Model/demo.pt')
model.to(device)

mean_x = model.mean_x
std_x = model.std_x

# 通过六个参数的预测函数
def predict(param):
    param = np.array(param, dtype=np.float32).reshape(1, 6)
    param = torch.from_numpy(param)

    param = ((param - mean_x) / std_x)
    param = param.type(torch.float32)
    print("param==============>>", param)
    outputs = model(param)
    _, predicted = torch.max(outputs.data, dim=1)
    predicted = predicted.item()
    if predicted == 0:
        print("状态码：", predicted, "一切正常")
    if predicted == 1:
        print("状态码：", predicted, "大概率摔倒")
    if predicted == 2:
        print("状态码：", predicted, "已经摔倒，快通知护工")
if __name__ == '__main__':
    param = [2.5950 ,2.0000 ,110.1900 ,20.2070 ,65.1900 ,1.0000]  # ==>2
    predict(param)

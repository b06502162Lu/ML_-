import os
import sys
import glob
import random
import argparse
import numpy as np
import pandas as pd
from PIL import Image
'''
    PyTorch
'''
import torch
import torch.nn as nn
from torch.optim import Adam
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
import cv2

isGPU = torch.cuda.is_available()
print ('PyTorch GPU device is available: {}'.format(isGPU))



def load_data(img_path=sys.argv[1], label_path=sys.argv[2], shuffle=True):
    train_images = sorted(glob.glob(os.path.join(img_path, '*.jpg')))
    train_labels = pd.read_csv(label_path)
    train_labels = train_labels.iloc[:,1:].values.tolist()
    train_datas = list(zip(train_images, train_labels))
    if shuffle == True:
        random.seed(14000)
        random.shuffle(train_datas)

    train_set = train_datas[:28500]
    valid_set = train_datas[28500:]

    return train_set, valid_set
"""
class dataset(Dataset):
    def __init__(self, data, transform):
        self.data = data
        self.transform = transform
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        img = Image.open(self.data[idx][0])
        img = self.transform(img)
        label = self.data[idx][1]
        return img, label
"""
train_set, valid_set = load_data()
print(train_set)


train_x = np.zeros((len(train_set), 48, 48), dtype=np.uint8)
train_y = np.zeros((len(train_set)), dtype=np.uint8)

print(len(train_set))
print(len(valid_set))
count = 0
for i, label in train_set:
        img = cv2.imread(i, cv2.IMREAD_GRAYSCALE)
        #print(i)
        #print(img.shape[0])
        #print(img.shape[1])
        if img.shape[0] != 48 or img.shape[1] != 48:
          print("something wrong")
        #print(img.shape[2])
        #print(ima.shape[3])
        for i in range(48):
          for j in range(48):
             train_x[count, i, j] = img[i][j]
       
        train_y[count] = int(label[0])
        count += 1


print(valid_set)


val_x = np.zeros((len(valid_set), 48, 48), dtype=np.uint8)
val_y = np.zeros((len(valid_set)), dtype=np.uint8)

#print(len(train_set))
print(len(valid_set))
count_val = 0
for i, label in valid_set:
        img = cv2.imread(i, cv2.IMREAD_GRAYSCALE)
        #print(i)
        #print(img.shape[0])
        #print(img.shape[1])
        if img.shape[0] != 48 or img.shape[1] != 48:
          print("something wrong")
        #print(img.shape[2])
        #print(ima.shape[3])
        for i in range(48):
          for j in range(48):
             val_x[count_val, i, j] = img[i][j]
       
        val_y[count_val] = int(label[0])
        count_val += 1


# training 時做 data augmentation
train_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.RandomHorizontalFlip(), # 隨機將圖片水平翻轉
    transforms.RandomRotation(15), # 隨機旋轉圖片
    transforms.ToTensor(), # 將圖片轉成 Tensor，並把數值 normalize 到 [0,1] (data normalization)
])

# testing 時不需做 data augmentation
test_transform = transforms.Compose([
    transforms.ToPILImage(),                                    
    transforms.ToTensor(),
])

class ImgDataset(Dataset):
    def __init__(self, x, y=None, transform=None):
        self.x = x
        # label is required to be a LongTensor
        self.y = y
        if y is not None:
            self.y = torch.LongTensor(y)
        self.transform = transform
    def __len__(self):
        return len(self.x)
    def __getitem__(self, index):
        X = self.x[index]
        if self.transform is not None:
            X = self.transform(X)
        if self.y is not None:
            Y = self.y[index]
            return X, Y
        else:
            return X


batch_size = 128
train_set = ImgDataset(train_x, train_y, train_transform)
val_set = ImgDataset(val_x, val_y, test_transform)
train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)


class Classifier(nn.Module):
    def __init__(self):
        super(Classifier, self).__init__()
        # torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        # torch.nn.MaxPool2d(kernel_size, stride, padding)
        
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=5, padding=2),
            nn.LeakyReLU(negative_slope=0.05),
            nn.BatchNorm2d(32),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.LeakyReLU(negative_slope=0.05),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, kernel_size=3,padding=1),
            nn.LeakyReLU(negative_slope=0.05),
            nn.BatchNorm2d(128),
            nn.MaxPool2d(2),

            nn.Conv2d(128, 128, kernel_size=3,padding=1),
            nn.LeakyReLU(negative_slope=0.05),
            nn.BatchNorm2d(128),
            nn.MaxPool2d(2),

            
            
            
        )
        self.fc = nn.Sequential(
            nn.Linear(3*3*128, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Linear(256, 7)
        )

    def forward(self, x):
        out = self.cnn(x)
        out = out.view(out.size()[0], -1)
        return self.fc(out)

import time
model = Classifier().cuda()
loss = nn.CrossEntropyLoss() # 因為是 classification task，所以 loss 使用 CrossEntropyLoss
optimizer = torch.optim.Adam(model.parameters(), lr=0.001) # optimizer 使用 Adam
num_epoch = 50

for epoch in range(num_epoch):
    print(epoch)
    epoch_start_time = time.time()
    train_acc = 0.0
    train_loss = 0.0
    val_acc = 0.0
    val_loss = 0.0

    model.train() # 確保 model 是在 train model (開啟 Dropout 等...)
    for i, data in enumerate(train_loader):
        optimizer.zero_grad() # 用 optimizer 將 model 參數的 gradient 歸零
        train_pred = model(data[0].cuda()) # 利用 model 得到預測的機率分佈 這邊實際上就是去呼叫 model 的 forward 函數
        batch_loss = loss(train_pred, data[1].cuda()) # 計算 loss （注意 prediction 跟 label 必須同時在 CPU 或是 GPU 上）
        batch_loss.backward() # 利用 back propagation 算出每個參數的 gradient
        optimizer.step() # 以 optimizer 用 gradient 更新參數值

        train_acc += np.sum(np.argmax(train_pred.cpu().data.numpy(), axis=1) == data[1].numpy())
        train_loss += batch_loss.item()
    
    model.eval()
    with torch.no_grad():
        for i, data in enumerate(val_loader):
            val_pred = model(data[0].cuda())
            batch_loss = loss(val_pred, data[1].cuda())

            val_acc += np.sum(np.argmax(val_pred.cpu().data.numpy(), axis=1) == data[1].numpy())
            val_loss += batch_loss.item()

        #將結果 print 出來
        print('[%03d/%03d] %2.2f sec(s) Train Acc: %3.6f Loss: %3.6f | Val Acc: %3.6f loss: %3.6f' % \
            (epoch + 1, num_epoch, time.time()-epoch_start_time, \
             train_acc/train_set.__len__(), train_loss/train_set.__len__(), val_acc/val_set.__len__(), val_loss/val_set.__len__()))


torch.save(model, 'model14.pkl')
torch.save(model.state_dict(), 'model14_params.pkl')

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

# testing 時不需做 data augmentation
test_transform = transforms.Compose([
    transforms.ToPILImage(),                                    
    transforms.ToTensor(),
])

#處理test data
count_test = 0
image_dir = sorted(os.listdir(sys.argv[1]))
print(image_dir)
print(len(image_dir))
test_x = np.zeros((7178, 48, 48), dtype=np.uint8)

for image in image_dir :
    
    if image==".DS_Store":
      continue
    path = (str(sys.argv[1])+"/")+image
    #print(path)
    img = cv2.imread(path,cv2.IMREAD_GRAYSCALE)
    #print(img)
    for i in range(48):
      for j in range(48):
          test_x[count_test, i, j] = img[i][j]
    count_test += 1
     

print("Size of Testing data = {}".format(len(test_x)))

print(test_x.shape[0])
print(test_x.shape[1])
print(test_x.shape[2])
#print(test_x.shape[3])

basis = torch.load('basis.pkl')
basis.load_state_dict(torch.load('basis_params.pkl'))

model2 = torch.load('model2.pkl')
model2.load_state_dict(torch.load('model2_params.pkl'))

model3 = torch.load('model3.pkl')
model3.load_state_dict(torch.load('model3_params.pkl'))

model4 = torch.load('model4.pkl')
model4.load_state_dict(torch.load('model4_params.pkl'))

model5 = torch.load('model5.pkl')
model5.load_state_dict(torch.load('model5_params.pkl'))

model6 = torch.load('model6.pkl')
model6.load_state_dict(torch.load('model6_params.pkl'))

model7 = torch.load('model7.pkl')
model7.load_state_dict(torch.load('model7_params.pkl'))

model8 = torch.load('model8.pkl')
model8.load_state_dict(torch.load('model8_params.pkl'))

model9 = torch.load('model9.pkl')
model9.load_state_dict(torch.load('model9_params.pkl'))

model10 = torch.load('model10.pkl')
model10.load_state_dict(torch.load('model10_params.pkl'))

model11 = torch.load('model11.pkl')
model11.load_state_dict(torch.load('model11_params.pkl'))

model12 = torch.load('model12.pkl')
model12.load_state_dict(torch.load('model12_params.pkl'))

model13 = torch.load('model13.pkl')
model13.load_state_dict(torch.load('model13_params.pkl'))

model14 = torch.load('model14.pkl')
model14.load_state_dict(torch.load('model14_params.pkl'))

model15 = torch.load('model15.pkl')
model15.load_state_dict(torch.load('model15_params.pkl'))


batch_size = 128

test_set = ImgDataset(test_x, transform=test_transform)
test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)


basis.eval()
prediction_basis = []
with torch.no_grad():
    for i, data in enumerate(test_loader):
        test_pred = basis(data.cuda())
        test_label = np.argmax(test_pred.cpu().data.numpy(), axis=1)
        for y in test_label:
            prediction_basis.append(y)
#print(prediction_basis)

model2.eval()
prediction_model2 = []
with torch.no_grad():
    for i, data in enumerate(test_loader):
        test_pred = model2(data.cuda())
        test_label = np.argmax(test_pred.cpu().data.numpy(), axis=1)
        for y in test_label:
            prediction_model2.append(y)

model3.eval()
prediction_model3 = []
with torch.no_grad():
    for i, data in enumerate(test_loader):
        test_pred = model3(data.cuda())
        test_label = np.argmax(test_pred.cpu().data.numpy(), axis=1)
        for y in test_label:
            prediction_model3.append(y)

model4.eval()
prediction_model4 = []
with torch.no_grad():
    for i, data in enumerate(test_loader):
        test_pred = model4(data.cuda())
        test_label = np.argmax(test_pred.cpu().data.numpy(), axis=1)
        for y in test_label:
            prediction_model4.append(y)

model5.eval()
prediction_model5 = []
with torch.no_grad():
    for i, data in enumerate(test_loader):
        test_pred = model5(data.cuda())
        test_label = np.argmax(test_pred.cpu().data.numpy(), axis=1)
        for y in test_label:
            prediction_model5.append(y)

model6.eval()
prediction_model6 = []
with torch.no_grad():
    for i, data in enumerate(test_loader):
        test_pred = model6(data.cuda())
        test_label = np.argmax(test_pred.cpu().data.numpy(), axis=1)
        for y in test_label:
            prediction_model6.append(y)


model7.eval()
prediction_model7 = []
with torch.no_grad():
    for i, data in enumerate(test_loader):
        test_pred = model7(data.cuda())
        test_label = np.argmax(test_pred.cpu().data.numpy(), axis=1)
        for y in test_label:
            prediction_model7.append(y)

model8.eval()
prediction_model8 = []
with torch.no_grad():
    for i, data in enumerate(test_loader):
        test_pred = model8(data.cuda())
        test_label = np.argmax(test_pred.cpu().data.numpy(), axis=1)
        for y in test_label:
            prediction_model8.append(y)

model9.eval()
prediction_model9 = []
with torch.no_grad():
    for i, data in enumerate(test_loader):
        test_pred = model9(data.cuda())
        test_label = np.argmax(test_pred.cpu().data.numpy(), axis=1)
        for y in test_label:
            prediction_model9.append(y)


model10.eval()
prediction_model10 = []
with torch.no_grad():
    for i, data in enumerate(test_loader):
        test_pred = model10(data.cuda())
        test_label = np.argmax(test_pred.cpu().data.numpy(), axis=1)
        for y in test_label:
            prediction_model10.append(y)

model11.eval()
prediction_model11 = []
with torch.no_grad():
    for i, data in enumerate(test_loader):
        test_pred = model11(data.cuda())
        test_label = np.argmax(test_pred.cpu().data.numpy(), axis=1)
        for y in test_label:
            prediction_model11.append(y)


model12.eval()
prediction_model12 = []
with torch.no_grad():
    for i, data in enumerate(test_loader):
        test_pred = model12(data.cuda())
        test_label = np.argmax(test_pred.cpu().data.numpy(), axis=1)
        for y in test_label:
            prediction_model12.append(y)
model13.eval()
prediction_model13 = []
with torch.no_grad():
    for i, data in enumerate(test_loader):
        test_pred = model13(data.cuda())
        test_label = np.argmax(test_pred.cpu().data.numpy(), axis=1)
        for y in test_label:
            prediction_model13.append(y)

model14.eval()
prediction_model14 = []
with torch.no_grad():
    for i, data in enumerate(test_loader):
        test_pred = model14(data.cuda())
        test_label = np.argmax(test_pred.cpu().data.numpy(), axis=1)
        for y in test_label:
            prediction_model14.append(y)

model15.eval()
prediction_model15 = []
with torch.no_grad():
    for i, data in enumerate(test_loader):
        test_pred = model15(data.cuda())
        test_label = np.argmax(test_pred.cpu().data.numpy(), axis=1)
        for y in test_label:
            prediction_model15.append(y)

matrix = (np.array(prediction_basis)).reshape(-1,1)
print(matrix)

matrix = np.concatenate((matrix,(np.array(prediction_model2)).reshape(-1,1)),axis=1)
matrix = np.concatenate((matrix,(np.array(prediction_model3)).reshape(-1,1)),axis=1)
matrix = np.concatenate((matrix,(np.array(prediction_model4)).reshape(-1,1)),axis=1)
matrix = np.concatenate((matrix,(np.array(prediction_model5)).reshape(-1,1)),axis=1)
matrix = np.concatenate((matrix,(np.array(prediction_model6)).reshape(-1,1)),axis=1)
matrix = np.concatenate((matrix,(np.array(prediction_model7)).reshape(-1,1)),axis=1)
matrix = np.concatenate((matrix,(np.array(prediction_model8)).reshape(-1,1)),axis=1)
matrix = np.concatenate((matrix,(np.array(prediction_model9)).reshape(-1,1)),axis=1)
matrix = np.concatenate((matrix,(np.array(prediction_model10)).reshape(-1,1)),axis=1)
matrix = np.concatenate((matrix,(np.array(prediction_model11)).reshape(-1,1)),axis=1)
matrix = np.concatenate((matrix,(np.array(prediction_model12)).reshape(-1,1)),axis=1)
matrix = np.concatenate((matrix,(np.array(prediction_model13)).reshape(-1,1)),axis=1)
matrix = np.concatenate((matrix,(np.array(prediction_model14)).reshape(-1,1)),axis=1)
matrix = np.concatenate((matrix,(np.array(prediction_model15)).reshape(-1,1)),axis=1)





print((matrix).shape[0])
print(matrix.shape[1])

print(matrix[0][:])



"""
#將結果寫入 csv 檔
with open(sys.argv[2], 'w') as f:
    f.write('id,label\n')
    for i, y in  enumerate(prediction):
        f.write('{},{}\n'.format(i, y))
"""

def find_max(array_):
    d = dict()
    d[0] =0
    d[1] =0
    d[2] =0
    d[3] =0
    d[4] =0
    d[5] =0
    d[6] =0
    for i in range(15):
        d[array_[i]] = d[array_[i]] +1

    l = list()
    for i in range(7):
        l.append(d[i])
    l.sort()
    for i in d :
        if d[i]==l[-1]:
            return i

ans = list()
for i in range(matrix.shape[0]):
  label = find_max(matrix[i])
  ans.append(label)
#print(len(ans))


#將結果寫入 csv 檔
with open(sys.argv[2], 'w') as f:
    f.write('id,label\n')
    for i, y in  enumerate(ans):
        f.write('{},{}\n'.format(i, y))

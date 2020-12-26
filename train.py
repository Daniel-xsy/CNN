from dataLoader import MNISTDataset
from torchvision.transforms import transforms
from CNN import CNN
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
import time
import torch.optim as optim

def main():
    #加载数据集
    train_dataset=MNISTDataset(root_dir='../MNIST/',set_name='train',transform=transforms.ToTensor())
    test_dataset=MNISTDataset(root_dir='../MNIST/',set_name='test',transform=transforms.ToTensor())
    dataloader_train = DataLoader(train_dataset, num_workers=0,batch_size=64)
    dataloader_test=DataLoader(test_dataset, num_workers=0,batch_size=1)

    #使用GPU计算
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
    cnn=CNN().to(device)
    optimizer=optim.Adam(cnn.parameters(),lr=0.01)
    loss_func=nn.CrossEntropyLoss()
    accuracy=[]
    for epoch in range(3):
        for iter_num,(x,y) in enumerate(dataloader_train):
            output=cnn(x.to(device))
            loss=loss_func(output,y.to(device))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            print(
                'Epoch: {} | Iteration: {} | loss: {:1.5f}'.format(
                    epoch,iter_num,loss.data
                )               
            )

        
        err_count=0
        print('evaluating...')
        for step,(x,y) in enumerate(dataloader_test):
            pred=cnn(x.to(device))
            pred=pred.cpu()
            result=np.argmax(list(pred[0]))
            if result!=y:
                err_count+=1
        length=len(dataloader_test)
        print('正确分类: {} | 错误分类: {} | 准确率: {} '.format(
            length-err_count,err_count,float((length-err_count)/length)))
        accuracy.append((length-err_count)/length)
        time.sleep(4)
    print('saving model...')
    torch.save(cnn,'/cnn')
    for i in range(len(accuracy)):
        print('Epoch: {} | accuracy: {.4f}'.format(i,accuracy[i]))


if __name__ == "__main__":
    main()
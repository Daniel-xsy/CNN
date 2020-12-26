import torch.nn as nn

class CNN(nn.Module):

    def __init__(self):
        super(CNN,self).__init__()
        self.cov1=nn.Sequential(
            nn.Conv2d(
                in_channels=1,
                out_channels=16,
                kernel_size=2
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
        )
        self.cov2=nn.Sequential(
            nn.Conv2d(
                in_channels=16,
                out_channels=32,
                kernel_size=2
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3),
        )
        self.fc=nn.Linear(512,10)

    def forward(self,x):
        x=self.cov1(x)
        x=self.cov2(x)
        x=x.view(x.size(0),-1)
        x=self.fc(x)
        return x

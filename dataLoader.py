import numpy as np
from skimage import io
from skimage import transform
import matplotlib.pyplot as plt
import os
import torch
import torchvision
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import transforms
from PIL import Image
import gzip

class MNISTDataset(Dataset):
    def __init__(self,root_dir,set_name='train',transform=None):
        self.root_dir=root_dir
        self.set_name=set_name
        self.transform=transform
        (image,label)=self.load_data()
        self.image=image
        self.label=label
    def load_data(self):
        #读取MNIST数据集
        with gzip.open(os.path.join(self.root_dir,self.set_name+'-labels-idx1-ubyte.gz'),'rb') as lbpath:
            label = np.frombuffer(lbpath.read(), np.uint8, offset=8)
        with gzip.open(os.path.join(self.root_dir,self.set_name+'-images-idx3-ubyte.gz'),'rb') as imgpath:
            image = np.frombuffer(
                imgpath.read(), np.uint8, offset=16).reshape(len(label), 28, 28)
        return (image,label)
    def __getitem__(self,index):
        img,label=self.image[index],int(self.label[index])
        if self.transform is not None:
            img=self.transform(img)
        return img,label
    def __len__(self):
        return len(self.image)
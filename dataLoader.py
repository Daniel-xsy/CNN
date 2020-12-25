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
    def loaddata(self):
        with gzip.open(os.path.join(self.root_dir,self.set_name+'.idx1-ubyte'),'rb') as lbpath:
            label = np.frombuffer(lbpath.read(), np.uint8, offset=8)
        with gzip.open(os.path.join(self.root_dir,self.set_name+'.idx3-ubyte'),'rb') as imgpath:
            image = np.frombuffer(
                imgpath.read(), np.uint8, offset=16).reshape(len(label), 28, 28)
        return (image,label)
    def __getitem__(self,index):
        pass
    def __len__(self):
        pass
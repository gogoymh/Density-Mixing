import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, datasets
import os
import numpy as np
import random
import json

class New_CIFAR10(Dataset):
    def __init__(self, path, input_transform=None, num_class=10):
        super().__init__()
        
        save_file = os.path.join(path, 'class_table.txt')
        self.input_transform = input_transform
        
        baseset = datasets.CIFAR10(
                        "./data/CIFAR10",
                        train=True,
                        download=True
                        )
        
        if os.path.isfile(save_file):
            with open(save_file, "r") as fp:
                self.class_table = json.load(fp)
            
        else:            
            class_table = []
            for i in range(num_class):
                class_table.append([])
            
            for i in range(len(baseset)):
                _, y = baseset.__getitem__(i)
                
                class_table[y].append(i)
                print("[%d/%d] is done." % (i, len(baseset)))
            
            with open(save_file, "w") as fp:
                json.dump(class_table, fp)
            
            self.class_table = class_table
        
        self.baseset = baseset
        self.len = len(baseset)
        
    def __getitem__(self, index):
        x1, y1 = self.baseset.__getitem__(index)
        
        '''
        if np.random.uniform() >= 0.2:
            mix_index = np.random.choice(self.class_table[y], 1)[0]
            x2, _ = self.baseset.__getitem__(mix_index)
        else:
            x2 = x1
        '''
        
        #mix_index = np.random.choice(self.class_table[y], 1)[0]
        #x2, _ = self.baseset.__getitem__(mix_index)
        
        rand_index = np.random.choice(self.len, 1)[0]
        x2, y2 = self.baseset.__getitem__(rand_index)
        
        '''
        classes = [0,1,2,3,4,5,6,7,8,9]
        classes.pop(y1)
        rand_class = np.random.choice(classes, 1)[0]
        rand_index = np.random.choice(self.class_table[rand_class], 1)[0]
        x2, y2 = self.baseset.__getitem__(rand_index)
        '''
        
        if self.input_transform is not None:
            x1 = self.input_transform(x1)
            x2 = self.input_transform(x2)
        
        y = torch.Tensor([y1, y2]).long()
        
        return x1, x2, y
    
    def __len__(self):
        return self.len
    
if __name__ == "__main__":
    path = "C://유민형//개인 연구//Density Mixing//"
    a = New_CIFAR10(path)
    

    import matplotlib.pyplot as plt
    
    index = np.random.choice(50000, 1)[0]
    #index = 0
    b1, b2, c = a.__getitem__(index)
    
    plt.imshow(b1)
    plt.show()
    plt.close()
    
    plt.imshow(b2)
    plt.show()
    plt.close()
    
    print(c)
    
    b1 = transforms.ToTensor()(b1)
    b2 = transforms.ToTensor()(b2)
    
    b1 = b1.unsqueeze(0)
    b2 = b2.unsqueeze(0)
    
    k = 8
    
    d1 = b1.unfold(2,k,k).unfold(3,k,k).reshape(1,3,int((32/k)**2),k,k).permute(0,2,1,3,4)
    d2 = b2.unfold(2,k,k).unfold(3,k,k).reshape(1,3,int((32/k)**2),k,k).permute(0,2,1,3,4)
    
    
    num = np.random.choice(int((32/k)**2), 1)[0]
    #num = np.random.choice([i for i in range(20,45)], 1)[0]
    e = np.random.choice(int((32/k)**2), num, replace=False)
    d1[:,e,:,:,:] = d2[:,e,:,:,:]
    '''
    num = np.random.choice(int((32/k)**2), 1)[0]
    d1[:,num:,:,:,:] = d2[:,num:,:,:,:]
    '''
    
    f1 = d1.permute(0,2,1,3,4).reshape(1,3,int(32/k),int(32/k),k,k)
    g1 = f1.permute(0,1,2,4,3,5).reshape(1,3,32,32)
    
    plt.imshow(g1.squeeze().numpy().transpose(1,2,0))
    plt.show()
    plt.close()
    
    
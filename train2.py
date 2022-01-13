import torch
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
import torch.nn as nn
import torch.optim as optim
import numpy as np

from resnet import resnet56_mix, classifier, resnet56_part1, resnet56_part2
from new_cifar10 import New_CIFAR10


#path = "C://유민형//개인 연구//Density Mixing//"
#path = "/DATA/ymh/density_mixing"
path = "/home/compu/ymh/Density Mixing"
train_loader = DataLoader(New_CIFAR10(path, input_transform=transforms.Compose([
                                transforms.RandomCrop(32, padding=4),
                                transforms.RandomHorizontalFlip(),
                                transforms.ToTensor(),
                                transforms.Normalize(
                                        (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))]
                                )), batch_size=128, shuffle=True)

test_loader = DataLoader(
                datasets.CIFAR10(
                        './data/CIFAR10',
                        train=False,
                        download=True,
                        transform=transforms.Compose([
                                transforms.ToTensor(),
                                transforms.Normalize(
                                        (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))]
                                ),
                        ),
                batch_size=64, shuffle=False)

device = torch.device("cuda:0")
model1 = resnet56_mix()
#model = nn.DataParallel(model)
model1.to(device)

model2 = classifier()
#model = nn.DataParallel(model)
model2.to(device)

'''
pretrained_path = "c://results/resnet56_pretrained.pth" # your path
if device == "cuda:0":
    checkpoint = torch.load(pretrained_path)
else:
    checkpoint = torch.load(pretrained_path, map_location=lambda storage, location: 'cpu')
model.load_state_dict(checkpoint['model_state_dict'])
'''

params = list(model1.parameters()) + list(model2.parameters())
#optimizer = optim.Adam(params, lr=0.1)
optimizer = optim.SGD(params, lr=0.1)
scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[50,150], gamma=0.1)

#criterion = nn.CrossEntropyLoss()
def softXEnt (input, target):
    logprobs = nn.functional.log_softmax (input, dim = 1)
    return  -(target * logprobs).sum() / input.shape[0]

def make_target(batch_size, y, num):
    target = torch.zeros((batch_size, 10))
    value = torch.zeros((batch_size, 2))
    val = num / 64
    value[:,0] = val
    value[:,1] = 1 - val
    return target.scatter_add_(1,y,value)

best_acc = 0
for epoch in range(300):
    model1.train()
    model2.train()
    runnning_loss = 0
    for x1, x2, y in train_loader:
        optimizer.zero_grad()
               
        latent1 = model1(x1.float().to(device))
        latent2 = model1(x2.float().to(device))
        
        num = np.random.choice(64,1)[0]
        where = np.random.choice(64, num, replace=False)
        
        latent1[:, where] = latent2[:, where]
        
        output = model2(latent1)
        
        target = make_target(y.shape[0], y, num)
        loss = softXEnt(output, target.to(device))
        
        #loss = criterion(output, y.long().to(device))
        loss.backward()
        optimizer.step()
        runnning_loss += loss.item()
        #print(loss.item())
        
    runnning_loss /= len(train_loader)
    print("[Epoch:%d] [Loss:%f]" % ((epoch+1), runnning_loss), end=" ")
    
    accuracy = 0
    with torch.no_grad():
        model1.eval()
        model2.eval()
        correct = 0
        for x, y in test_loader:
            latent = model1(x.float().to(device))
            output = model2(latent)
            pred = output.argmax(1, keepdim=True)
            correct += pred.eq(y.long().to(device).view_as(pred)).sum().item()
                
        accuracy = correct / len(test_loader.dataset)
        '''
        if save:
            torch.save({'epoch': epoch,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'loss': losses[epoch].item()}, "/home/cscoi/MH/resnet56_real4.pth")
            save_again = False
            print("[Accuracy:%f]" % accuracy)
            print("Saved early")
            break
        ''' 
        if accuracy >= best_acc:
            print("[Accuracy:%f] **Best**" % accuracy)
            best_acc = accuracy
        else:
            print("[Accuracy:%f]" % accuracy)
        
        
    scheduler.step()
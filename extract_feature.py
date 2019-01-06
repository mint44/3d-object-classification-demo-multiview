import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from torch.autograd import Variable
from PIL import Image
import torch.utils.data as data
import pandas as pd
import os
from skimage import io
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import numpy as np

resnet = models.resnet50(pretrained=True)
modules=list(resnet.children())[:-1]
resnet=nn.Sequential(*modules)
for p in resnet.parameters():
    p.requires_grad = False

class ImageDataset(data.Dataset):
  """My Image dataset."""
  
  def __init__(self, csv_file, root_dir, transform=None):
      """
      Args:
          csv_file (string): Path to the csv file with annotations.
          root_dir (string): Directory with all the images.
          transform (callable, optional): Optional transform to be applied
              on a sample.
      """
      self.csv_frame = pd.read_csv(csv_file,header = None, delim_whitespace=True)
      self.root_dir = root_dir
      self.transform = transform

  def __len__(self):
      return len(self.csv_frame)

  def __getitem__(self, idx):
      img_name = os.path.join(self.root_dir,
                              self.csv_frame.iloc[idx, 0])
      sample = io.imread(img_name.strip())[:,:,:3] #keep RGB, remove A
      target = self.csv_frame.iloc[idx, 1]


      if self.transform:
          sample = self.transform(sample)

      return sample, target

class ImageDatasetFromList(data.Dataset):
  """My Image dataset."""
  
  def __init__(self, list_file, transform=None):
      """
      Args:
          csv_file (string): Path to the csv file with annotations.
          root_dir (string): Directory with all the images.
          transform (callable, optional): Optional transform to be applied
              on a sample.
      """
      self.list_file = list_file
      self.transform = transform

  def __len__(self):
      return len(self.list_file)

  def __getitem__(self, idx):
      img_name =self.list_file[idx]
      sample = io.imread(img_name)[:,:,:3] #keep RGB, remove A

      if self.transform:
          sample = self.transform(sample)

      return sample, -1 # -1 is place holder for none class


def extract(model, data_loader):
    result_x = np.zeros( (len(trainset_loader.dataset) , 2048))
    result_target = np.zeros( (len(trainset_loader.dataset),1 ))
    t = 0
    model.eval()
    for batch_idx, (x, target) in enumerate(tqdm.tqdm(data_loader)):
        x, target = Variable(x, volatile=True), Variable(target, volatile=True)
        #print 'batch: ', batch_idx, 'size', x.shape
        out = model(x)
        out = out.cpu().data.numpy()
        out = np.squeeze(out)
        result_x[t: t + out.shape[0] , :] = out
        result_target[t:t+out.shape[0] , :] = target
        t = t + out.shape[0] 
        #print out.shape
        
    return result_x, result_target



def extract_cpu(model, data_loader):
    result_x = np.zeros( (len(data_loader.dataset) , 2048))
    t = 0
    model.eval()
    for batch_idx, (x, target) in enumerate(data_loader):
        #x, target = Variable(x), Variable(target)
        #print 'batch: ', batch_idx, 'size', x.shape
        out = model(x)
        out = out.data.numpy()
        out = np.squeeze(out)
        result_x[t: t + out.shape[0] , :] = out
        t = t + out.shape[0] 
        #print out.shape
        
    return result_x


def extract_feature_from_list_imgs(list_imgs):
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    transfomation = transforms.Compose([
                transforms.ToPILImage(),
                #transforms.Resize((224,224)),
                transforms.ToTensor(),
                normalize,
            ])
    trainset = ImageDatasetFromList(list_imgs, transfomation)

    trainset_loader = torch.utils.data.DataLoader(trainset,
                                             batch_size=4, shuffle=False,
                                             num_workers=4)
    
    x = extract_cpu(resnet, trainset_loader)
    return x

'''
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
transfomation = transforms.Compose([
            transforms.ToPILImage(),
            #transforms.Resize((224,224)),
            transforms.ToTensor(),
            normalize,
        ])
trainset = ImageDataset('/home/minhb/ModelNet40/data/ModelNet40/modelnet40_test_list_label_26.csv', '/home/minhb/ModelNet40/data/ModelNet40/render_modelnet40_26/', transfomation)

trainset_loader = torch.utils.data.DataLoader(trainset,
                                             batch_size=17, shuffle=False,
                                             num_workers=2)

resnet = resnet.cuda()

len(trainset_loader.dataset)

x,y = extract(resnet, trainset_loader)

np.save('x_test',x)
np.save('y_test',y)
'''
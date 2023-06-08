import glob
import random
import os

from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms
import numpy as np

class ImageDataset(Dataset):
    def __init__(self, root, transforms_=None, unaligned=False, mode='train'):
        self.transform = transforms.Compose(transforms_)
        self.unaligned = unaligned

        self.files_A = sorted(glob.glob(os.path.join(root, '%s/A' % mode) + '/*.*'))
        self.files_B = sorted(glob.glob(os.path.join(root, '%s/B' % mode) + '/*.*'))

    def __getitem__(self, index):
        item_A = self.transform(Image.open(self.files_A[index % len(self.files_A)]))

        if self.unaligned:
            item_B = self.transform(Image.open(self.files_B[random.randint(0, len(self.files_B) - 1)]))
        else:
            item_B = self.transform(Image.open(self.files_B[index % len(self.files_B)]))

        return {'A': item_A, 'B': item_B}

    def __len__(self):
        return max(len(self.files_A), len(self.files_B))


class Animeset(Dataset):
    """
        torch.transform has two effects:
            1.  normalizing your data no matter what. If it is 
                a uint8 image data, all values are divided by 255(max)
            2. Changing channel order from H W C(Pillow/opencv) to C H W 
    """
    def __init__(self, dataset_dir, transforms_=None, ):
        self.transforms_ = transforms_
        self.transform = transforms.Compose(transforms_)
        
        self.filelistA = glob.glob(dataset_dir + '/sketch/*.jpg') 
        self.filelistB = glob.glob(dataset_dir + '/frame/*.jpg') 
        
    def __len__(self):
        return max(len(self.filelistA), len(self.filelistB))
    
    def __getitem__(self, index):
        img_A = Image.open(self.filelistA[index % len(self.filelistA)])
        img_B = Image.open(self.filelistB[random.randint(0, len(self.filelistB) - 1)])
        if self.transforms_:
            """
            Only takes in PIL(Image.open) or Tensor(detransform)
            If data are read with numpy, convert to PIL first + [transform.ToPILImage()]
            """
            img_A = self.transform(img_A)
            img_B = self.transform(img_B)
        return {'A':img_A, 'B':img_B}
        
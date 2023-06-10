import argparse
import sys
import os

import torchvision.transforms as transforms
from torchvision.utils import save_image
from torch.utils.data import DataLoader
from torch.autograd import Variable
import torch
import numpy as np

from networks.models import Generator
from datasets import Animeset

from networks.UNet import UNet
from networks.ESRGAN import RRDBNet, GeneratorRRDB

parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', type=int, default=1, help='size of the batches')
parser.add_argument('--dataroot', type=str, default='./dataset', help='root directory of the dataset')
parser.add_argument('--input_nc', type=int, default=3, help='number of channels of input data')
parser.add_argument('--output_nc', type=int, default=3, help='number of channels of output data')
parser.add_argument('--size', type=int, default=256, help='size of the data (squared assumed)')
parser.add_argument('--cuda', type=bool, default=True, help='use GPU computation')
parser.add_argument('--n_cpu', type=int, default=8, help='number of cpu threads to use during batch generation')
parser.add_argument('--generator_A2B', type=str, default='./models/20_RRDB_netG_A2B.pth', help='A2B generator checkpoint file')
parser.add_argument('--generator_B2A', type=str, default='./models/20_RRDB_netG_B2A.pth', help='B2A generator checkpoint file')
parser.add_argument('--mode', type=str, default='test', help='use test for just generating sketch')
parser.add_argument('--patch_n', type=int, default=1, help='divide input in case CUDA out of memory, n^2')
args = parser.parse_args()
print(args)

###### Definition of variables ######
# Networks
netG_A2B = GeneratorRRDB(args.input_nc)
netG_B2A = GeneratorRRDB(args.input_nc)

netG_A2B.cuda()
netG_B2A.cuda()

# Load state dicts
netG_A2B.load_state_dict(torch.load(args.generator_A2B))
netG_B2A.load_state_dict(torch.load(args.generator_B2A))

# Set model's test mode
netG_A2B.eval()
netG_B2A.eval()

# Inputs & targets memory allocation
Tensor = torch.cuda.FloatTensor if args.cuda else torch.Tensor

MEAN = (0.5, 0.5, 0.5)
STD = (0.5, 0.5, 0.5)
de_transform_ = transforms.Compose([transforms.Normalize(mean=(0, 0, 0), std=tuple(1/x for x in STD)),
                                    transforms.Normalize(mean=tuple(-x for x in MEAN), std=(1, 1, 1))])

# Dataset loader
transforms_ = [ transforms.ToTensor(),
                transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5)) ]
dataloader = DataLoader(Animeset(args.dataroot, transforms_=transforms_, mode=args.mode), 
                        batch_size=args.batch_size, shuffle=False, num_workers=args.n_cpu)
###################################

###### Testing######

# Create output dirs if they don't exist
out_dir = './output/infer/imgs/'
if not os.path.exists(out_dir):
    os.makedirs(out_dir)

for cnt, batch in enumerate(dataloader):
    # if cnt > 20: break
    if args.mode == 'train':
        input_A = Tensor(batch['A'].shape)
        input_B = Tensor(batch['B'].shape)
        real_A = Variable(input_A.copy_(batch['A']))
        real_B = Variable(input_B.copy_(batch['B']))
        fake_B = netG_A2B(real_A).data
        fake_A = netG_B2A(real_B).data
    else:
        # set patch = 4 if out of cuda memory        
        patch_n = args.patch_n
        p_h, p_w = int(batch['B'].shape[2]/np.sqrt(patch_n)), int(batch['B'].shape[3]/np.sqrt(patch_n))
        input_B = Tensor(batch['B'].shape[0], batch['B'].shape[1], p_h, p_w)
        fake_A = Tensor(batch['B'].shape)
        for p in range(patch_n):
            i, j = int(p // np.sqrt(patch_n)), int(p % np.sqrt(patch_n))
            
            patch = batch['B'][:, :, i*p_h:(i+1)*p_h, j*p_w:(j+1)*p_w]
            
            real_B_patch = Variable(input_B.copy_(patch))
            fake_A_patch = netG_B2A(real_B_patch).data
            
            fake_A[:, :, i*p_h:(i+1)*p_h, j*p_w:(j+1)*p_w] = fake_A_patch
        
        input_B = Tensor(batch['B'].shape)
        real_B = Variable(input_B.copy_(batch['B']))
        
    # Save image files
    print("Saving===============>")
    save_image(fake_A[0], out_dir + str(cnt) + '_fake_A.jpg')
    save_image(de_transform_(real_B)[0], out_dir + str(cnt) + '_real_B.jpg')
    if args.mode == 'train':
        save_image(de_transform_(fake_B)[0], out_dir + str(cnt) + '_fake_B.jpg')
        save_image(real_A[0], out_dir + str(cnt) + '_real_A.jpg')
    
    sys.stdout.write('\rGenerated images %04d of %04d' % (cnt+1, len(dataloader)))

sys.stdout.write('\n')
###################################

import os

import argparse
import itertools

from torchvision.utils import save_image
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.autograd import Variable
from PIL import Image
import torch
import numpy as np

from networks.models import Generator
from networks.models import Discriminator
from utils.utils import ReplayBuffer
from utils.utils import LambdaLR
from utils.utils import Logger
from utils.utils import weights_init_normal
from datasets import ImageDataset, Animeset

from matplotlib import pyplot as plt
from tqdm import tqdm
from networks.UNet import UNet
from networks.ESRGAN import RRDBNet

import wandb
wandb.init(project="Anime-cycleGAN")

parser = argparse.ArgumentParser()
parser.add_argument('--epoch', type=int, default=0, help='starting epoch')
parser.add_argument('--n_epochs', type=int, default=200, help='number of epochs of training')
parser.add_argument('--batch_size', type=int, default=1, help='size of the batches')
parser.add_argument('--dataroot', type=str, default='./dataset/', help='root directory of the dataset')
parser.add_argument('--lr', type=float, default=0.0002, help='initial learning rate')
parser.add_argument('--decay_epoch', type=int, default=100, help='epoch to start linearly decaying the learning rate to 0')
parser.add_argument('--size', type=int, default=256, help='size of the data crop (squared assumed)')
parser.add_argument('--input_nc', type=int, default=3, help='number of channels of input data')
parser.add_argument('--output_nc', type=int, default=3, help='number of channels of output data')
parser.add_argument('--cuda', type=bool, default=True, help='use GPU computation')
parser.add_argument('--n_cpu', type=int, default=8, help='number of cpu threads to use during batch generation')
args = parser.parse_args()
print(args)

###### Definition of variables ######
# Networks
netG_A2B = RRDBNet(args.input_nc, args.output_nc, 32, 2)
netG_B2A = RRDBNet(args.input_nc, args.output_nc, 32, 2)
netD_A = Discriminator(args.input_nc)
netD_B = Discriminator(args.output_nc)

if args.cuda:
    netG_A2B.cuda()
    netG_B2A.cuda()
    netD_A.cuda()
    netD_B.cuda()

netG_A2B.apply(weights_init_normal)
netG_B2A.apply(weights_init_normal)
netD_A.apply(weights_init_normal)
netD_B.apply(weights_init_normal)

# Lossess
criterion_GAN = torch.nn.MSELoss()
criterion_cycle = torch.nn.L1Loss()
criterion_identity = torch.nn.L1Loss()

# Optimizers & LR schedulers
optimizer_G = torch.optim.Adam(itertools.chain(netG_A2B.parameters(), netG_B2A.parameters()),
                                lr=args.lr, betas=(0.5, 0.999))
optimizer_D_A = torch.optim.Adam(netD_A.parameters(), lr=args.lr, betas=(0.5, 0.999))
optimizer_D_B = torch.optim.Adam(netD_B.parameters(), lr=args.lr, betas=(0.5, 0.999))

lr_scheduler_G = torch.optim.lr_scheduler.LambdaLR(optimizer_G, lr_lambda=LambdaLR(args.n_epochs, args.epoch, args.decay_epoch).step)
lr_scheduler_D_A = torch.optim.lr_scheduler.LambdaLR(optimizer_D_A, lr_lambda=LambdaLR(args.n_epochs, args.epoch, args.decay_epoch).step)
lr_scheduler_D_B = torch.optim.lr_scheduler.LambdaLR(optimizer_D_B, lr_lambda=LambdaLR(args.n_epochs, args.epoch, args.decay_epoch).step)

# Inputs & targets memory allocation
Tensor = torch.cuda.FloatTensor if args.cuda else torch.Tensor
input_A = Tensor(args.batch_size, args.input_nc, args.size, args.size)
input_B = Tensor(args.batch_size, args.output_nc, args.size, args.size)
target_real = Variable(Tensor(args.batch_size).fill_(1.0), requires_grad=False)
target_fake = Variable(Tensor(args.batch_size).fill_(0.0), requires_grad=False)

fake_A_buffer = ReplayBuffer()
fake_B_buffer = ReplayBuffer()

MEAN = (0.5, 0.5, 0.5)
STD = (0.5, 0.5, 0.5)
de_transform_ = transforms.Compose([transforms.Normalize(mean=(0, 0, 0), std=tuple(1/x for x in STD)),
                                    transforms.Normalize(mean=tuple(-x for x in MEAN), std=(1, 1, 1))])
# Dataset loader
transforms_ = [ transforms.RandomCrop(args.size), 
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(MEAN, STD)]
train_dataset = Animeset(args.dataroot, transforms_=transforms_)
dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.n_cpu)
print("Data Loaded====================>")

###################################

###### Training ######
for epoch in tqdm(range(args.epoch, args.n_epochs)):
    for i, batch in enumerate(dataloader):
        # Set model input
        real_A = Variable(input_A.copy_(batch['A']))
        real_B = Variable(input_B.copy_(batch['B']))
        out_test = de_transform_(real_B)[0]
        
        ###### Generators A2B and B2A ######
        optimizer_G.zero_grad()

        # Identity loss
        # G_A2B(B) should equal B if real B is fed
        same_B = netG_A2B(real_B)
        loss_identity_B = criterion_identity(same_B, real_B)*5.0
        # G_B2A(A) should equal A if real A is fed
        same_A = netG_B2A(real_A)
        loss_identity_A = criterion_identity(same_A, real_A)*5.0

        # GAN loss
        fake_B = netG_A2B(real_A)
        pred_fake = netD_B(fake_B)
        loss_GAN_A2B = criterion_GAN(pred_fake, target_real)

        fake_A = netG_B2A(real_B)
        pred_fake = netD_A(fake_A)
        loss_GAN_B2A = criterion_GAN(pred_fake, target_real)

        # Cycle loss
        recovered_A = netG_B2A(fake_B)
        loss_cycle_ABA = criterion_cycle(recovered_A, real_A)*10.0

        recovered_B = netG_A2B(fake_A)
        loss_cycle_BAB = criterion_cycle(recovered_B, real_B)*10.0

        # Total loss
        loss_G = loss_identity_A + loss_identity_B + loss_GAN_A2B + loss_GAN_B2A + loss_cycle_ABA + loss_cycle_BAB
        loss_G.backward()
        
        optimizer_G.step()
        ###################################

        ###### Discriminator A ######
        optimizer_D_A.zero_grad()

        # Real loss
        pred_real = netD_A(real_A)
        loss_D_real = criterion_GAN(pred_real, target_real)

        # Fake loss
        fake_A = fake_A_buffer.push_and_pop(fake_A)
        pred_fake = netD_A(fake_A.detach())
        loss_D_fake = criterion_GAN(pred_fake, target_fake)

        # Total loss
        loss_D_A = (loss_D_real + loss_D_fake)*0.5
        loss_D_A.backward()

        optimizer_D_A.step()
        ###################################

        ###### Discriminator B ######
        optimizer_D_B.zero_grad()

        # Real loss
        pred_real = netD_B(real_B)
        loss_D_real = criterion_GAN(pred_real, target_real)
        
        # Fake loss
        fake_B = fake_B_buffer.push_and_pop(fake_B)
        pred_fake = netD_B(fake_B.detach())
        loss_D_fake = criterion_GAN(pred_fake, target_fake)

        # Total loss
        loss_D_B = (loss_D_real + loss_D_fake)*0.5
        loss_D_B.backward()

        optimizer_D_B.step()

        ###################################

        wandb.log({'loss_G': loss_G, 'loss_G_identity': (loss_identity_A + loss_identity_B), 'loss_G_GAN': (loss_GAN_A2B + loss_GAN_B2A),
                    'loss_G_cycle': (loss_cycle_ABA + loss_cycle_BAB), 'loss_D': (loss_D_A + loss_D_B)})


    # Update learning rates
    lr_scheduler_G.step()
    lr_scheduler_D_A.step()
    lr_scheduler_D_B.step()

    if epoch % 5 == 0:
        real_A_img = save_image(de_transform_(real_A)[0], os.path.join('./output/imgs', str(epoch) + '_real_A.jpg'))
        real_B_img = save_image(de_transform_(real_B)[0], os.path.join('./output/imgs', str(epoch) + '_real_B.jpg'))
        fake_A_img = save_image(de_transform_(fake_A)[0], os.path.join('./output/imgs', str(epoch) + '_fake_A.jpg'))
        fake_B_img = save_image(de_transform_(fake_B)[0], os.path.join('./output/imgs', str(epoch) + '_fake_B.jpg'))

    # Save models checkpoints
    if epoch % 10 == 0:
        torch.save(netG_A2B.state_dict(), 'models/' + str(epoch) + 'netG_A2B.pth')
        torch.save(netG_B2A.state_dict(), 'models/' + str(epoch) + 'netG_B2A.pth')
        torch.save(netD_A.state_dict(), 'models/' + str(epoch) + 'netD_A.pth')
        torch.save(netD_B.state_dict(), 'models/' + str(epoch) + 'netD_B.pth')
###################################

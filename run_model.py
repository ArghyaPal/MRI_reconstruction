import argparse
import cmath
import logging
import math
import pathlib
import random
import shutil

import h5py
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.cuda as cuda
import torch.nn as nn
from skimage.measure import compare_ssim as ssim
from tensorboardX import SummaryWriter
from torch.autograd import Variable
from torch.nn import functional as F
from torch.utils.data import Dataset

import utils
from data import transforms
from anet_model import AnetModel
from args import get_args

args = get_args()
train_loader, dev_loader = utils.create_data_loaders(args)

# ### Custom dataset class

def build_model(args):
    model = AnetModel(
        in_chans=2,
        out_chans=2,
        chans=args.num_chans,
        num_pool_layers=args.num_pools,
        drop_prob=args.drop_prob
    ).to(args.device)
    return model

# def build_optim(args, params):
#     optimizer = torch.optim.RMSprop(params, args.learning_rate, weight_decay=args.weight_decay)
#     return optimizer

# def load_model(checkpoint_file):
#     checkpoint = torch.load(checkpoint_file)
#     args = checkpoint['args']
#     model = build_model(args)
#     model.load_state_dict(checkpoint['model'])

#     optimizer = build_optim(args, model.parameters())
#     optimizer.load_state_dict(checkpoint['optimizer'])
#     return checkpoint, model, optimizer

# ### Image normalized

loss_func = nn.MSELoss()
best_val_loss = 1e9
writer = SummaryWriter(log_dir=args.exp_dir+'/summary')
valid_loss=[]
train_loss=[]
print('Total number of epochs:', args.epoch)
print('Total number of training iterations: ',len(train_loader))
print('Total number of validation iterations: ',len(dev_loader))

if args.resume:
    checkpoint, model, optimizer = utils.load_model(args.exp_dir + "/best_model.pt", build_model(args))
    best_dev_loss = checkpoint['best_dev_loss']
    start_epoch = checkpoint['epoch']
    if checkpoint['state']=='train':
        train = False
    del checkpoint
else:
    model = build_model(args)
    # if args.data_parallel:
        # model = torch.nn.DataParallel(model)
    optimizer = utils.build_optim(args, model.parameters())
    best_dev_loss = 1e9
    start_epoch = 0
    train = True

print(model)

# if args.resume:
#     checkpoint, model optimizer = load_model(args.checkpoint)
#     #best_val_loss = checkpoint['best_val_loss']
#     start_epoch = checkpoint['epoch']
#     if checkpoint['state']=='train':
#         train = False
#     del checkpoint
# else:
#     encoder = Encoder().cuda()
#     decoder = Decoder().cuda()
#     parameters = list(encoder.parameters())+ list(decoder.parameters())
#     optimizer = torch.optim.Adam(parameters, lr=args.learning_rate)
#     start_epoch = 0
#     train = True

for i in range(start_epoch, args.epoch):
    print("Epoch: ",i)
    global_step = i * len(train_loader) 
    ##########################################TRAINING PHASE######################################################
    if train:
        print("Training Phase")
        total_loss_kspace = total_loss_image = 0.0
        model.train()
        for j, data in enumerate(train_loader):
            original_kspace, masked_kspace, mask, target, fname, slice_index = data

            # normalizing the kspace
            nmasked_kspace, mean, std = utils.standardize(masked_kspace)
            noriginal_kspace, mean, std = utils.standardize(original_kspace, mean, std)

            # transforming the input according to dimension and type 
            noriginal_kspace, nmasked_kspace = utils.transformshape(noriginal_kspace), utils.transformshape(nmasked_kspace)

            nmasked_kspace = Variable(nmasked_kspace).to(args.device)
            noriginal_kspace = Variable(noriginal_kspace).to(args.device)
            
            # forward pass
            outputkspace = model(nmasked_kspace)

            # finding the kspace loss
            loss_kspace = loss_func(outputkspace, noriginal_kspace)
            loss_image = loss_func(utils.kspaceto2dimage(utils.transformback(outputkspace)), utils.kspaceto2dimage(utils.transformback(noriginal_kspace)))

            # setting up all the gradients to zero
            optimizer.zero_grad()

            #backward pass
            (loss_kspace + loss_image).backward()
            optimizer.step()

            total_loss_kspace += loss_kspace.data.item()
            total_loss_image += loss_image.data.item()

            if j % 100 == 0:
                avg_loss_kspace, avg_loss_image = total_loss_kspace/(j + 1), total_loss_image/(j + 1)
                print(j+1, ': AVG TRAINING LOSS: Kspace:', avg_loss_kspace, 'Image', avg_loss_image, 'ITR LOSS: Kspace', loss_kspace.data.item(), 'Image', loss_image.data.item())

                if j % 500 == 0:
                    utils.compareimageoutput(original_kspace, masked_kspace, outputkspace, mask, writer, global_step + j + 1, 0)

            writer.add_scalar('TrainKspaceLoss', loss_kspace.data.item(), global_step + j+1)
            writer.add_scalar('TrainImageLoss', loss_image.data.item(), global_step + j+1)
        utils.save_model(args, args.exp_dir, i+1 , model, optimizer, best_val_loss, False, 'train')    
        # train_loss.append(total_loss_kspace/len(train_loader))
    train = True
    
    ################################VALIDATION#######################################################
    print("Validation Phase")
    # validation loss
    total_val_loss = 0.0
    model.eval()
    for j,data in enumerate(dev_loader):
        original_kspace, masked_kspace, mask, target, fname, slice_index = data

        # normalizing the kspace
        nmasked_kspace, mean, std = utils.standardize(masked_kspace)
        noriginal_kspace, mean, std = utils.standardize(original_kspace, mean, std)

        # transforming the input according to dimension and type 
        noriginal_kspace, nmasked_kspace = utils.transformshape(noriginal_kspace), utils.transformshape(nmasked_kspace)

        nmasked_kspace = Variable(nmasked_kspace).to(args.device)
        noriginal_kspace = Variable(noriginal_kspace).to(args.device)
        
        # forward pass
        outputkspace = model(nmasked_kspace)
        
        # finding the kspace loss
        loss_kspace = loss_func(outputkspace, noriginal_kspace)
        loss_image = loss_func(utils.kspaceto2dimage(utils.transformback(outputkspace)), utils.kspaceto2dimage(utils.transformback(noriginal_kspace)))

        loss_itr = loss_kspace.data.item() + loss_image.data.item()
        
        total_val_loss += loss_itr
        
        if j % 100 == 0:
            avg_loss = total_val_loss/(j+1)
            print(j+1, ': AVG VALIDATION LOSS:', avg_loss, 'ITR LOSS:', loss_itr)
            if j % 200 == 0:
                utils.compareimageoutput(original_kspace,masked_kspace,outputkspace,mask,writer,global_step + j+1, 0)
        
        writer.add_scalar('ValidationLoss', loss_itr, global_step + j+1)
        
    valid_loss.append(total_val_loss / len(dev_loader))
    
    print('saving')
    is_new_best = valid_loss[-1] < best_val_loss
    best_val_loss = min(best_val_loss, valid_loss[-1])
    print("best val loss :",best_val_loss)
    utils.save_model(args, args.exp_dir, i+1, model, optimizer, best_val_loss, is_new_best, 'valid')    
writer.close()

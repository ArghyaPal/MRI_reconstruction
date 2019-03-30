import numpy as np
import cmath
import math
import shutil
import torch
import torch.cuda as cuda
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.autograd import Variable
from torch.nn import functional as F
from skimage.measure import compare_ssim as ssim
from sklearn import preprocessing
import matplotlib.pyplot as plt

def cartesianToPolar(input):
    polar = None
    if input.dtype != 'complex64':
        magnitude = input.norm(dim=-1)
        phase = torch.atan(torch.div(input[...,0], input[...,1]))
        phase[torch.isnan(phase)] = 0.0
        polar = np.stack([magnitude, phase], axis=-1)
    else:
        magnitude = np.vectorize(np.linalg.norm)
        phase = np.vectorize(np.angle)        
        polar = np.stack([magnitude(input), phase(input)], axis=-1)
    return torch.Tensor(polar)

def polarToCartesian(input):
    output = np.zeros(input.shape[:-1], dtype=np.complex64)
    it = np.nditer(output, flags=['multi_index'])

    while not it.finished:
        output[it.multi_index] = cmath.rect(input[it.multi_index][0], input[it.multi_index][1])
        temp = it.iternext()
    
    return torch.Tensor(np.stack([output.real, output.imag], axis = -1))



# ### Setting up the arguments

class Arguments:
    def __init__(self,batch_size,data_path,center_fractions,accelerations,challenge,sample_rate,resolution,resume,learning_rate,epoch,reluslope,exp_dir,checkpoint):
        self.batch_size = batch_size
        self.data_path = data_path
        self.center_fractions = center_fractions
        self.accelerations = accelerations
        self.challenge = challenge
        self.sample_rate = sample_rate
        self.resolution = resolution
        self.resume = resume
        self.learning_rate = learning_rate
        self.epoch = epoch
        self.reluslope = reluslope
        self.exp_dir = exp_dir
        self.checkpoint = checkpoint
        
# ### Custom dataset class

import pathlib
import random

import h5py
from torch.utils.data import Dataset


class SliceData(Dataset):
    """
    A PyTorch Dataset that provides access to MR image slices.
    """

    def __init__(self, root, transform, challenge, sample_rate=1, non_zero_ratio = 0.0, limit = -1):
        """
        Args:
            root (pathlib.Path): Path to the dataset.
            transform (callable): A callable object that pre-processes the raw data into
                appropriate form. The transform function should take 'kspace', 'target',
                'attributes', 'filename', and 'slice' as inputs. 'target' may be null
                for test data.
            challenge (str): "singlecoil" or "multicoil" depending on which challenge to use.
            sample_rate (float, optional): A float between 0 and 1. This controls what fraction
                of the volumes should be loaded.
        """
        if challenge not in ('singlecoil', 'multicoil'):
            raise ValueError('challenge should be either "singlecoil" or "multicoil"')

        self.transform = transform
        self.recons_key = 'reconstruction_esc' if challenge == 'singlecoil' else 'reconstruction_rss'
        self.challenge = challenge
        self.non_zero_ratio = non_zero_ratio
        self.examples = []

        files = list(pathlib.Path(root).iterdir())

        if sample_rate < 1:
            random.shuffle(files)
            num_files = round(len(files) * sample_rate)
            files = files[:num_files]
        
        for fname in sorted(files):
            f = h5py.File(fname, 'r')
            kspace = f['kspace']
            num_slices = kspace.shape[0]

            if non_zero_ratio > 0.0:
                slices = f[self.recons_key]
                num_pixels = np.prod(slices[0].shape)
                for slice in range(num_slices):
                    if non_zero_condition(slices[slice], num_pixels, non_zero_ratio):
                        self.examples += [(fname, slice)]
                    else:
                        print("Ignoring index: ", slice, " of file: ", fname)
            else:
                self.examples += [(fname, slice) for slice in range(num_slices)]
            
            if limit > 0:
                self.examples = self.examples[0:limit]


    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):
        fname, slice = self.examples[i]
        with h5py.File(fname, 'r') as data:
            kspace = data['kspace'][slice]
            target = data[self.recons_key][slice] if self.recons_key in data else None

            return self.transform(kspace, target, self.challenge, fname.name, slice)


# ### Data Transform (return original and masked kspace)

from data import transforms
class DataTransform:
    """
    Data Transformer for training DAE.
    """

    def __init__(self, mask_func, resolution, reduce, polar, use_seed=True):
        self.mask_func = mask_func
        self.use_seed = use_seed
        self.resolution = resolution
        self.reduce = reduce
        self.polar = polar

    def __call__(self, kspace, target, challenge, fname, slice_index):
        original_kspace = transforms.to_tensor(kspace)
        
        if self.reduce:
            original_kspace = reducedimension(original_kspace, self.resolution)
        
        # Apply mask
        seed = None if not self.use_seed else tuple(map(ord, fname))
        masked_kspace, mask = transforms.apply_mask(original_kspace, self.mask_func, seed)

        # Inverse Fourier Transform to get zero filled solution
        image = transforms.ifft2(masked_kspace)
        # Crop input image
        image = transforms.complex_center_crop(image, (self.resolution, self.resolution))
        # Absolute value
        image = transforms.complex_abs(image)
        # Apply Root-Sum-of-Squares if multicoil data
        if challenge == 'multicoil':
            image = transforms.root_sum_of_squares(image)
        # Normalize input
        image, mean, std = transforms.normalize_instance(image, eps=1e-11)

        target = transforms.to_tensor(target)
        # Normalize target
        target = transforms.normalize(target, mean, std, eps=1e-11)
        target = target.clamp(-6, 6)

        if self.polar:
            original_kspace = cartesianToPolar(original_kspace)
            masked_kspace = cartesianToPolar(masked_kspace)

        return original_kspace, masked_kspace, mask, target, fname, slice_index


# ### Creating data loaders

from common.subsample import MaskFunc
from torch.utils.data import DataLoader
def create_datasets(args, limit = -1):
    train_mask = MaskFunc(args.center_fractions, args.accelerations)
    dev_mask = MaskFunc(args.center_fractions, args.accelerations)

    train_data = SliceData(
        root= args.data_path + '/singlecoil_train',
        transform=DataTransform(train_mask, args.resolution, args.reduce, args.polar),
        sample_rate=args.sample_rate,
        challenge=args.challenge,
        non_zero_ratio = args.non_zero_ratio,
        limit = limit
    )
    dev_data = SliceData(
        root= args.data_path + '/singlecoil_val',
        transform=DataTransform(dev_mask, args.resolution, args.reduce, args.polar, use_seed=True),
        sample_rate=args.sample_rate,
        challenge=args.challenge,
        non_zero_ratio = args.non_zero_ratio,
        limit = limit
    )
    return dev_data, train_data


def create_data_loaders(args, if_shuffle = True, limit = -1):
    dev_data, train_data = create_datasets(args, limit = limit)
    
    train_loader = DataLoader(
        dataset=train_data,
        batch_size=args.batch_size,
        shuffle=if_shuffle, 
        pin_memory=True,
        num_workers = 4,
    )
    dev_loader = DataLoader(
        dataset=dev_data,
        batch_size=args.batch_size,
        pin_memory=True,
        num_workers = 4,
    )
    return train_loader, dev_loader

# #### Some utility function

import matplotlib.pyplot as plt

def non_zero_condition(image, num_pixels, non_zero_ratio):
    if non_zero_ratio > 0.0:
        non_zero_percentage = np.count_nonzero(image)/num_pixels
    else:
        non_zero_percentage = 1.0

    if (non_zero_percentage > non_zero_ratio):
        return True
    else:
        return False


def reducedimension(kspace, resolution):
    image = croppedimage(kspace, resolution)
    kspace = transforms.fft2(image)
    return kspace
    
def croppedimage(kspace, resolution):
    image = transforms.ifft2(kspace)
    image = transforms.complex_center_crop(image, (resolution, resolution))
    return image

def kspaceto2dimage(kspace, polar, cropping = False, resolution = None):
    if polar:
        kspace = polarToCartesian(kspace)

    if cropping:
        if not resolution:
            raise Exception("If cropping = True, pass the value for resolution for the function: kspaceto2dimage")
        image = croppedimage(kspace, resolution)
    else:
        image = transforms.ifft2(kspace)
    # Absolute value
    image = transforms.complex_abs(image)
    # Normalize input
    image, mean, std = transforms.normalize_instance(image, eps=1e-11)
    image = image.clamp(-6, 6)

    return image

def plotimage(image):
    plt.imshow(np.array(image))
    plt.show()
    
def transformshape(kspace):
    s = kspace.shape
    kspace = torch.reshape(kspace, (s[0],s[3],s[1],s[2]))
    return kspace

def transformback(kspace):
    s = kspace.shape
    kspace = torch.reshape(kspace, (s[0],s[2],s[3],s[1]))
    return kspace

def mse(imageA, imageB):
    err = np.sum((imageA.astype("float") - imageB.astype("float")) ** 2)
    err /= float(imageA.shape[0] * imageA.shape[1])
    return err

def compare_images(imageA, imageB,imageC,writer,iteration):
    mb = mse(imageA, imageB)
    sb = ssim(imageA, imageB)
    mc = mse(imageA, imageC)
    sc = ssim(imageA, imageC)

    fig = plt.figure()
    plt.suptitle("Target vs network MSE: %.2f, SSIM: %.2f" % (mb, sb)+" Target vs Zeroimputed MSE: %.2f, SSIM: %.2f" % (mc, sc))
     
    ax = fig.add_subplot(2, 2, 1)
    plt.imshow(imageA)
    plt.axis("off")

    ax = fig.add_subplot(2, 2, 2)
    plt.imshow(imageB)
    plt.axis("off")
    
    ax = fig.add_subplot(2, 2, 3)
    plt.imshow(imageA)
    plt.axis("off")

    ax = fig.add_subplot(2, 2, 4)
    plt.imshow(imageC)
    plt.axis("off")
    
    writer.add_figure('Comparision', fig, global_step = iteration)    

def compareimageoutput(original_kspace, masked_kspace, outputkspace, mask, writer, iteration, index, polar):
    assert original_kspace.size(-1) == 2
    assert masked_kspace.size(-1) == 2
    assert outputkspace.size(-3) == 2

    unmask = np.where(mask==1.0, 0.0, 1.0)
    unmask = transforms.to_tensor(unmask)
    unmask = unmask.float()
    output = transformback(outputkspace.data.cpu())
    output = output * unmask
    output = output + masked_kspace.data.cpu()
    imageA = np.array(kspaceto2dimage(original_kspace.data.cpu(), polar))[index]
    imageB = np.array(kspaceto2dimage(output, polar))[index]
    imageC = np.array(kspaceto2dimage(masked_kspace.data.cpu(), polar))[index]
    compare_images(imageA,imageB,imageC,writer,iteration)

def unitize(data, divisor = None):
    assert data.shape[-1] == 2
    norms = np.linalg.norm(data, axis = -1)

    if divisor is None:
        divisor = np.max(norms, axis = (1, 2))
        divisor = torch.stack(
                [torch.Tensor(divisor), torch.Tensor(divisor)], dim=-1).reshape(
                    shape=data.shape[:-3] + (1,1,2)
                )

    return torch.div(data, divisor), divisor
"""     a = np.array(data[0,:,:,0])**2 + np.array(data[0,:,:,1])**2
    if divisor is None:
        divisor = math.sqrt(a.max())
    data = data/divisor
    return data,divisor
 """

def standardize(data, mean = None, std = None):
    assert data.shape[-1] == 2

    if mean is None and std is None:
        mean = torch.Tensor(np.mean(np.array(data), axis=(-3,-2)).reshape(data.shape[:-3] + (1,1,2)))
        std = torch.Tensor(np.std(np.array(data), axis=(-3,-2)).reshape(data.shape[:-3] + (1,1,2)))

    answer = (data - mean)/std
    answer[torch.isnan(answer)] = 0.0
    return answer, mean, std

def destandardize(data, mean, std):
    return data * std + mean

def imagenormalize(data, divisor=None):
    """kspace generated by normalizing image space"""
    #getting image from masked data
    image = transforms.ifft2(data)
    #normalizing the image
    nimage, divisor = normalize(image, divisor)
    #getting kspace data from normalized image
    data = transforms.ifftshift(image, dim=(-3, -2))
    data = torch.fft(data, 2)
    data = transforms.fftshift(data, dim=(-3, -2)) 
    return data,divisor

def save_model(args, exp_dir, epoch, model, optimizer, best_dev_loss, is_new_best, state):
    torch.save(
        {
            'epoch': epoch,
            'args': args,
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'best_dev_loss': best_dev_loss,
            'exp_dir': exp_dir,
            'state' : state
        },
        f = exp_dir + "/model.pt"
    )
    if is_new_best:
        shutil.copyfile(exp_dir + '/model.pt', exp_dir + '/best_model.pt')

def build_optim(args, params):
    optimizer = torch.optim.RMSprop(params, args.learning_rate, weight_decay=args.weight_decay)
    return optimizer


def load_model(checkpoint_file, model):
    checkpoint = torch.load(checkpoint_file)
    args = checkpoint['args']
    if args.data_parallel:
        model = torch.nn.DataParallel(model)
    model.load_state_dict(checkpoint['model'])

    optimizer = build_optim(args, model.parameters())
    optimizer.load_state_dict(checkpoint['optimizer'])
    return checkpoint, model, optimizer

def save_tensors(tensors_dict, out_dir, fname):
    out_dir.mkdir(exist_ok=True)
    with h5py.File(out_dir / fname, 'w') as f:
        for tensor_name in tensors_dict:
            tensors_dict[tensor_name]= [t.numpy() for t in tensors_dict[tensor_name]]
            f.create_dataset(tensor_name, data=tensors_dict[tensor_name])

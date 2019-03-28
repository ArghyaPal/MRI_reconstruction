import torch
import numpy as np 
import utils
from args import get_args
from data.transforms import complex_abs
import matplotlib.pyplot as plt 

args = get_args()

train_loader, dev_loader = utils.create_data_loaders(args, if_shuffle = False)

prev_file = ""
tensor_dict = {'blurred_images': [], 'target_images': []}
print("All the outputs will be stored in the folder: ", args.out_dir)

for i, data in enumerate(train_loader):
    original_kspace, masked_kspace, mask, target, fname, slice_index = data
    fname = fname[0]
    blurred_image = utils.kspaceto2dimage(masked_kspace, cropping=True, resolution = args.resolution)

    if prev_file != fname and prev_file != "":
        utils.save_tensors(tensor_dict, args.out_dir, prev_file)
        tensor_dict['blurred_images'] = [blurred_image.squeeze()]
        tensor_dict['target_images'] = [target.squeeze()]
        prev_file = fname
    elif prev_file == fname:
        tensor_dict['blurred_images'].append(blurred_image.squeeze())
        tensor_dict['target_images'].append(target.squeeze())
    elif prev_file == "":
        prev_file = fname
    else:
        print("Unknown condition achieved!")

print("All the outputs are successfully stored in the folder: ", args.out_dir)  



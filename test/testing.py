# mbt2018_mean not used, mbt2018 used wo. eval()
# youtubeVIS

#!/usr/bin/env python
# coding: utf-8

# # 1. Main training and validation code

# In[25]:


import torch
import numpy as np
import os
import sys
import warnings

warnings.filterwarnings('ignore')
import imageio
from compressai.zoo import mbt2018_mean

try:
    import wandb
    wandb_exist = True
except ImportError:
    wandb_exist = False
    
import argparse
import logging
import time
import random
from torch import optim

from torch.utils.data import DataLoader, RandomSampler

model_path = os.path.abspath('..')
sys.path.insert(1, model_path)

from p_model import p_m
# from b_model import b_m
from utils import float_to_uint8, MSE, PSNR, calculate_distortion_loss
from utils import TestDataset
from utils import image_compress, save_model
from utils import load_model, build_info, update_train_info
from utils import update_val_info, update_best_val_info, zero_train_info

torch.backends.cudnn.benchmark = True

# ### Compression Functions

# In[ ]:


def p_frame_compression(x_previous, x_current, model, height, width):
    """P-frame compression function"""
    dec_p, _, size_p, _ = model(x_previous, x_current, train=False)

    uint8_real_p = float_to_uint8(x_current[0, :, :height, :width].cpu().numpy())
    uint8_dec_out_p = float_to_uint8(dec_p[0, :, :height, :width].cpu().detach().numpy())
    
    psnr = PSNR(MSE(uint8_dec_out_p.astype(np.float64), uint8_real_p.astype(np.float64)), data_range=255)
    pixels = uint8_real_p.shape[0] * uint8_real_p.shape[1]
    
    return dec_p, psnr, size_p, pixels
    
    
# In[ ]:


def b_frame_compression(x_previous, x_current, x_future, model, height, width):
    """B-frame compression function"""
    dec_b, size_b = model(x_previous, x_current, x_future)

    uint8_real_b = float_to_uint8(x_current[0, :, :height, :width].cpu().numpy())
    uint8_dec_out_b = float_to_uint8(dec_b[0, :, :height, :width].cpu().detach().numpy())
    
    psnr = PSNR(MSE(uint8_dec_out_b.astype(np.float64), uint8_real_b.astype(np.float64)), data_range=255)
    pixels = uint8_real_b.shape[0] * uint8_real_b.shape[1]
    
    return dec_b, psnr, size_b, pixels


# In[ ]:


def i_frame_compression(x, i_model, height, width):
    """I-frame compression function"""
    dec_i, size_i = image_compress(x, i_model)
    uint8_dec_out_i = float_to_uint8(dec_i[0, :, :height, :width].cpu().detach().numpy())
    uint8_real_i = float_to_uint8(x[0, :, :height, :width].cpu().numpy())

    psnr = PSNR(MSE(uint8_dec_out_i.astype(np.float64), uint8_real_i.astype(np.float64)), data_range=255)
    pixels = uint8_real_i.shape[0] * uint8_real_i.shape[1]
    
    return dec_i, psnr, size_i, pixels


# ### Test Function

# In[ ]:


def test(i_model, p_model, b_model, device, args):
    with torch.no_grad():
        
        average_psnr_dict = {}
        average_bpp_dict =  {}
        average_bpp = 0
        average_psnr = 0
        total_pixels = 0
        total_frames = 0

        folder_names = ["beauty", "bosphorus", "honeybee", "jockey", "ready", "shake", "yatch"]
        # folder_names = ["shake"]
        
        for video_num in range(len(folder_names)):
            # GOP size is 13 as default but we take 14 images for backward coding
            test_dataset = TestDataset(args.test_path, video_num, gop_size=args.gop_size, 
                                       i_p_interval=args.i_p_interval, skip_frames=args.skip_frames, 
                                       test_size=None)
            
            test_loader = DataLoader(test_dataset, batch_size=None, shuffle=False, num_workers=args.workers)
            
            this_bpp = 0
            this_psnr = 0
            this_pixels = 0
            this_frames = 0
            
            # Frame order for decoding
            coding_order = [1, 2, 3, 4]
            decoding_parents = {1: 0, 2: 1, 3: 2, 4: 3}
            decoded_frames = {}
            
            # Loading videos in batches of form I-B-B-B-B-B-B-B-P
            for video in test_loader:
                # Denoting whether the first frame of the batch is an i-frame
                has_i_frame = video["has_i_frame"]
                batch_first_frame = video["first_frame"]
                batch_last_frame = video["last_frame"]
                
                video_images = video["frames"].to(device).float()
                # os.makedirs("uvg/" + folder_names[folder_number], exist_ok=True)
                video_total_images = video_images.shape[0]
                
                logging.info("Test has I frame " + str(has_i_frame) + ". Frames: " + 
                             batch_first_frame + "-" + batch_last_frame)
                
                # Code every ?th (for exanple 56th) frame using I-frame coding model, specified in
                # dataset function in utils.py
                # 0 is for initial batch with I-frame as first frame
                # 1 is for batch with last frame coded as I-frame
                # 2 is for batch with no I-frame and last frame coded as P-frame
                if has_i_frame == 0:
                    x1 = video_images[0]
                    b, c, h, w = x1.shape
        
                    dec_i, psnr_i, size_i, pixels_i = i_frame_compression(x1, i_model, height=h, 
                                                                          width=w)
                    decoded_frames[0] = dec_i
                    
                    this_psnr += psnr_i
                    this_bpp += size_i
                    this_pixels += pixels_i
                    this_frames += 1
                    
                    average_psnr += psnr_i
                    average_bpp += size_i
                    total_pixels += pixels_i
                    total_frames += 1
                    
                    print("I frame ", psnr_i, size_i / (h*w))
                
                for index, i in enumerate(coding_order):
                    x = video_images[i]
                    # Compress the frame using p-frame coding or b-frame coding
                    # The first frame here is guaranteed to be p-frame, afterwards we turn off the switch
                    if (has_i_frame == 1) and (i==coding_order[-1]):
                        dec_last, psnr, size, pixels = i_frame_compression(x, i_model, height=h, width=w)
                        
                        decoded_frames[i] = dec_last
                        
                        print("I frame", psnr, size / (h*w))
                        
                    elif (has_i_frame in [0, 2]) and (i==coding_order[-1]):
                        
                        dec_last, psnr, size, pixels = p_frame_compression(decoded_frames[decoding_parents[i]], x, p_model, height=h, width=w)
                        # dec_last, psnr, size, pixels = i_frame_compression(x, i_model, height=h, width=w)
                        
                        decoded_frames[i] = dec_last
                        
                        print("P frame", psnr, size / (h*w))
                        
                    else:
#                        dec_b, psnr, size, pixels = b_frame_compression(decoded_frames[decoding_parents[i][0]], 
#                                                                        x, 
#                                                                        decoded_frames[decoding_parents[i][1]], 
#                                                                        b_model,
#                                                                        height=h, width=w)                                         
#                        decoded_frames[i] = dec_b
                        dec, psnr, size, pixels = p_frame_compression(decoded_frames[decoding_parents[i]], x, p_model, height=h, width=w)
                        decoded_frames[i] = dec 
                        print("P frame", psnr, size / (h*w))
                        
                    
                    this_psnr += psnr
                    this_bpp += size
                    this_pixels += pixels
                    this_frames += 1
                    
                    average_psnr += psnr
                    average_bpp += size
                    total_pixels += pixels
                    total_frames += 1
                
                decoded_frames = {0: dec_last}

            average_psnr_dict[folder_names[video_num]] = this_psnr / this_frames
            average_bpp_dict[folder_names[video_num]] = this_bpp / this_pixels
            
    average_psnr /= total_frames
    average_bpp /= total_pixels

    return average_psnr.item(), average_bpp.item(), average_psnr_dict, average_bpp_dict


# ### Main Function

# In[ ]:
# We just train the b-coding model
# P-frame coding model is freezed after a complete training

# Argument parser
parser = argparse.ArgumentParser()

# Hyperparameters, paths and settings are given
# prior the training and validation
parser.add_argument("--test_path", type=str, default="/userfiles/ecetin17/full_test/")
parser.add_argument("--gop_size", type=int, default=56)                                         # test gop sizes
parser.add_argument("--skip_frames", type=int, default=4)                                       # how many frames to skip in test time
parser.add_argument("--i_p_interval", type=int, default=4)                                      # how many frames between i and p frames
parser.add_argument("--device", type=str, default="cuda")                                       # device "cuda" or "cpu"
parser.add_argument("--workers", type=int, default=4)                                           # number of workers
parser.add_argument("--seeds", type=tuple, default=(1, 2, 3))                                   # seeds for randomness

parser.add_argument("--alpha", type=int, default=3141)                                          # alpha for rate-distortion trade-off
parser.add_argument("--compressor_q", type=int, default=8)                                      # I-frame compressor quality factor

parser.add_argument("--p_pretrained", type=str, default="../DVC3141.pth")                       # save model to folder (670k)
parser.add_argument("--wandb", type=bool, default=False)                                         # Store results in wandb
parser.add_argument("--log_results", type=bool, default=True)                                   # Store results in log

args = parser.parse_args()

args.project_name = "RLVC_test"

args.test_name = "I" + str(args.compressor_q) + "P" + str(args.alpha) + "_4frames" + "_gop" + str(args.gop_size)

logging.basicConfig(filename=args.test_name + "_full.log", level=logging.INFO)


# In[ ]:


def main(args):

    device = torch.device(args.device)
    
    # args.i_p_interval = args.gop_size // (13 + 1)
    # args.i_p_interval = 1
    
    # Group name stays the same, change the name for different (or same) trials
    if args.wandb and wandb_exist:
        wandb.init(project=args.project_name, name=args.test_name, config=vars(args))
    
    p_model = p_m.P_Model().to(device).float()
    # b_model = b_m.B_Model().to(device).float()
    pretrained_dict = torch.load(args.p_pretrained)
    
    p_model = load_model(p_model, pretrained_dict)
    # b_model = load_model(b_model, pretrained_dict)
    
    i_model = mbt2018_mean(quality=args.compressor_q, metric="mse", 
                           pretrained=True).to(device).float()
    
    i_model.eval()
    p_model.eval()
    # b_model.eval()
    b_model = None

    time_start = time.perf_counter()
    
    avg_psnr, avg_bpp, avg_psnr_dict, avg_bpp_dict = test(i_model, p_model, b_model, device, args)

    time_end = time.perf_counter()
    duration = time_end - time_start
    
    if args.wandb and wandb_exist:
        rd_data = [[avg_bpp, avg_psnr]]
        
        # Match column names with chart axes
        rd_table = wandb.Table(data=rd_data, columns=["bpp", "PSNR"])
        
        # "bpp vs PSNR" sets the table name, to match the tables, add them in same named table!!
        # title changes the chart title, axes are "bpp" and "PSNR"
        wandb.log({"bpp vs PSNR": wandb.plot.scatter(rd_table, "bpp", "PSNR", title="RD Curve")})
        
        time_data = [[args.test_name, duration]]
        time_table = wandb.Table(data=time_data, columns = ["Model", "Time (sec)"])
        wandb.log({"Duration": wandb.plot.bar(time_table, "Model", "Time (sec)", title="Duration of Test")})
    
    if args.log_results:
        logging.info("PSNR log: %s", avg_psnr_dict)
        logging.info("bpp log: %s", avg_bpp_dict)
    
        logging.info("-------------------------------")
    
        logging.info("Average PSNR: " + str(avg_psnr))
        logging.info("Average bpp: " + str(avg_bpp))
        logging.info("Duration (sec): " + str(duration))


# In[ ]:


if __name__ == '__main__':
    main(args)


# In[ ]:





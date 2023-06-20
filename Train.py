import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
import matplotlib.pyplot as plt
import math
import argparse
import json
import os
import random
from Data import *
from Network import *

'''
This work is slightly inspired by 
https://colab.research.google.com/github/keras-team/keras-io/blob/master/examples/vision/ipynb/nerf.ipynb
'''
# To encode the position into higher dimension(corresponding Fourier feature tensors)
def positional_encoding(tensor, high_dim):
    ps = [tensor]
    for i in range(high_dim):
        ps.append(torch.sin((2.0**i) * tensor))
        ps.append(torch.cos((2.0**i) * tensor))
    ps = torch.concat(ps, axis =-1)
    return ps

# To compute origin point and direction vector of rays
def rays(device,height, width, focal_len, cam_pose):
  grid_x, grid_y = torch.meshgrid(torch.arange(width).to(device),torch.arange(height).to(device))
  grid_x, grid_y = grid_x.transpose(-1,-2),grid_y.transpose(-1,-2)
  directions=torch.stack([(grid_x - width * .5) / focal_len, -(grid_y - height * .5) / focal_len, -torch.ones_like(grid_x)], dim=-1)
  rays_dir=torch.sum(directions[..., None, :] * cam_pose[:3, :3], dim=-1)
  rays_ori=cam_pose[:3,-1].expand(rays_dir.shape)
  return rays_dir, rays_ori

# To get the sample points
def querypoints(device, ray_ori, ray_dir, near, far, sample, rand= False): 
    depth = torch.linspace(near,far,sample) 
    # Inject uniform noise into sample space to make the sampling continuous.
    if rand is True:
        noise_shape = list(ray_ori.shape[:-1])+[sample]
        noise = torch.rand(noise_shape)*((far-near)/sample)
        depth += noise
    depth = depth.to(device)
  # r = o + t*d  
    query_pt = torch.tensor(ray_ori[..., None, :] + ray_dir[..., None, :] * depth[..., :, None],dtype = torch.float32)
    return query_pt, depth

def mini_batch(inp_tensor, size: int):
    batch = [inp_tensor[i:i + size] for i in range(0, inp_tensor.shape[0], size)]
    return batch

def plot_loss(num_epoch, loss):
    plt.figure(figsize=(10, 4))
    plt.plot(num_epoch, loss)
    plt.title("Loss")
    plt.savefig("Loss.png")

def volumetric_rendering(depth, rays_ori, rgb, sigma):    
    e_10 = torch.tensor([1e10], dtype = rays_ori.dtype, device = rays_ori.device)
    # print((depth[...,:1].shape))
    e_10 = e_10.expand(depth[...,:1].shape)
    delta_i= depth[...,1:] - depth[...,:-1]
    adjacent_dist = torch.concat((delta_i, e_10), dim = -1)
    alpha = 1.0 - torch.exp(-1 * sigma * adjacent_dist)
    #transmittance
    wts = alpha * cumulative_product(1.0 - alpha + 1e-10)
    rgb_map = (wts[..., None] * rgb).sum(dim = -2)  
    return wts, rgb_map

def depth_mapping(wts,depth):
    w_depth = wts * depth
    depth_map = w_depth.sum(dim=-1)
    return depth_map

#  To calculate the cumulative product to calculate alpha
def cumulative_product(tensor) :
  product = torch.cumprod(tensor, dim=-1)
  product = torch.roll(product, 1, dims=-1)
  product[..., 0] = 1.0
  return product

# Pipeline 
def train(device, height, width, focal_len, cam_pose, near, far, sample, high_N, batch_size, model):
    rays_dir, rays_ori = rays(device,height, width, focal_len, cam_pose)
    query_pt, depth = querypoints(device,rays_ori,rays_dir, near, far, sample, rand=False)
    flat_query_pt = torch.Tensor(query_pt.reshape((-1,3)))
    gamma = positional_encoding(flat_query_pt, high_N)
    batch = mini_batch(gamma, batch_size)
    
    model_out = []
    for b in batch:
        model_out.append((model(b)))
    radiance = torch.cat(model_out, dim=0)
    unflatten = list(query_pt.shape[:-1]) + [4]
    radiance_field = torch.reshape(radiance, unflatten)
    sigma = torch.relu(radiance_field[...,3])   
    rgb = torch.sigmoid(radiance_field[...,:3])   
    wts, rgb_map = volumetric_rendering(depth, rays_ori, rgb, sigma)
    depth_map = depth_mapping(wts,depth)
    acc_map = torch.einsum('ijk->ij', wts)
    return rgb_map, depth_map, acc_map


def train_all(num_epoch, device, height, width, focal_len, all_cam_pose, all_images, near, far, sample, high_N, batch_size):
 
  # model = model_NeRF()
  model = model_tinyNeRF()
  model = model.to(device)
  optimizer = torch.optim.Adam(model.parameters(), 5e-3)

  torch.manual_seed(9458)
  random.seed(9458)

  Loss = []
  Epochs = []

  for i in range(num_epoch+1):
    img_idx = random.randint(0, all_images.shape[0]-1)
    img = all_images[img_idx].to(device)
    img=img.float()
    cam_pose = all_cam_pose[img_idx].to(device)

    rgb,_,_ = train(device, height, width, focal_len, cam_pose, near, far, sample, high_N, batch_size, model)    
    loss=torch.nn.functional.mse_loss(rgb, img)
    # print('loss',loss)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    

    if (i % 50 == 0):
      rgb,_,_ = train(device, height, width, focal_len, cam_pose, near, far, sample, high_N, batch_size, model)
      loss=torch.nn.functional.mse_loss(rgb, img)
      plt.imshow(rgb.detach().cpu().numpy())
      plt.title(f"Iteration {i}")
      plt.savefig("rgb.png")
      Loss.append(loss.item())
      Epochs.append(i+1)
      plot_loss(Epochs, Loss)
      print('i',i)
  print('DONEEE!!, Check me out!')


  SaveName =  './Checkpoint/' + 'model_' + str(i) + '.ckpt'
  torch.save({'epoch': Epochs, 'model_state_dict': model.state_dict(), 'optimizer_state_dict': optimizer.state_dict()}, SaveName)



def main():

  Parser = argparse.ArgumentParser()
  Parser.add_argument('--nEpochs', type=int, default=1500)
  Parser.add_argument('--samples', type=int, default=32)
  Parser.add_argument('--MiniBatchSize', type=int, default=4096)
  Parser.add_argument('--near', type=int, default=2)
  Parser.add_argument('--far',type=int, default=6)
  Parser.add_argument('--dim_encode',type=int, default=6)

  Args = Parser.parse_args()
  num_epoch = Args.nEpochs
  sample = Args.samples
  batch_size = Args.MiniBatchSize
  near = Args.near
  far = Args.far
  high_N = Args.dim_encode
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#   data = read_json('./transforms_train.json')
#   focal_len, all_cam_pose, all_images = full_data(device,data)
  focal_len, all_cam_pose, all_images = get_tiny_data(device)
  height, width = all_images.shape[1:3]
  print(height,'height')
  print(width,'width')

  train_all(num_epoch, device, height, width, focal_len, all_cam_pose, all_images, near, far, sample, high_N, batch_size)


if __name__ == "__main__":
  main()


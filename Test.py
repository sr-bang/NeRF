import torch
# import torchvision
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
import matplotlib.pyplot as plt
import math
import argparse
import imageio
from Data import *
from Network import *
from Train import *


'''
This work is slightly inspired by 
https://colab.research.google.com/github/keras-team/keras-io/blob/master/examples/vision/ipynb/nerf.ipynb
'''
# Translation matrix for movement in t
def get_translation(t):
    matrix = [
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 1, t],
        [0, 0, 0, 1],
    ]
    return torch.tensor(matrix, dtype=torch.float32)

# Rotation matrix for movement in phi
def get_rotation_phi(phi):
    matrix = [
        [1, 0, 0, 0],
        [0, np.cos(phi), -np.sin(phi), 0],
        [0, np.sin(phi), np.cos(phi), 0],
        [0, 0, 0, 1],
    ]
    return torch.tensor(matrix, dtype=torch.float32)

# Rotation matrix for movement in theta
def get_rotation_theta(theta):
    matrix = [
        [np.cos(theta), 0, -np.sin(theta), 0],
        [0, 1, 0, 0],
        [np.sin(theta), 0, np.cos(theta), 0],
        [0, 0, 0, 1],
    ]
    return torch.tensor(matrix, dtype=torch.float32)

# Camera to world matrix for the corresponding theta, phi and t
def pose_spherical(theta, phi, t):
    c2w = get_translation(t)
    c2w = get_rotation_phi(phi / 180.0 * np.pi) @ c2w
    c2w = get_rotation_theta(theta / 180.0 * np.pi) @ c2w
    c2w = torch.tensor([[-1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]], dtype=torch.float32) @ c2w
    return c2w


def Test(all_images, all_cam_pose, focal_len, height, width, high_N, near, far, batch_size, sample, device):
    
    # model = model_NeRF()
    model = model_tinyNeRF()
    CheckPoint = torch.load('./Checkpoint/model_1500.ckpt')
    model.load_state_dict(CheckPoint['model_state_dict'])
    model.to(torch.float32)
    model = model.to(device)
    model.eval()

    rgb_test,_,_ = train(device, height, width, focal_len, all_cam_pose, near, far, sample, high_N, batch_size, model)
    plt.imshow(rgb_test.detach().cpu().numpy())
    plt.title(f"Testing Image")
    plt.savefig("rgb_test.png")
    
    # Iterate over different theta value and generate scenes
    frames = []
    for theta in (np.linspace(0.0, 360.0, 120)):
        c2w = pose_spherical(theta,-30, 4.0)    
        transform = c2w.to(device)
        rgb_values,_,_ = train(device, height, width, focal_len, transform, near, far, sample, high_N, batch_size, model)
        frames.append((255*np.clip(rgb_values.detach().cpu().numpy(),0,1)).astype(np.uint8))
        # print('theta',theta)
    imageio.mimwrite("lego_gif.mp4", frames, fps=30, quality=7, macro_block_size=None)
    print("Video Out!")



def main():

    Parser = argparse.ArgumentParser()

    Parser.add_argument('--samples', type=int, default=32)
    Parser.add_argument('--Nc', type=int, default=32)
    Parser.add_argument('--MiniBatchSize', type=int, default=4096)
    Parser.add_argument('--near', type=int, default=2)
    Parser.add_argument('--far',type=int, default=6)
    Parser.add_argument('--dim_encode',type=int, default=6)

    Args = Parser.parse_args()
    sample = Args.samples
    batch_size = Args.MiniBatchSize
    near = Args.near
    far = Args.far
    high_N = Args.dim_encode
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    focal_len, all_cam_pose, all_images = get_tiny_data(device)
    # print(focal_len)
    img_test = all_images[23]
    pose_test = all_cam_pose[23]
    height, width = all_images.shape[1:3]
    # print(height,width)
    high_N = 6
    
    Test(img_test, pose_test, focal_len, height, width, high_N, near, far, batch_size, sample, device)


if __name__ == '__main__':
    main()
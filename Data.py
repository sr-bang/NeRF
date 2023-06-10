import torch
import numpy as np
import cv2
import math
import json
import os



def read_json(jsonPath):
	with open(jsonPath, "r") as fp:
		jsondata = json.load(fp)
	return jsondata

def get_para(index):
    if torch.is_tensor(index):
        index = index.tolist()

    data = read_json('./transforms_train.json')
    img_file = data["frames"][index]["file_path"] + ".png"
    img_name = os.path.join(img_file)
    image = cv2.imread(img_name)

    image = cv2.resize(image, (200,200), interpolation=cv2.INTER_AREA)
    transforms = data["frames"][index]["transform_matrix"]
    transforms = torch.tensor(transforms)
    camera_angle_x = data["camera_angle_x"]
    focal = 0.5*image.shape[0] / math.tan(0.5 * camera_angle_x)
    return {'focal' : focal, 'image':image, 'transforms':transforms}


def full_data(device,data):
    cam_pose= []
    images = []
    focal_len =[]
    total_len = len(data['frames'])
    for i in range(total_len):
        onepara = get_para(i)
        focal_len = onepara['focal']
        cam_pose.append(onepara['transforms'])
        images.append(torch.tensor(onepara['image']))
    return torch.tensor(focal_len).to(device), torch.stack(cam_pose).to(device), torch.stack(images).to(device)


def get_tiny_data(device):
    images = np.load('./tiny_nerf_data/images.npy')
    images = torch.tensor(images).to(device)
    cam_poses = torch.tensor(np.load('./tiny_nerf_data/poses.npy')).to(device)
    focal_len = torch.tensor(np.load('./tiny_nerf_data/focal.npy'))
    return focal_len, cam_poses, images



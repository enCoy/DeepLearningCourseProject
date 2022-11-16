from torch.utils.data import Dataset, DataLoader
import torch
import numpy as np
from utils.utils import scene_to_point_cloud, get_intrinsics
import pickle
import os
import random
from dataset_preprocess import get_data

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

# OLD CUSTOM DATALOADER LOADING DATA FROM LABEL.PKL FILES
class CustomDataLoader(Dataset):
    # returns
    def __init__(self, data_directory, apply_normalization=True):
        # read the files in this data directory
        self.data_directory = data_directory
        self.data_list = os.listdir(data_directory)
        self.apply_normalization = apply_normalization

        random.shuffle(self.data_list)

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        # load the files and return
        file = os.path.join(self.data_directory, self.data_list[idx])
        with open(os.path.join(file), 'rb') as handle:
            file = pickle.load(handle)
        if self.data_list[idx][0:6] =='camera':
            intrinsics = np.array([[577.5, 0, 319.5], [0, 577.5, 239.5], [0, 0, 1]])
        elif self.data_list[idx][0:4] =='real':
            intrinsics = np.array([[591.0125, 0, 322.525], [0, 590.16775, 244.11084], [0, 0, 1]])
        else:
            intrinsics = None
            print("This is neither REAL275 nor CAMERA!")
        # # find the length of data
        # length_data = len(file['x_depth'])
        # sampling_interval = length_data / self.sampling_size
        # sampling_idx = (np.arange(self.sampling_size) * sampling_interval).astype(int)
        # # use this as a mask
        # x_depth = np.expand_dims(file['x_depth'][sampling_idx], axis=1)  # sampling size x 1
        # y_depth = np.expand_dims(file['y_depth'][sampling_idx], axis=1) # sampling size x 1
        # z_depth = np.expand_dims(file['z_depth'][sampling_idx], axis=1) # sampling size x 1
        # rgb_data = file['rgb'][sampling_idx, :] # # sampling size x 3
        # concatenate data column wise
        depth = file['depth']
        point_cloud = scene_to_point_cloud(depth, intrinsics)
        rgb = file['rgb_colored']

        point_cloud = torch.tensor(point_cloud)
        rgb = torch.tensor(rgb)

        labels = file['bbox_3d'].astype(np.float32)
        if self.apply_normalization:
            # divide rgb by 255
            rgb = rgb /255
            # point_cloud = torch.reshape(self.pc_norm(point_cloud), (self.HEIGHT, self.WIDTH, self.CHANNELS))
            # apply channel wise min_max scaling to point cloud data
            r_max = torch.amax(point_cloud, dim=(0, 1))  # channel dimension will remain
            r_min = torch.amin(point_cloud, dim=(0, 1))  # channel dimension will remain
            t_min = torch.tensor([[-1,-1,-1]], dtype=torch.float64)
            t_max = torch.tensor([[+1, +1, +1]], dtype=torch.float64)
            point_cloud = ((point_cloud - r_min)/(r_max - r_min)) * (t_max - t_min) + t_min

        # features shape (self.sampling_size, num_features=6)
        # labels shape (3, 8) - 8 from total num of vertices, 3 from xyz
        return point_cloud, rgb, labels

class CustomDataLoaderV2(Dataset):
    # returns
    def __init__(self, data_directory, data_name='train',apply_normalization=True):
        # read the files in this data directory
        self.data_directory = data_directory
        self.data_list =  open(self.data_directory + '/' +data_name + "_list_all.txt", "r").read().split("\n")
        self.meta_list = open(self.data_directory + '/' +data_name + "_meta.txt", "r").read().split("\n")
        self.apply_normalization = apply_normalization

    def __len__(self):
        return len(self.data_list)
    #
    def __getitem__(self, idx):
        if idx >= len(self.data_list):
            raise StopIteration
        obj_path = self.data_list[idx]
        obj_meta = self.meta_list[idx]
        # print("here is obj path: ", obj_path)
        dataset_type = None
        if obj_path[0:4]=='Real':
            dataset_type = 'Real'
        elif obj_path[0:6]=='CAMERA':
            dataset_type = 'CAMERA'
        else:
            intrinsics = None
            print("This is neither REAL275 nor CAMERA!")
        intrinsics = get_intrinsics(dataset_type)
        # now we will return point cloud depth data, rgb color data, mask and label
        point_cloud, rgb, mask, bbox_coords = get_data(self.data_directory, obj_path,
                                                                                    obj_meta, dataset_type, intrinsics)
        if (point_cloud is None or rgb is None) or (mask is None or bbox_coords is None):
            del self.data_list[idx]
            del self.meta_list[idx]
            return self.__getitem__(idx)

        point_cloud = torch.tensor(point_cloud)
        rgb = torch.tensor(rgb)

        if self.apply_normalization:
            # divide rgb by 255
            rgb = rgb /255
            # point_cloud = torch.reshape(self.pc_norm(point_cloud), (self.HEIGHT, self.WIDTH, self.CHANNELS))
            # apply channel wise min_max scaling to point cloud data
            r_max = torch.amax(point_cloud, dim=(0, 1))  # channel dimension will remain
            r_min = torch.amin(point_cloud, dim=(0, 1))  # channel dimension will remain
            t_min = torch.tensor([[-1,-1,-1]], dtype=torch.float64)
            t_max = torch.tensor([[+1, +1, +1]], dtype=torch.float64)
            point_cloud = ((point_cloud - r_min)/(r_max - r_min)) * (t_max - t_min) + t_min

        # features shape (self.sampling_size, num_features=6)
        # labels shape (3, 8) - 8 from total num of vertices, 3 from xyz
        # point cloud shape: BatchSize x H x W x 3
        # rgb shape: BatchSize x H x W x 3
        # mask shape: BatchSize x H x W
        # bbox_coords shape: BatchSize x 3 x 8
        return point_cloud, rgb, mask, bbox_coords

        # # # find the length of data
        # # length_data = len(file['x_depth'])
        # # sampling_interval = length_data / self.sampling_size
        # # sampling_idx = (np.arange(self.sampling_size) * sampling_interval).astype(int)
        # # # use this as a mask
        # # x_depth = np.expand_dims(file['x_depth'][sampling_idx], axis=1)  # sampling size x 1
        # # y_depth = np.expand_dims(file['y_depth'][sampling_idx], axis=1) # sampling size x 1
        # # z_depth = np.expand_dims(file['z_depth'][sampling_idx], axis=1) # sampling size x 1
        # # rgb_data = file['rgb'][sampling_idx, :] # # sampling size x 3
        # # concatenate data column wise
        # depth = file['depth']
        # point_cloud = scene_to_point_cloud(depth, intrinsics)
        # rgb = file['rgb_colored']
        #


if __name__ == "__main__":
    user_dir = r'C:\Users\Cem Okan'  # change this on your computer
    data_dir = user_dir + r'\Dropbox (GaTech)\deep_learning_data'
    # sampling size does not work right now
    train_dataset = CustomDataLoaderV2(data_dir, data_name='train', apply_normalization=True)
    val_dataset = CustomDataLoaderV2(data_dir, data_name='val', apply_normalization=True)
    test_dataset = CustomDataLoaderV2(data_dir, data_name='test',apply_normalization=True)

    batch_size = 4
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=128, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=True)
    for (point_cloud, rgb, mask, bbox_coords) in val_loader:
        continue



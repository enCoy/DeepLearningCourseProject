from torch.utils.data import Dataset, DataLoader
import torch
import numpy as np
from utils.utils import scene_to_point_cloud
import pickle
import os
import random


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

# def calculate_feature_statistics(data_loader, feature_dim = 6):
#     general_sum = torch.zeros(feature_dim)
#     general_std_sum = torch.zeros(feature_dim)
#     counter = 0
#     num_batches = len(data_loader)
#
#     starting_time = time.time()
#     for x, y in data_loader:
#         # x is a tensor shaped: (batch_size ,sampling_size=900, num_feature=6)
#         # y is a tensor shaped: (batch_size ,3, 8)
#         # find the average x,y,z in the batch
#         average_sample_coord = torch.mean(x, 1) #  batch_size x num_feature
#         # sum over batch
#         general_sum += average_sample_coord.sum(axis=0)
#
#         general_std_sum += torch.std(average_sample_coord, 0)  # std of average sample coord across the batch
#         # counter holds the total number of examples
#         counter += average_sample_coord.size()[0]
#     average_batch_coord = general_sum / counter
#     std_batch_coord = general_std_sum / num_batches
#     finishing_time = time.time()
#     print(
#         f"One pass on training data for calculating feature statistics took {finishing_time - starting_time} seconds.")
#     return average_batch_coord, std_batch_coord

if __name__ == "__main__":
    user_dir = r'C:\Users\Cem Okan'  # change this on your computer
    data_dir = user_dir + r'\Dropbox (GaTech)\deep_learning_data\gts'
    train_data_dir = data_dir + r'\train_data'
    val_data_dir = data_dir + r'\val_data'
    test_data_dir = data_dir + r'\test_data'
    # sampling size does not work right now
    train_dataset = CustomDataLoader(train_data_dir, apply_normalization=True)
    val_dataset = CustomDataLoader(val_data_dir, apply_normalization=True)
    test_dataset = CustomDataLoader(test_data_dir, apply_normalization=True)

    batch_size = 4
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=128, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=True)
    # mean_stat, std_stat = calculate_feature_statistics(data_loader, feature_dim=6)
    # print("Feature mean statistics:", mean_stat)
    # print("Feature std statistics:", std_stat)
    # # apply feature statistics to data loader
    # dataset.apply_feature_stat([mean_stat, std_stat])

    # Loader x shape: # Batch_size x Height x (Width x 2) {RGB and coordinates concatenated} x 3
    # Loader y shape: # Batch_size x 3 x 8  {3 is for xyz dimensions, 8 is for vertices of the bounding box}
    for point_cloud, rgb, bbox in train_loader:
        print("point_cloud shape: ", point_cloud.shape)
        print("rgb shape: ", rgb.shape)
        print("bbox shape: ", bbox.shape)
        continue

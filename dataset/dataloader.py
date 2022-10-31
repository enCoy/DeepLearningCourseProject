from torch.utils.data import Dataset, DataLoader
import torch
import os
import numpy as np
from utils.utils import load_depth
import pickle
import cv2
from utils.utils import load_coord, load_colored, load_mask, load_label, load_depth
import os
import random

class CustomDataLoader(Dataset):
    # returns
    def __init__(self, data_directory, sampling_size=900):
        # read the files in this data directory
        self.data_directory = data_directory
        self.data_list = os.listdir(data_directory)
        self.sampling_size = sampling_size
        random.shuffle(self.data_list)

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        # load the files and return
        file = os.path.join(self.data_directory, self.data_list[idx])
        with open(os.path.join(file), 'rb') as handle:
            file = pickle.load(handle)

        # find the length of data
        length_data = len(file['x_depth'])
        sampling_interval = length_data / self.sampling_size
        sampling_idx = (np.arange(self.sampling_size) * sampling_interval).astype(int)

        # use this as a mask
        x_depth = np.expand_dims(file['x_depth'][sampling_idx], axis=1)  # sampling size x 1
        y_depth = np.expand_dims(file['y_depth'][sampling_idx], axis=1) # sampling size x 1
        z_depth = np.expand_dims(file['z_depth'][sampling_idx], axis=1) # sampling size x 1
        rgb_data = file['rgb'][sampling_idx, :] # # sampling size x 3
        # concatenate data column wise
        features = np.concatenate((x_depth, y_depth, z_depth, rgb_data), axis=1).astype(np.float32)
        labels = file['bbox_3d'].astype(np.float32)
        # these will need normalization, currently no normalization is applied

        # features shape (self.sampling_size, num_features=6)
        # labels shape (3, 8) - 8 from total num of vertices, 3 from xyz
        return torch.tensor(features), torch.tensor(labels)

if __name__ == "__main__":
    data_directory = r'C:\Users\Cem Okan\Dropbox (GaTech)\deep_learning_data\gts\train_data'
    dataset = CustomDataLoader(data_directory, sampling_size=900)
    data_loader = DataLoader(dataset, batch_size=16, shuffle=True)
    for x, y in data_loader:
        # x is a tensor shaped: (batch_size ,sampling_size=900, num_feature=6)
        # y is a tensor shaped: (batch_size ,3, 8)
        print("here is x: ", x.size())
        print("here is y: ", y.size())
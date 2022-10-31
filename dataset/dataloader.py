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
import time

class CustomDataLoader(Dataset):
    # returns
    def __init__(self, data_directory, sampling_size=900):
        # read the files in this data directory
        self.data_directory = data_directory
        self.data_list = os.listdir(data_directory)
        self.sampling_size = sampling_size
        self.feature_statistics = None
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
        # convert to tensor
        features = torch.tensor(features)
        labels = torch.tensor(labels)
        if self.feature_statistics is not None:  # if we calculated feature statistics and assigned it
            features -= self.feature_statistics[0]  # mean subtraction
            features /= self.feature_statistics[1]  # std division

        # features shape (self.sampling_size, num_features=6)
        # labels shape (3, 8) - 8 from total num of vertices, 3 from xyz
        return features, labels

    def apply_feature_stat(self, feature_stat):
        self.feature_statistics = feature_stat

def calculate_feature_statistics(data_loader, feature_dim = 6):
    general_sum = torch.zeros(feature_dim)
    general_std_sum = torch.zeros(feature_dim)
    counter = 0
    num_batches = len(data_loader)

    starting_time = time.time()
    for x, y in data_loader:
        # x is a tensor shaped: (batch_size ,sampling_size=900, num_feature=6)
        # y is a tensor shaped: (batch_size ,3, 8)
        # find the average x,y,z in the batch
        average_sample_coord = torch.mean(x, 1) #  batch_size x num_feature
        # sum over batch
        general_sum += average_sample_coord.sum(axis=0)
        general_std_sum += torch.std(average_sample_coord, 0)  # std of average sample coord across the batch
        # counter holds the total number of examples
        counter += average_sample_coord.size()[0]
    average_batch_coord = general_sum / counter
    std_batch_coord = general_std_sum / num_batches
    finishing_time = time.time()
    print(
        f"One pass on training data for calculating feature statistics took {finishing_time - starting_time} seconds.")
    return average_batch_coord, std_batch_coord

if __name__ == "__main__":
    data_directory = r'C:\Users\Cem Okan\Dropbox (GaTech)\deep_learning_data\gts\train_data'
    dataset = CustomDataLoader(data_directory, sampling_size=900)
    data_loader = DataLoader(dataset, batch_size=128, shuffle=True)
    mean_stat, std_stat = calculate_feature_statistics(data_loader, feature_dim=6)
    print("Feature mean statistics:", mean_stat)
    print("Feature std statistics:", std_stat)
    # apply feature statistics to data loader
    dataset.apply_feature_stat([mean_stat, std_stat])
    for x, y in data_loader:
        continue

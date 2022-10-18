from torch.utils.data import Dataset
import os
import numpy as np
from utils.utils import load_depth
import pickle
import cv2
from utils.utils import load_coord, load_colored, load_mask, load_label, load_depth


class CustomDataLoader(Dataset):
    # returns
    def __init__(self, data_dict_directory):
        with open(os.path.join(data_dict_directory), 'rb') as handle:
            self.data_list = pickle.load(handle)

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        # load the files and return

        # load the label - projected 3d bboxes for every instance in the scene
        label = load_label(self.data_list[idx])
        mask = load_mask(self.data_list[idx])
        coord = load_coord(self.data_list[idx])
        depth = load_depth(self.data_list[idx])
        colored = load_colored(self.data_list[idx])

        print("label shape: ", label.shape)
        print("mask shape: ", mask.shape)
        print("coord shape: ", coord.shape)
        print("depth shape: ", depth.shape)
        print("colored shape: ", colored.shape)

        return colored, coord, mask, depth, label


if __name__ == "__main__":
    data_directory = r'C:\Users\Cem Okan\Desktop\deep learning\project\data\Real/data_paths_dict.pickle'
    data_loader = CustomDataLoader(data_directory)
    for i in range(len(data_loader)):
        print("here is: ", data_loader[i])
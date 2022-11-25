from dataset.dataloader import CustomDataLoaderV3
from torch.utils.data import DataLoader
from lib.feature_extractor import FeatureExtractor
from utils.utils import split_into_boxes
import numpy as np
import torch

if __name__ == "__main__":
    user_dir = r'C:\Users\Cem Okan'  # change this on your computer
    data_dir = user_dir + r'\Dropbox (GaTech)\deep_learning_data'
    # sampling size does not work right now
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    extracted_feature_size = (80, 120)
    feature_extractor = FeatureExtractor(num_points=extracted_feature_size[0] * extracted_feature_size[1])
    feature_extractor.to(device)

    train_dataset = CustomDataLoaderV3(data_dir, data_name='train', apply_normalization=True, resize=extracted_feature_size)
    val_dataset = CustomDataLoaderV3(data_dir, data_name='val', apply_normalization=True, resize=extracted_feature_size)
    test_dataset = CustomDataLoaderV3(data_dir, data_name='test', apply_normalization=True, resize=extracted_feature_size)

    batch_size = 4
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

    for (pc, rgb, bbox_coords) in train_loader:
        # flatten pc data such that N x C x H x W will become N x C x (HxW)
        pc = torch.flatten(pc, start_dim=2, end_dim=3)
        # all_feats [N x color_feat, geo_feat, global_feat x (H_fix, W_fix)] - N x 128 + 256 + 1024 x (80, 120)
        all_feats = feature_extractor(rgb.float().to(device), pc.float().to(device), None)
        color_feats = torch.reshape(all_feats[:, :128, :], (all_feats.size()[0], 128, extracted_feature_size[0], extracted_feature_size[1]))
        geo_feats = torch.reshape(all_feats[:, 128:128 + 256, :],
                                    (all_feats.size()[0], 256, extracted_feature_size[0], extracted_feature_size[1]))
        # color feats shape: N x 128 x H_fix x W_fix
        # geo feats shape: N x 256 x H_fix x W_fix
        # now split it into boxes - 6 boxes (2 x 3) in this case
        splits = split_into_boxes(geo_feats, height_sep=2, width_sep=3)
        # splits shape: N x 256 x NumBoxes(height_sep x width_sep)  x H_fix x W_fix
        print("split shape: ", splits.size())
        # now these are rectangular images but for each image H and W will be different
        # let's resize every image to some value

        # out = feature_extractor()
from dataset.dataloader import CustomDataLoaderV3
from torch.utils.data import DataLoader
from lib.feature_extractor import FeatureExtractor
from lib.network import PCNet, ColorNet, GlobNet
from utils.utils import split_into_boxes, get_relationship_from_splits, get_3d_bbox, rigid_transform_3D
import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
import os
from matplotlib import pyplot as plt
import time


def eval(data, nets, resizer, criterion, device, scale_dims):
    pc = data[0]
    rgb = data[1]
    bbox_coords = data[2]
    scale_box = data[3][0].item()
    for k in range(len(nets)):
        nets[k].eval()
    with torch.no_grad():
        # feature extractor
        color_feats, geo_feats = nets[0](rgb.float().to(device), pc.float().to(device))
        # color feats shape: N x 32 x H_object x W_object
        # geo feats shape: N x 32 x H_object x W_object
        # resize
        geo_feats = resizer(geo_feats)
        color_feats = resizer(color_feats)
        # color feats shape: N x 32 x H_fix x W_fix
        # geo feats shape: N x 32 x H_fix x W_fix
        # now split it into boxes - 6 boxes (2 x 3) in this case
        splits = split_into_boxes(geo_feats, height_sep=scale_dims[0], width_sep=scale_dims[1])
        # splits shape: N x 32 x NumBoxes(height_sep x width_sep)  x H_fix x W_fix
        # now extract the relationships between slices
        splits_relations = get_relationship_from_splits(splits, rel_type='subtraction')
        # now it has shape N x 32 x Comb(Num_boxes, 2)  x H_fix x W_fix
        # now concatenate normal splits with these relations
        splits = torch.cat((splits, splits_relations), dim=2)
        # now it has shape N x 32 x (Comb(Num_boxes, 2) + num_boxes)  x H_fix x W_fix
        # now flatten embedding dim and channel dimensions to create a single channel dim
        splits = torch.flatten(splits, start_dim=1, end_dim=2)
        # now it has shape N x (32 x (Comb(Num_boxes, 2) + num_boxes))  x H_fix x W_fix
        pc_out = nets[1](splits.float().to(device))
        color_out = nets[2](color_feats.float().to(device))
        glob_feat = torch.cat((pc_out, color_out), dim=1)
        out = nets[3](glob_feat)
        # out is 24 dimensional bbox - we have to convert it into rotation and translation
        loss = criterion(out, bbox_coords.float().to(device))
        # RMSE error is more interpretable
        overall_loss = torch.sqrt(loss).detach()
        # reshape
        out = torch.reshape(out.cpu(), (3, 8)).numpy()  # Reshape from 1x24 to 3x8
        out /= 1000  # due to scaling error in data loader
        # now we will try to find R and t matrices
        # first create the initial box with the known scale
        bbox_3d = get_3d_bbox(scale_box, shift=0)  # shape 3x8
        # now find the rotation and translation between out and bbox_3d
        R, T = rigid_transform_3D(bbox_3d, out)


    return overall_loss, out.detach(), R, T

if __name__ == "__main__":
    user_dir = r'C:\Users\Cem Okan'  # change this on your computer
    data_dir = user_dir + r'\Dropbox (GaTech)\deep_learning_data'
    # sampling size does not work right now
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Device: ", device)
    extracted_feature_size = (32, 48)

    test_dataset = CustomDataLoaderV3(data_dir, data_name='test', apply_normalization=True, resize=extracted_feature_size)

    batch_size = 1  # due to proposed approach, this should be 1
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # initialize PCNet
    emb = 64
    globnet = GlobNet(2 * emb, 24)
    globnet.to(device)

    exp_folder = os.path.join(user_dir, data_dir, 'Experiments', 'ExpScale6')
    scale_dims = (2, 3)
    model_path = os.path.join(exp_folder, f'UseThis')
    feature_extractor = FeatureExtractor(num_points=extracted_feature_size[0] * extracted_feature_size[1])
    pcnet = PCNet(inchannels=21 * 32, out_dim=512)
    colornet = ColorNet(inchannels=32)
    pcnet.load_state_dict(torch.load(os.path.join(model_path, "pcnet.pth")))
    colornet.load_state_dict(torch.load(os.path.join(model_path, "colornet.pth")))
    feature_extractor.load_state_dict(torch.load(os.path.join(model_path, "feature_extractor.pth")))
    feature_extractor.to(device)
    pcnet.to(device)
    colornet.to(device)

    nets = [feature_extractor, pcnet, colornet, globnet]

    for (pc, rgb, bbox_coords, object_scale) in test_loader:
        data = [pc, rgb, bbox_coords, object_scale]
        overall_loss, pred, R, T =  eval(data, nets, test_dataset.resizer, nn.MSELoss(), device, scale_dims)
        print("overall loss: ", overall_loss)
        print("pred: ", pred)
        print("R: ", R)
        print("T: ", T)
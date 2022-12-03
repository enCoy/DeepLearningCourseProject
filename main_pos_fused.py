from dataset.dataloader import CustomDataLoaderV3
from torch.utils.data import DataLoader
from lib.feature_extractor import FeatureExtractor
from lib.network import PCNet, ColorNet, GlobNet
from utils.utils import split_into_boxes, get_relationship_from_splits
import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
import os
from matplotlib import pyplot as plt
import time

def plot_single_continuous_plot(x_axis, y_axis, title, x_label_pick, y_label_pick, color=None, linestyle=None,
                                hold_on=False, legend_enable=False, label=None, save_path=None):
    """
    Plots the given data
    :param x_axis: What will be the data on the x-axis?
    :type x_axis: 1D array
    :param y_axis: What will be the data on the y-axis?
    :type y_axis: 1D array
    :param title: title of the plot
    :type title: str
    :param x_label_pick: x-label name of the plot
    :type x_label_pick: str
    :param y_label_pick: y-label name of the plot
    :type y_label_pick: str
    :param color: specific color wanted. If none, it chooses randomly from the list below
    :type color: str
    :param linestyle: specific linestyle wanted. If none, it chooses randomly from the list below
    :type linestyle: str
    :param show_enable: True if you want to display. If you want to use hold on option, set it to false for the image
                        at the background
    :type show_enable: bool
    :param hold_on: True if you want to plot the current plot on top of the previous one
    :type hold_on: bool
    :param legend_enable: True if you want legend. Useful if you use hold on.
    :type legend_enable: bool
    :param label: Label of the plot that will be seen in legend
    :type label: str
    """
    LINESTYLES = ["solid", "dashed", "dotted", "dashdot"]
    FIGSIZE = (12, 8)
    TITLE_FONT_SIZE = 16
    LABEL_FONT_SIZE = 12

    if not hold_on:
        plt.figure(figsize=FIGSIZE)
    plt.title(title, fontsize=TITLE_FONT_SIZE, fontweight="bold", fontname="Arial")
    plt.xlabel(x_label_pick, fontsize=LABEL_FONT_SIZE, fontweight="normal", fontname="Arial")
    plt.ylabel(y_label_pick, fontsize=LABEL_FONT_SIZE, fontweight="normal", fontname="Arial")
    if x_axis is None:  # just use arange
        x_axis = np.arange(len(y_axis))+1
    plt.plot(x_axis, y_axis, color=color, linestyle=linestyle, label=label)

    plt.legend()
    if save_path is not None:
        image_format = 'png'  # e.g .png, .svg, etc
        plt.savefig(save_path, format=image_format, dpi=1200)

def plot_loss_curves(train_losses, val_losses,save_loc=None):
    epochs = np.arange(len(train_losses)) + 1
    plot_single_continuous_plot(epochs, train_losses, 'Loss curve', 'Epoch', 'Loss', color='tab:red', label='train')
    plot_single_continuous_plot(epochs, val_losses, 'Loss curve', 'Epoch', 'Loss', color='navy',
                                hold_on=True, legend_enable=True, label='val',
                                save_path=os.path.join(save_loc, "Train-Val Loss Curve.png"))

#
def train_loop(scales, globnet,
               resizer, loader, criterion, optimizer, device,
               scale_dims=((1, 1), (2, 3), (4, 4))):
    overall_loss = 0
    optimizer.zero_grad()
    for s in range(len(scales)):
        for k in range(len(scales[s])):
            scales[s][k].train()
    counter = 0
    for (pc, rgb, bbox_coords) in loader:
        counter += 1
        overall_feature = None
        for s in range(len(scales)):
            color_feats, geo_feats = scales[s][0](rgb.float().to(device), pc.float().to(device))
            geo_feats = resizer(geo_feats)
            color_feats = resizer(color_feats)
            splits = split_into_boxes(geo_feats, height_sep=scale_dims[s][0], width_sep=scale_dims[s][1])
            splits_relations = get_relationship_from_splits(splits, rel_type='subtraction')
            splits = torch.cat((splits, splits_relations), dim=2)
            splits = torch.flatten(splits, start_dim=1, end_dim=2)
            pc_out = scales[s][1](splits.float().to(device))
            color_out = scales[s][2](color_feats.float().to(device))
            s_feature = torch.cat((pc_out, color_out), dim=1)
            if s == 0:
                overall_feature = s_feature
            else:
                overall_feature = torch.cat((overall_feature, s_feature), dim=1)

        out = globnet(overall_feature)
        loss = criterion(out, bbox_coords.float().to(device))
        loss.backward()

        clipping_value = 5  # arbitrary value of your choosing
        torch.nn.utils.clip_grad_norm_(globnet.parameters(), clipping_value)
        optimizer.step()
        # RMSE error is more interpretable
        overall_loss += torch.sqrt(loss).detach()
    overall_loss = overall_loss / counter

    return overall_loss, globnet


def val_loop(scales, globnet,
               resizer, loader, criterion, device,
               scale_dims=((1, 1), (2, 3), (4, 4))):
    overall_loss = 0
    for s in range(len(scales)):
        for k in range(len(scales[s])):
            scales[s][k].eval()
    counter = 0
    for (pc, rgb, bbox_coords) in loader:
        counter += 1
        overall_feature = None
        for s in range(len(scales)):
            color_feats, geo_feats = scales[s][0](rgb.float().to(device), pc.float().to(device))
            geo_feats = resizer(geo_feats)
            color_feats = resizer(color_feats)
            splits = split_into_boxes(geo_feats, height_sep=scale_dims[s][0], width_sep=scale_dims[s][1])
            splits_relations = get_relationship_from_splits(splits, rel_type='subtraction')
            splits = torch.cat((splits, splits_relations), dim=2)
            splits = torch.flatten(splits, start_dim=1, end_dim=2)
            pc_out = scales[s][1](splits.float().to(device))
            color_out = scales[s][2](color_feats.float().to(device))
            s_feature = torch.cat((pc_out, color_out), dim=1)
            if s == 0:
                overall_feature = s_feature
            else:
                overall_feature = torch.cat((overall_feature, s_feature), dim=1)

        out = globnet(overall_feature)
        loss = criterion(out, bbox_coords.float().to(device))

        # RMSE error is more interpretable
        overall_loss += torch.sqrt(loss).detach()
    overall_loss = overall_loss / counter

    return overall_loss

if __name__ == "__main__":
    user_dir = r'C:\Users\Cem Okan'  # change this on your computer
    data_dir = user_dir + r'\Dropbox (GaTech)\deep_learning_data'
    # sampling size does not work right now
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Device: ", device)
    extracted_feature_size = (32, 48)


    train_dataset = CustomDataLoaderV3(data_dir, data_name='train', apply_normalization=True, resize=extracted_feature_size)
    val_dataset = CustomDataLoaderV3(data_dir, data_name='val', apply_normalization=True, resize=extracted_feature_size)
    test_dataset = CustomDataLoaderV3(data_dir, data_name='test', apply_normalization=True, resize=extracted_feature_size)

    batch_size = 1  # due to proposed approach, this should be 1
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

    # initialize PCNet
    emb = 64
    globnet = GlobNet(3 * 2 * emb, 24)
    globnet.to(device)

    exp_folder = os.path.join(user_dir, data_dir, 'Experiments', 'ExpScale')

    scale_1_exp_folder = os.path.join(user_dir, data_dir, 'Experiments', 'ExpScale1')
    scale_1_model_path = os.path.join(scale_1_exp_folder, f'UseThis')
    scale_1_feature_extractor = FeatureExtractor(num_points=extracted_feature_size[0] * extracted_feature_size[1])
    scale_1_pcnet = PCNet(inchannels=1 * 32, out_dim=3072)
    scale_1_colornet = ColorNet(inchannels=32)
    scale_1_pcnet.load_state_dict(torch.load(os.path.join(scale_1_model_path, "pcnet.pth")))
    scale_1_colornet.load_state_dict(torch.load(os.path.join(scale_1_model_path, "colornet.pth")))
    scale_1_feature_extractor.load_state_dict(torch.load(os.path.join(scale_1_model_path, "feature_extractor.pth")))
    scale_1_feature_extractor.to(device)
    scale_1_pcnet.to(device)
    scale_1_colornet.to(device)

    scale_2_exp_folder = os.path.join(user_dir, data_dir, 'Experiments', 'ExpScale6')
    scale_2_model_path = os.path.join(scale_2_exp_folder, f'UseThis')
    scale_2_feature_extractor = FeatureExtractor(num_points=extracted_feature_size[0] * extracted_feature_size[1])
    scale_2_pcnet = PCNet(inchannels=21 * 32, out_dim=512)
    scale_2_colornet = ColorNet(inchannels=32)
    scale_2_pcnet.load_state_dict(torch.load(os.path.join(scale_2_model_path, "pcnet.pth")))
    scale_2_colornet.load_state_dict(torch.load(os.path.join(scale_2_model_path, "colornet.pth")))
    scale_2_feature_extractor.load_state_dict(torch.load(os.path.join(scale_2_model_path, "feature_extractor.pth")))
    scale_2_feature_extractor.to(device)
    scale_2_pcnet.to(device)
    scale_2_colornet.to(device)

    scale_3_exp_folder = os.path.join(user_dir, data_dir, 'Experiments', 'ExpScale16')
    scale_3_model_path = os.path.join(scale_3_exp_folder, f'UseThis')
    scale_3_feature_extractor = FeatureExtractor(num_points=extracted_feature_size[0] * extracted_feature_size[1])
    scale_3_pcnet = PCNet(inchannels=136 * 32, out_dim=128)
    scale_3_colornet = ColorNet(inchannels=32)
    scale_3_pcnet.load_state_dict(torch.load(os.path.join(scale_3_model_path, "pcnet.pth")))
    scale_3_colornet.load_state_dict(torch.load(os.path.join(scale_3_model_path, "colornet.pth")))
    scale_3_feature_extractor.load_state_dict(torch.load(os.path.join(scale_3_model_path, "feature_extractor.pth")))
    scale_3_feature_extractor.to(device)
    scale_3_pcnet.to(device)
    scale_3_colornet.to(device)

    # freeze all these layers
    for param in scale_1_pcnet.parameters():
        param.requires_grad = False
    for param in scale_1_colornet.parameters():
        param.requires_grad = False
    for param in scale_1_feature_extractor.parameters():
        param.requires_grad = False
    for param in scale_2_pcnet.parameters():
        param.requires_grad = False
    for param in scale_2_colornet.parameters():
        param.requires_grad = False
    for param in scale_2_feature_extractor.parameters():
        param.requires_grad = False
    for param in scale_3_pcnet.parameters():
        param.requires_grad = False
    for param in scale_3_colornet.parameters():
        param.requires_grad = False
    for param in scale_3_feature_extractor.parameters():
        param.requires_grad = False

    # we have 3 scales, each produces 2*emb dimensional embedding

    crit_x = nn.MSELoss()
    optim_x = optim.SGD([*globnet.parameters()], lr=0.00003, momentum=0.9)

    scale_1_nets = [scale_1_feature_extractor, scale_1_pcnet, scale_1_colornet]
    scale_2_nets = [scale_2_feature_extractor, scale_2_pcnet, scale_2_colornet]
    scale_3_nets = [scale_3_feature_extractor, scale_3_pcnet, scale_3_colornet]
    scales = [scale_1_nets, scale_2_nets, scale_3_nets]
    num_epochs = 50
    train_losses = []
    val_losses = []

    prev_time = time.time()
    for epoch in range(num_epochs):
        print(f"Epoch {epoch + 1}")
        epoch_train_loss, nets = train_loop(scales, globnet, resizer=train_dataset.resizer,
                                              loader=train_loader, criterion=crit_x, optimizer=optim_x, device=device)
        train_losses.append(epoch_train_loss.item())

        current_time = time.time()
        print(f"Train loss: {epoch_train_loss}    Time passed: {current_time - prev_time}")
        prev_time = current_time

        epoch_val_loss = val_loop(scales, globnet, resizer=val_dataset.resizer,
                                              loader=val_loader, criterion=crit_x, device=device)
        val_losses.append(epoch_val_loss.item())
        current_time = time.time()
        print(f"Val loss: {epoch_val_loss}    Time passed: {current_time - prev_time}")
        prev_time = current_time

        epoch_folder = os.path.join(exp_folder, f'Epoch-{epoch+1}')
        if not os.path.isdir(epoch_folder):
            os.makedirs(epoch_folder)
        plot_loss_curves(train_losses, val_losses, save_loc=epoch_folder)

        # save models
        torch.save(globnet.state_dict(), os.path.join(epoch_folder, "fuse_net.pth"))



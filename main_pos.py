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
def train_loop(nets, resizer, loader, criterion, optimizer, device):
    overall_loss = 0
    for k in range(len(nets)):
        nets[k].train()
    counter = 0
    for (pc, rgb, bbox_coords) in loader:
        counter += 1
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
        splits = split_into_boxes(geo_feats, height_sep=2, width_sep=3)
        # splits shape: N x 32 x NumBoxes(height_sep x width_sep)  x H_fix x W_fix

        # SALIL YOU CAN USE IT UNTIL HERE
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

        loss = criterion(out, bbox_coords.float().to(device))
        loss.backward()

        clipping_value = 5  # arbitrary value of your choosing
        torch.nn.utils.clip_grad_norm_(nets[0].parameters(), clipping_value)
        torch.nn.utils.clip_grad_norm_(nets[1].parameters(), clipping_value)
        torch.nn.utils.clip_grad_norm_(nets[2].parameters(), clipping_value)
        torch.nn.utils.clip_grad_norm_(nets[3].parameters(), clipping_value)

        optimizer.step()
        # RMSE error is more interpretable
        overall_loss += torch.sqrt(loss)


    overall_loss = overall_loss / counter

    return overall_loss, nets

def val_loop(nets, resizer, loader, criterion, device):
    overall_loss = 0
    for k in range(len(nets)):
        nets[k].eval()
    with torch.no_grad():
        counter = 0
        for (pc, rgb, bbox_coords) in loader:
            counter += 1
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
            splits = split_into_boxes(geo_feats, height_sep=2, width_sep=3)
            # splits shape: N x 32 x NumBoxes(height_sep x width_sep)  x H_fix x W_fix

            # SALIL YOU CAN USE IT UNTIL HERE
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

            loss = criterion(out, bbox_coords.float().to(device))
            # RMSE error is more interpretable
            overall_loss += torch.sqrt(loss)


    overall_loss = overall_loss / counter

    return overall_loss


if __name__ == "__main__":
    user_dir = r'C:\Users\Cem Okan'  # change this on your computer
    data_dir = user_dir + r'\Dropbox (GaTech)\deep_learning_data'
    # sampling size does not work right now
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    extracted_feature_size = (32, 48)
    feature_extractor = FeatureExtractor(num_points=extracted_feature_size[0] * extracted_feature_size[1])
    feature_extractor.to(device)

    train_dataset = CustomDataLoaderV3(data_dir, data_name='train', apply_normalization=True, resize=extracted_feature_size)
    val_dataset = CustomDataLoaderV3(data_dir, data_name='val', apply_normalization=True, resize=extracted_feature_size)
    test_dataset = CustomDataLoaderV3(data_dir, data_name='test', apply_normalization=True, resize=extracted_feature_size)

    batch_size = 1  # due to proposed approach, this should be 1
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

    # initialize PCNet
    emb = 64
    pcnet = PCNet(inchannels=21 * 32)
    colornet = ColorNet(inchannels=32)
    globnet = GlobNet(2 * emb, 24)
    pcnet.to(device)
    colornet.to(device)
    globnet.to(device)

    crit_x = nn.MSELoss()
    optim_x = optim.SGD([*feature_extractor.parameters(), *pcnet.parameters(),
                           *colornet.parameters(), *globnet.parameters()], lr=0.00005, momentum=0.9)
    nets = [feature_extractor, pcnet, colornet, globnet]

    exp_folder = os.path.join(user_dir, data_dir, 'Experiments','ExpScale6')

    num_epochs = 50
    train_losses = []
    val_losses = []
    for epoch in range(num_epochs):
        print(f"Epoch {epoch + 1}")
        epoch_train_loss, nets = train_loop(nets=nets, resizer=train_dataset.resizer,
                                              loader=train_loader, criterion=crit_x, optimizer=optim_x, device=device)
        train_losses.append(epoch_train_loss.item())
        print("Train loss:", epoch_train_loss)
        epoch_val_loss = val_loop(nets=nets, resizer=val_dataset.resizer,
                                              loader=val_loader, criterion=crit_x, device=device)
        val_losses.append(epoch_val_loss.item())
        print("Val loss:", epoch_val_loss)

        epoch_folder = os.path.join(exp_folder, f'Epoch-{epoch+1}')
        if not os.path.isdir(epoch_folder):
            os.makedirs(epoch_folder)
        plot_loss_curves(train_losses, val_losses, save_loc=epoch_folder)

        # save models
        torch.save(nets[0].state_dict(), os.path.join(epoch_folder, "feature_extractor.pth"))
        torch.save(nets[1].state_dict(), os.path.join(epoch_folder, "pcnet.pth"))
        torch.save(nets[2].state_dict(), os.path.join(epoch_folder, "colornet.pth"))
        torch.save(nets[3].state_dict(), os.path.join(epoch_folder, "globnet.pth"))


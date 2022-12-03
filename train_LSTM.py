from dataset.dataloader import CustomDataLoaderV3
from torch.utils.data import DataLoader
from lib.feature_extractor import FeatureExtractor
from lib.network import LSTMPose
from utils.utils import split_into_boxes, get_relationship_from_splits
import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
import os
from matplotlib import pyplot as plt
import time
from torchviz import make_dot

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
def train_loop(net, loader, criterion, optimizer, device):

    overall_loss = 0
    net.train()
    counter = 0
    # start = time.time()
    for (pc, rgb, bbox_coords) in loader:
        counter += 1

        plt.imshow(rgb.squeeze().permute(1,2,0))

        plt.show()
        pc = pc.float().to(device)
        rgb= rgb.float().to(device)


        out = net(rgb,pc)

        #make_dot(out, params=dict(net.named_parameters())).render("LSTMpose_graph", format="png")


        loss = criterion(out, bbox_coords.float().to(device))
        optimizer.zero_grad()
        loss.backward()

        clipping_value = 5  # arbitrary value of your choosing
        torch.nn.utils.clip_grad_norm_(net.parameters(), clipping_value)


        optimizer.step()
        # RMSE error is more interpretable
        overall_loss += torch.sqrt(loss).detach()

        # if ((counter % 100) == 0):
        #     end = time.time()
        #     print("Counter is: {} ".format(counter))
        #     print("time take: {}".format(end-start))
        #     start = end





    overall_loss = overall_loss / counter

    return overall_loss, net

def val_loop(net, loader, criterion, device):
    overall_loss = 0
    net.eval()
    with torch.no_grad():
        counter = 0
        for (pc, rgb, bbox_coords) in loader:

            pc = pc.float().to(device)
            rgb = rgb.float().to(device)

            counter += 1
            out = net(rgb,pc)

            loss = criterion(out, bbox_coords.float().to(device))
            # RMSE error is more interpretable
            overall_loss += torch.sqrt(loss).detach()

    overall_loss = overall_loss / counter

    return overall_loss

def run_inference(net, loader, criterion, device):
    overall_loss = 0
    net.eval()
    with torch.no_grad():
        counter = 0

        for (pc, rgb, bbox_coords) in loader:
            pc = pc.float().to(device)
            rgb = rgb.float().to(device)

            counter += 1
            out = net(rgb, pc)


            diff = bbox_coords.float().to(device) - out
            print(diff)

            loss = criterion(out, bbox_coords.float().to(device))
            print(loss)
            # RMSE error is more interpretable
            overall_loss += torch.sqrt(loss).detach()



if __name__ == "__main__":
    user_dir = r'C:\Users\nadun'  # change this on your computer
    data_dir = user_dir + r'\Dropbox (GaTech)\deep_learning_data (1)'
    # sampling size does not work right now
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    extracted_feature_size = (36, 36)


    train_dataset = CustomDataLoaderV3(data_dir, data_name='train', apply_normalization=True, resize=extracted_feature_size)
    val_dataset = CustomDataLoaderV3(data_dir, data_name='val', apply_normalization=True, resize=extracted_feature_size)
    test_dataset = CustomDataLoaderV3(data_dir, data_name='test', apply_normalization=True, resize=extracted_feature_size)

    batch_size = 1  # due to proposed approach, this should be 1
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)





    #init LSTM_net
    net = LSTMPose(24, device=device)
    net.to(device)
    #
    #
    #
    crit_x = nn.MSELoss()
    # optim_x = optim.SGD(net.parameters(), lr=0.00005, momentum=0.9)
    #
    #
    # exp_folder = os.path.join(user_dir, data_dir, 'Experiments','LSTMPose')
    #
    # num_epochs = 50
    # train_losses = []
    # val_losses = []

    net.load_state_dict(torch.load(r'C:\Users\nadun\Dropbox (GaTech)\deep_learning_data (1)\Experiments\LSTMPose\Epoch-13\LSTMnet.pth'))

    run_inference(net, test_loader, crit_x, device)

    # for epoch in range(num_epochs):
    #     print(f"Epoch {epoch + 1}")
    #     epoch_train_loss, nets = train_loop(net, loader=train_loader, criterion=crit_x, optimizer=optim_x, device=device)
    #     train_losses.append(epoch_train_loss.item())
    #     print("Train loss:", epoch_train_loss)
    #     epoch_val_loss = val_loop(net, loader=val_loader, criterion=crit_x, device=device)
    #     val_losses.append(epoch_val_loss.item())
    #     print("Val loss:", epoch_val_loss)
    #
    #     epoch_folder = os.path.join(exp_folder, f'Epoch-{epoch+1}')
    #     if not os.path.isdir(epoch_folder):
    #         os.makedirs(epoch_folder)
    #     plot_loss_curves(train_losses, val_losses, save_loc=epoch_folder)

        # save models
        # torch.save(net.state_dict(), os.path.join(epoch_folder, "LSTMnet.pth"))


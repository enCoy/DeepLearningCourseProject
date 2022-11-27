from dataset.dataloader import CustomDataLoaderV3
from torch.utils.data import DataLoader
from lib.feature_extractor import FeatureExtractor
from lib.network import PCNet, ColorNet, GlobNet
from utils.utils import split_into_boxes, get_relationship_from_splits
import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
#
def train_loop(nets, resizer, loader, criterion, optimizer, device):
    overall_loss = 0
    for k in range(len(nets)):
        nets[k].train()
    counter = 0
    for (pc, rgb, bbox_coords) in loader:
        counter += 1
        print("counter: ", counter)
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
        optimizer.step()
        # RMSE error is more interpretable
        overall_loss += torch.sqrt(loss)


    overall_loss = overall_loss / counter
    print("overall loss: ", overall_loss)
    print("len loader: ", len(loader))

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
    optim_x = optim.Adam([*feature_extractor.parameters(), *pcnet.parameters(),
                           *colornet.parameters(), *globnet.parameters()], lr=0.0001)
    nets = [feature_extractor, pcnet, colornet, globnet]

    num_epochs = 5
    for epoch in range(num_epochs):
        print(f"Epoch {epoch + 1}")
        epoch_train_loss, nets = train_loop(nets=nets, resizer=train_dataset.resizer,
                                              loader=train_loader, criterion=crit_x, optimizer=optim_x, device=device)
        print("Train loss:", epoch_train_loss)
        epoch_val_loss = val_loop(nets=nets, resizer=val_dataset.resizer,
                                              loader=val_loader, criterion=crit_x, device=device)
        print("Val loss:", epoch_val_loss)



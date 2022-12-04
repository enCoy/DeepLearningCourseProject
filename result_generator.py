from dataset.dataloader import CustomDataLoaderV3
from torch.utils.data import DataLoader
from lib.feature_extractor import FeatureExtractor
from lib.network import PCNet, ColorNet, GlobNet
from utils.utils import split_into_boxes, get_relationship_from_splits, get_3d_bbox, rigid_transform_3D, transform_coordinates_3d
import numpy as np
import torch
from utils.utils import calculate_2d_projections
from visualizers.visualization import draw_bboxes
import torch.optim as optim
import torch.nn as nn
import os
from matplotlib import pyplot as plt
import time


def eval(data, nets, resizer, criterion, device, scale_dims, sRT, s_correction, object_scale):
    pc = data[0]
    rgb = data[1]
    bbox_coords = data[2]
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
        out = torch.reshape(out, (3, 8)).cpu().numpy()  # Reshape from 1x24 to 3x8

    return overall_loss, out

def eval_fuse(data, scales, globnet,
               resizer, criterion, device,
               scale_dims, sRT, s_correction, object_scale):
    pc = data[0]
    rgb = data[1]
    bbox_coords = data[2]
    overall_loss = 0
    for s in range(len(scales)):
        for k in range(len(scales[s])):
            scales[s][k].eval()
    with torch.no_grad():
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
        # reshape
        out = torch.reshape(out, (3, 8))  # Reshape from 1x24 to 3x8

        # now we will try to find R and t matrices
        # fix the prediction
        fixed_out = fix_mistake(out.cpu().numpy(), sRT, s_correction, object_scale)
        # first create the initial box with the known scale - this will be the source
        bbox_3d = get_3d_bbox(object_scale, shift=0)  # shape 3x8
        # now find the rotation and translation between out and bbox_3d
        R, T = get_transformation_terms(bbox_3d, fixed_out.numpy())
        # now we can use this for our metric

    return overall_loss, out.cpu().numpy() , R, T




def fix_mistake(coords, sRT, s_correction, object_scale):
    coords = transform_coordinates_3d(coords, np.linalg.inv(sRT))
    coords = np.matmul(np.diag(1 / s_correction.flatten()), coords)
    coords = coords * object_scale
    coords = transform_coordinates_3d(coords, sRT)
    return coords


def get_transformation_terms(source, target):
    # source 3xN is the points that are transformed
    # target 3xN is the transformed version
    R, T = rigid_transform_3D(source, target)
    return R, T



def visualize_bboxes(image, gt_bbox, pred_bboxes, object, object_scale, s_correction, sRT, model_names):
    """
    @param image: full image containing the object - tensor of shape (1,H,W,C)
    @param gt_bbox: gt bbox for object - tensor of shape (1,24)
    @param pred_bbox: pred bbox for objects - tensor of shape (1,24)
    @param object: img crop of object we are drawing the bbox for - tensor of shape (1,C,H_crop,W_crop)
    @param s_correction:
    """
    #Change everything from tensors to numpy for utils
    image = image.squeeze().cpu().numpy()
    gt_bbox = gt_bbox.cpu()
    object = object.squeeze().permute(1,2,0).cpu().numpy()
    intrinsics = np.array([[577.5, 0, 319.5], [0, 577.5, 239.5], [0, 0, 1]], dtype=np.float)

    gt_bbox = torch.reshape(gt_bbox, (3,8)).numpy() #Reshape from 1x24 to 3x8

    # apply correction for data loader mistake - to gt
    gt_bbox = transform_coordinates_3d(gt_bbox, np.linalg.inv(sRT))
    gt_bbox = np.matmul(np.diag(1/s_correction.flatten()), gt_bbox)
    gt_bbox = gt_bbox * object_scale.item()
    gt_bbox = transform_coordinates_3d(gt_bbox, sRT)
    gt_bbox = gt_bbox.numpy()

    gt_2D = calculate_2d_projections(gt_bbox, intrinsics)
    gt_img = draw_bboxes(np.copy(image), gt_2D, (0, 255, 0))

    fig = plt.figure(figsize=(10, 7))
    rows = 2
    columns = 3

    fig.add_subplot(rows, columns, 1)  # visualize the full image with the gt bbox
    plt.imshow(gt_img)
    plt.title("Ground truth")

    # apply correction for data loader mistake - to pred
    for j in range(2, 2 + len(pred_bboxes)):
        pred_bbox = pred_bboxes[j - 2]

        pred_bbox = np.reshape(pred_bbox, (3, 8))
        pred_bbox = transform_coordinates_3d(pred_bbox, np.linalg.inv(sRT))
        pred_bbox = np.matmul(np.diag(1 / s_correction.flatten()), pred_bbox)
        pred_bbox = pred_bbox * object_scale.item()
        pred_bbox = transform_coordinates_3d(pred_bbox, sRT)
        pred_bbox = pred_bbox.numpy()

        pred_2D = calculate_2d_projections(pred_bbox, intrinsics)
        pred_img = draw_bboxes(np.copy(image), pred_2D, (0, 255, 0))

        fig.add_subplot(rows, columns, j)  # show the predicted bbox
        plt.imshow(pred_img)
        plt.title(f"{model_names[j - 2]}")

    fig.add_subplot(rows, columns, 2 + len(pred_bboxes)) #show the object
    plt.imshow(object)
    plt.title("Object")

    plt.show()


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

    emb = 64
    # initialize model 1 with scale 1
    exp_folder = os.path.join(user_dir, data_dir, 'Experiments', 'ExpScale1')
    scale_dims_m1 = (1, 1)
    model_path = os.path.join(exp_folder, f'UseThis')
    feature_extractor = FeatureExtractor(num_points=extracted_feature_size[0] * extracted_feature_size[1])
    pcnet = PCNet(inchannels=1 * 32, out_dim=3072)
    colornet = ColorNet(inchannels=32)
    globnet = GlobNet(2 * emb, 24)
    pcnet.load_state_dict(torch.load(os.path.join(model_path, "pcnet.pth")))
    colornet.load_state_dict(torch.load(os.path.join(model_path, "colornet.pth")))
    feature_extractor.load_state_dict(torch.load(os.path.join(model_path, "feature_extractor.pth")))
    globnet.load_state_dict(torch.load(os.path.join(model_path, "globnet.pth")))
    feature_extractor.to(device)
    pcnet.to(device)
    colornet.to(device)
    globnet.to(device)
    model_1_nets = [feature_extractor, pcnet, colornet, globnet]

    # initialize model 2 with scale 6
    exp_folder = os.path.join(user_dir, data_dir, 'Experiments', 'ExpScale6')
    scale_dims_m2 = (2, 3)
    model_path = os.path.join(exp_folder, f'UseThis')
    feature_extractor = FeatureExtractor(num_points=extracted_feature_size[0] * extracted_feature_size[1])
    pcnet = PCNet(inchannels=21 * 32, out_dim=512)
    colornet = ColorNet(inchannels=32)
    globnet = GlobNet(2 * emb, 24)
    pcnet.load_state_dict(torch.load(os.path.join(model_path, "pcnet.pth")))
    colornet.load_state_dict(torch.load(os.path.join(model_path, "colornet.pth")))
    feature_extractor.load_state_dict(torch.load(os.path.join(model_path, "feature_extractor.pth")))
    globnet.load_state_dict(torch.load(os.path.join(model_path, "globnet.pth")))
    feature_extractor.to(device)
    pcnet.to(device)
    colornet.to(device)
    globnet.to(device)
    model_2_nets = [feature_extractor, pcnet, colornet, globnet]

    # initialize model 3 with scale 16
    exp_folder = os.path.join(user_dir, data_dir, 'Experiments', 'ExpScale16')
    scale_dims_m3 = (4, 4)
    model_path = os.path.join(exp_folder, f'UseThis')
    feature_extractor = FeatureExtractor(num_points=extracted_feature_size[0] * extracted_feature_size[1])
    pcnet = PCNet(inchannels=136 * 32, out_dim=128)
    colornet = ColorNet(inchannels=32)
    globnet = GlobNet(2 * emb, 24)
    pcnet.load_state_dict(torch.load(os.path.join(model_path, "pcnet.pth")))
    colornet.load_state_dict(torch.load(os.path.join(model_path, "colornet.pth")))
    feature_extractor.load_state_dict(torch.load(os.path.join(model_path, "feature_extractor.pth")))
    globnet.load_state_dict(torch.load(os.path.join(model_path, "globnet.pth")))
    feature_extractor.to(device)
    pcnet.to(device)
    colornet.to(device)
    globnet.to(device)
    model_3_nets = [feature_extractor, pcnet, colornet, globnet]



    # now fusion model
    scale_1_nets = model_1_nets[0:3]
    scale_2_nets = model_2_nets[0:3]
    scale_3_nets = model_3_nets[0:3]
    fusion_model_scales = [scale_1_nets, scale_2_nets, scale_3_nets]
    fusenet = GlobNet(3 * 2 * emb, 24)
    exp_folder = os.path.join(user_dir, data_dir, 'Experiments', 'ExpScaleFused')
    model_path = os.path.join(exp_folder, f'UseThis')
    fusenet.load_state_dict(torch.load(os.path.join(model_path, "fuse_net.pth")))
    fusenet.to(device)
    scales = [scale_1_nets, scale_2_nets, scale_3_nets]


    for (pc, rgb, bbox_coords, object_scale, s_correction, image_full, sRT) in test_loader:
        data = [pc, rgb, bbox_coords]
        sRT = np.squeeze(sRT)

        # get predictions from models
        # MODEL 1
        m1_loss, m1_pred = eval(data, model_1_nets, test_dataset.resizer,
                                            nn.MSELoss(), device, scale_dims_m1, sRT, s_correction, object_scale.item())
        # now we will try to find R and t matrices
        # fix the prediction
        fixed_out = fix_mistake(m1_pred, sRT, s_correction, object_scale.item())
        # first create the initial box with the known scale - this will be the source
        bbox_3d = get_3d_bbox(object_scale.item(), shift=0)  # shape 3x8
        # now find the rotation and translation between out and bbox_3d
        m1_R, m1_T = get_transformation_terms(bbox_3d, fixed_out.numpy())
        # now we can use this for our metric

        #MODEL 2
        m2_loss, m2_pred =  eval(data, model_2_nets, test_dataset.resizer,
                                         nn.MSELoss(), device, scale_dims_m2, sRT, s_correction, object_scale.item())
        # now we will try to find R and t matrices
        # fix the prediction
        fixed_out = fix_mistake(m2_pred, sRT, s_correction, object_scale.item())
        # first create the initial box with the known scale - this will be the source
        bbox_3d = get_3d_bbox(object_scale.item(), shift=0)  # shape 3x8
        # now find the rotation and translation between out and bbox_3d
        m2_R, m2_T = get_transformation_terms(bbox_3d, fixed_out.numpy())
        # now we can use this for our metric

        # MODEL 3
        m3_loss, m3_pred = eval(data, model_3_nets, test_dataset.resizer,
                                            nn.MSELoss(), device, scale_dims_m3, sRT, s_correction, object_scale.item())
        # now we will try to find R and t matrices
        # fix the prediction
        fixed_out = fix_mistake(m3_pred, sRT, s_correction, object_scale.item())
        # first create the initial box with the known scale - this will be the source
        bbox_3d = get_3d_bbox(object_scale.item(), shift=0)  # shape 3x8
        # now find the rotation and translation between out and bbox_3d
        m3_R, m3_T = get_transformation_terms(bbox_3d, fixed_out.numpy())
        # now we can use this for our metric

        # MODEL FUSION
        fuse_loss, fuse_pred, fuse_R, fuse_T = eval_fuse(data, scales, fusenet,
                  test_dataset.resizer, nn.MSELoss(), device,
                  ((1, 1), (2, 3), (4, 4)), sRT, s_correction, object_scale.item())
        # now we will try to find R and t matrices
        # fix the prediction
        fixed_out = fix_mistake(fuse_pred, sRT, s_correction, object_scale.item())
        # first create the initial box with the known scale - this will be the source
        bbox_3d = get_3d_bbox(object_scale.item(), shift=0)  # shape 3x8
        # now find the rotation and translation between out and bbox_3d
        m4_R, m4_T = get_transformation_terms(bbox_3d, fixed_out.numpy())
        # now we can use this for our metric

        preds = [m1_pred, m2_pred, m3_pred, fuse_pred]
        visualize_bboxes(image_full, bbox_coords, preds, rgb, object_scale,
                         s_correction, sRT,
                         model_names=['Kernel: 1', 'Kernel: 6', 'Kernel: 16', 'Fusion of kernels'])

        # for checking if RT eestimation works fine
        # gt_srt = np.copy(sRT)
        # s_correction = s_correction.flatten().numpy()
        # gt_srt[0, 0:3] /= s_correction[0]
        # gt_srt[1, 0:3] /= s_correction[1]
        # gt_srt[2, 0:3] /= s_correction[2]
        # print("GT RT mat: ", gt_srt)
        # print("Found R: ", R)
        # print("Found T: ", T)

        # print("overall loss: ", overall_loss)
        # print("pred: ", pred)
        # print("R: ", R)
        # print("T: ", T)
        # s_correction = np.diag(s_correction.flatten())
        # print("s_correction: ", s_correction)
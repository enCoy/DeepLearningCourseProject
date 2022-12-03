import cv2
import pickle
import os
import numpy as np
from utils.utils import align_rotation, align_nocs_to_depth, get_3d_bbox, get_sRT_mat, transform_coordinates_3d, calculate_2d_projections
from utils.utils import load_depth, load_coord, load_mask, load_label, load_colored
from tqdm import tqdm
from dataset.dataset_preprocess import process_data
import glob
from matplotlib import pyplot as plt
import torch
# import scipy

def visualize_scene(image, object):
    """

    @param image: Scene to be visualized - tensor of shape (1,H,W,C)
    @param object: crop of object to be visualized - tensor of shape (1,C,H,W)
    """
    image = image.squeeze.cpu().numpy()
    object = object.squeeze().permute(1,2,0).cpu().numpy()

    fig = plt.figure(figsize=(10, 7))
    rows = 1
    columns = 2

    fig.add_subplot(rows, columns, 1)
    plt.imshow(image)

    fig.add_subplot(rows, columns, 2)
    plt.imshow(object)
    plt.show()
    plt.close(fig)

def visualize_bboxes(image, gt_bbox, pred_bbox, object):
    """

    @param image: full image containing the object - tensor of shape (1,H,W,C)
    @param gt_bbox: gt bbox for object - tensor of shape (1,24)
    @param pred_bbox: pred bbox for objects - tensor of shape (1,24)
    @param object: img crop of object we are drawing the bbox for - tensor of shape (1,C,H_crop,W_crop)
    """
    #Change everything from tensors to numpy for utils
    image = image.squeeze().cpu().numpy()
    gt_bbox = gt_bbox.cpu()
    pred_bbox = pred_bbox.cpu()
    object = object.squeeze().permute(1,2,0).cpu().numpy()

    intrinsics = np.array([[577.5, 0, 319.5], [0, 577.5, 239.5], [0, 0, 1]], dtype=np.float)

    gt_bbox = torch.reshape(gt_bbox, (3,8)).numpy() #Reshape from 1x24 to 3x8
    pred_bbox = torch.reshape(pred_bbox, (3,8)).numpy()

    gt_bbox = np.divide(gt_bbox, 1000)
    pred_bbox = np.divide(pred_bbox, 1000)

    gt_2D = calculate_2d_projections(gt_bbox, intrinsics)
    gt_img = draw_bboxes(np.copy(image), gt_2D, (0, 255, 0))

    fig = plt.figure(figsize=(10, 7))
    rows = 1
    columns = 3

    fig.add_subplot(rows, columns, 1) #visualize the full image with the gt bbox
    plt.imshow(gt_img)
    plt.title("Ground truth")

    pred_2D = calculate_2d_projections(pred_bbox, intrinsics)
    pred_img = draw_bboxes(image, pred_2D, (0,255,0))

    fig.add_subplot(rows, columns, 2) #show the predicted bbox
    plt.imshow(pred_img)
    plt.title("Predicted")

    fig.add_subplot(rows, columns, 3) #show the object
    plt.imshow(object)
    plt.title("Object ")


    plt.show()



def draw_bboxes(img, img_pts, color):
    img_pts = np.int32(img_pts).reshape(-1, 2)
    # draw ground layer in darker color
    colors = [(255, 0, 0),
              (255, 0, 128),
              (255, 0, 255),
              (18, 199, 199),
              (137, 0, 255),
              (0, 0, 255),
              (3, 105, 165),
              (0, 205, 255),
              (0, 255, 0),
              (125, 51, 51),
              (255, 162, 0),
              (0, 0, 0)
              ]

    color_ground = (int(color[0]*0.3), int(color[1]*0.3), int(color[2]*0.3))
    counter = 0
    for i, j in zip([4, 5, 6, 7], [5, 7, 4, 6]):
        # img = cv2.line(img, tuple(img_pts[i]), tuple(img_pts[j]), color_ground, 2)
        img = cv2.line(img, tuple(img_pts[i]), tuple(img_pts[j]), colors[counter], 2)
        counter +=1
    # draw pillars in minor darker color
    color_pillar = (int(color[0]*0.6), int(color[1]*0.6), int(color[2]*0.6))
    for i, j in zip(range(4), range(4, 8)):
        # img = cv2.line(img, tuple(img_pts[i]), tuple(img_pts[j]), color_pillar, 2)
        img = cv2.line(img, tuple(img_pts[i]), tuple(img_pts[j]), colors[counter], 2)
        counter += 1
    # draw top layer in original color
    for i, j in zip([0, 1, 2, 3], [1, 3, 0, 2]):
        # img = cv2.line(img, tuple(img_pts[i]), tuple(img_pts[j]), color, 2)
        img = cv2.line(img, tuple(img_pts[i]), tuple(img_pts[j]), colors[counter], 2)
        counter += 1
    return img

def draw_detections_gt(img, out_dir, img_id, intrinsics,
                    gt_sRT, gt_size, gt_class_ids):
    # img is the colored image data
    # out_dir is output_directory
    # img_id is the 4digit number in the name of the image
    # intrinsics is camera intrinsics matrix
    # gt_sRT is 4x4 Scale x Rotation x Translation transformation matrix - dimension NumClassIDs x 4x4
    # gt size is probably the scale part of sRT - dimension NumClassIDs x3x1
    # gt class
    out_path = os.path.join(out_dir, '{}_gt_boxes.png'.format(img_id))
    # for the moment let's assume that gt_size is just the scale part of sRT
    for i in range(gt_sRT.shape[0]):
        if gt_class_ids[i] in [1, 2, 4]:
            sRT = align_rotation(gt_sRT[i, :, :])
        else:
            sRT = gt_sRT[i, :, :]
        # based on the scale of the bounding box, construct a bounding box at the origin basically
        bbox_3d = get_3d_bbox(gt_size[i, :], shift=0)
        # scale rotate and translate that bounding box
        transformed_bbox_3d = transform_coordinates_3d(bbox_3d, sRT)
        # project that bounding box to 2d from 3d
        projected_bbox = calculate_2d_projections(transformed_bbox_3d, intrinsics)
        img = draw_bboxes(img, projected_bbox, (0, 255, 0))
    cv2.imwrite(out_path, img)


def generate_gt_3d_boxes(data_dir, data_name='train'):
    # creates label pickle files for training data set and
    intrinsics = np.array([[591.0125, 0, 322.525], [0, 590.16775, 244.11084], [0, 0, 1]])
    real_train = open(os.path.join(data_dir, f'Real/{data_name}_list_all.txt')).read().splitlines()

    # scale factors for all instances
    scale_factors = {}
    path_to_size = glob.glob(os.path.join(data_dir, f'obj_models/real_{data_name}', '*_norm.txt'))
    for inst_path in sorted(path_to_size):
        instance = os.path.basename(inst_path).split('.')[0]
        bbox_dims = np.loadtxt(inst_path)
        scale_factors[instance] = bbox_dims / np.linalg.norm(bbox_dims)

    valid_img_list = []
    for img_path in tqdm(real_train):
        # last 5 characters are for the index of the image
        output_dir = os.path.join(data_dir, 'Real', img_path[:-5], 'gt_bboxes')
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        img_full_path = os.path.join(data_dir, 'Real', img_path)
        all_exist = os.path.exists(img_full_path + '_color.png') and \
                    os.path.exists(img_full_path + '_coord.png') and \
                    os.path.exists(img_full_path + '_depth.png') and \
                    os.path.exists(img_full_path + '_mask.png') and \
                    os.path.exists(img_full_path + '_meta.txt')

        if not all_exist:
            continue


        colored, masks, coords, class_ids, instance_ids, model_list, bboxes, depth = process_data(img_full_path)
        if instance_ids is None:
            continue

        # compute pose
        num_insts = len(class_ids)
        scales = np.zeros((num_insts, 3))
        rotations = np.zeros((num_insts, 3, 3))
        translations = np.zeros((num_insts, 3))

        gt_sRT = np.zeros((num_insts, 4, 4))
        gt_size = np.zeros((num_insts, 3))
        for i in range(num_insts):
            s = scale_factors[model_list[i]]
            mask = masks[:, :, i]
            idxs = np.where(mask)
            coord = coords[:, :, i, :]
            # take the masked points and scale them
            coord_pts = s * (coord[idxs[0], idxs[1], :] - 0.5)
            coord_pts = coord_pts[:, :, None]
            img_pts = np.array([idxs[1], idxs[0]]).transpose()
            img_pts = img_pts[:, :, None].astype(float)
            distCoeffs = np.zeros((4, 1))  # no distoration
            retval, rvec, tvec = cv2.solvePnP(coord_pts, img_pts, intrinsics, distCoeffs)
            assert retval
            R, _ = cv2.Rodrigues(rvec)
            T = np.squeeze(tvec)

            scales[i] = s
            rotations[i] = R
            translations[i] = T

            # get sRT matrix
            gt_sRT[i, :, :] = get_sRT_mat(s, R, T)
            gt_size[i, :] = s


        # draw the ground truth boxed image
        img = load_colored(img_full_path)
        img_id = img_path[-4:]
        draw_detections_gt(img, output_dir, img_id, intrinsics,
                           gt_sRT, gt_size, gt_class_ids=instance_ids)

def generate_gt_3d_boxes_camera(data_dir, data_name='val'):
    # creates label pickle files for training data set and
    intrinsics = np.array([[577.5, 0, 319.5], [0, 577.5, 239.5], [0, 0, 1]],
                dtype=np.float)
    camera_train = open(os.path.join(data_dir, f'CAMERA/{data_name}_list_all.txt')).read().splitlines()

    for img_path in tqdm(camera_train):
        # last 5 characters are for the index of the image
        output_dir = os.path.join(data_dir, 'CAMERA', img_path[:-5], 'gt_bboxes')
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        img_full_path = os.path.join(data_dir, 'CAMERA', img_path)
        all_exist = os.path.exists(img_full_path + '_color.png') and \
                    os.path.exists(img_full_path + '_coord.png') and \
                    os.path.exists(img_full_path + '_depth.png') and \
                    os.path.exists(img_full_path + '_mask.png') and \
                    os.path.exists(img_full_path + '_meta.txt')
        if not all_exist:
            continue

        # use meta.txt for scale factors
        # scale factors for all instances
        scale_factors = {}
        meta_path = img_full_path + '_meta.txt'
        with open(meta_path, 'r') as f:
            i = 0
            for line in f:
                # lets say for image 0000 our line is 1 6 mug2_scene3_norm
                line_info = line.strip().split(' ')  # ['1', '6', 'mug2_scene3_norm']
                inst_id = int(line_info[0])  # 1 in this case
                cls_id = int(line_info[1])  # 6 in this case
                # background objects and non-existing objects
                if cls_id == 0:
                    continue
                model_folder_id = line_info[2]
                model_id = line_info[3]
                bbox_txt_file_name = os.path.join(data_dir, 'obj_models', 'val', model_folder_id, model_id, 'bbox.txt')
                bbox_dims = np.loadtxt(bbox_txt_file_name)[0, :]
                scale_factors[model_id] = bbox_dims / np.linalg.norm(bbox_dims)

        colored, masks, coords, class_ids, instance_ids, model_list, bboxes, depth = process_data(img_full_path)
        if instance_ids is None:
            continue
        # Umeyama alignment of GT NOCS map with depth image
        scales, rotations, translations, error_messages, _ = \
            align_nocs_to_depth(masks, coords, depth, intrinsics, instance_ids, img_path)
        if error_messages:
            continue

        # compute pose
        num_insts = len(class_ids)
        gt_sRT = np.zeros((num_insts, 4, 4))
        gt_size = np.zeros((num_insts, 3))
        for i in range(num_insts):
            s = scale_factors[model_list[i]]
            R = rotations[i]
            T = translations[i]

            # get size matrix
            gt_sRT[i, :, :] = get_sRT_mat(s, R, T)
            gt_size[i, :] = scales[i]


        # draw the ground truth boxed image
        img = load_colored(img_full_path)
        img_id = img_path[-4:]
        draw_detections_gt(img, output_dir, img_id, intrinsics,
                           gt_sRT, gt_size, gt_class_ids=instance_ids)




if __name__ == "__main__":
    user_dir = r"C:\Users\Cem Okan"   # change this on your computer
    data_dir = user_dir + r"\Dropbox (GaTech)\deep_learning_data"
    # annotate_real_train(data_dir)
    generate_gt_3d_boxes_camera(data_dir, data_name='val')
    # generate_gt_3d_boxes(data_dir, data_name='test')
    # generate_gt_3d_boxes(data_dir, data_name='train')

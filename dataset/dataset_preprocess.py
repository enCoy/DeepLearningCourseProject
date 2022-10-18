import os
import glob
import cv2
import numpy as np
import math
import pickle
import _pickle as cPickle
from tqdm import tqdm
from utils.utils import load_depth, load_label, load_mask, load_coord, load_colored
from utils.utils import get_sRT_mat, align_rotation, get_3d_bbox, transform_coordinates_3d, calculate_2d_projections

def create_img_list(data_dir):
    # creates a txt file listing paths from the directory where images are
    for subset in ['train', 'test']:
        img_list = []
        img_dir = os.path.join(data_dir, 'Real', subset)
        folder_list = [name for name in sorted(os.listdir(img_dir)) if os.path.isdir(os.path.join(img_dir, name))]
        for folder in folder_list:
            img_paths = glob.glob(os.path.join(img_dir, folder, '*_color.png'))
            img_paths = sorted(img_paths)
            for img_full_path in img_paths:
                img_name = os.path.basename(img_full_path)
                img_ind = img_name.split('_')[0]
                img_path = os.path.join(subset, folder, img_ind)
                img_list.append(img_path)
        with open(os.path.join(data_dir, 'Real', subset + '_list_all.txt'), 'w') as f:
            for img_path in img_list:
                f.write("%s\n" % img_path)
    print('Write all data paths to file done!')



def process_data(img_path):

    depth = load_depth(img_path)
    mask = load_mask(img_path)
    coord_map = load_coord(img_path)

    all_inst_ids = sorted(list(np.unique(mask)))
    assert all_inst_ids[-1]  # remove background
    del all_inst_ids[-1]  # remove background
    num_all_inst = len(all_inst_ids)
    h, w = mask.shape

    class_ids = []
    instance_ids = []
    model_list = []
    # create an array that will hold mask for every instance
    masks = np.zeros([h, w, num_all_inst], dtype=np.uint8)
    coords = np.zeros((h, w, num_all_inst, 3), dtype=np.float32)
    bboxes = np.zeros((num_all_inst, 4), dtype=np.int32)

    meta_path = img_path + '_meta.txt'
    with open(meta_path, 'r') as f:
        i = 0

        for line in f:
            # lets say for image 0000 our line is 1 6 mug2_scene3_norm
            line_info = line.strip().split(' ')  # ['1', '6', 'mug2_scene3_norm']
            inst_id = int(line_info[0])  # 1 in this case
            cls_id = int(line_info[1])  # 6 in this case
            # background objects and non-existing objects
            if cls_id == 0 or (inst_id not in all_inst_ids):
                continue
            if len(line_info) == 3:  # Real275
                model_id = line_info[2]
            else:  # camera dataset, WE ARE NOT USING THIS ONE
                model_id = line_info[3]

            # process foreground objects
            # take the mask of that instance
            inst_mask = np.equal(mask, inst_id)
            # bounding box
            # find the x coordinates whose value equals to true in mask
            horizontal_indices = np.where(np.any(inst_mask, axis=0))[0]
            # find the x coordinates whose value equals to true in mask
            vertical_indices = np.where(np.any(inst_mask, axis=1))[0]

            x1, x2 = horizontal_indices[[0, -1]]
            y1, y2 = vertical_indices[[0, -1]]
            # they say x2 and y2 should not be part of the box - increment by 1
            x2 += 1
            y2 += 1

            # not enough valid depth observation
            final_mask = np.logical_and(inst_mask, depth > 0)
            if np.sum(final_mask) < 64:
                continue
            class_ids.append(cls_id)
            instance_ids.append(inst_id)
            model_list.append(model_id)
            masks[:, :, i] = inst_mask
            coords[:, :, i, :] = np.multiply(coord_map, np.expand_dims(inst_mask, axis=-1))
            bboxes[i] = np.array([y1, x1, y2, x2])
            i += 1
    # no valid foreground objects
    if i ==0:
        return None, None, None, None, None, None

    masks = masks[:, :, :i]
    coords = np.clip(coords[:, :, :i, :], 0, 1)
    bboxes = bboxes[:i, :]

    return masks, coords, class_ids, instance_ids, model_list, bboxes, depth

def annotate_real_train(data_dir):
    # creates label pickle files for training data set and
    real_train = open(os.path.join(data_dir, 'Real/train_list_all.txt')).read().splitlines()
    intrinsics = np.array([[591.0125, 0, 322.525], [0, 590.16775, 244.11084], [0, 0, 1]])
    # scale factors for all instances
    scale_factors = {}
    path_to_size = glob.glob(os.path.join(data_dir, 'obj_models/real_train', '*_norm.txt'))
    for inst_path in sorted(path_to_size):
        instance = os.path.basename(inst_path).split('.')[0]
        bbox_dims = np.loadtxt(inst_path)
        scale_factors[instance] = bbox_dims / np.linalg.norm(bbox_dims)

    valid_img_list = []
    for img_path in tqdm(real_train):
        img_full_path = os.path.join(data_dir, 'Real', img_path)
        all_exist = os.path.exists(img_full_path + '_color.png') and \
                    os.path.exists(img_full_path + '_coord.png') and \
                    os.path.exists(img_full_path + '_depth.png') and \
                    os.path.exists(img_full_path + '_mask.png') and \
                    os.path.exists(img_full_path + '_meta.txt')
        if not all_exist:
            continue
        masks, coords, class_ids, instance_ids, model_list, bboxes, depth = process_data(img_full_path)
        if instance_ids is None:
            continue

        # compute pose
        num_insts = len(class_ids)
        scales = np.zeros((num_insts, 3))
        rotations = np.zeros((num_insts, 3, 3))
        translations = np.zeros((num_insts, 3))
        projected_bboxes = np.zeros((num_insts, 8, 2))
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
            # # re-label for mug category
            # if class_ids[i] == 6:
            #     T0 = mug_meta[model_list[i]][0]
            #     s0 = mug_meta[model_list[i]][1]
            #     T = T - s * R @ T0
            #     s = s / s0
            scales[i] = s
            rotations[i] = R
            translations[i] = T

            # get the sRT matrix for that object
            sRT = get_sRT_mat(s, R, T)
            if class_ids[i] in [1, 2, 4]:
                sRT = align_rotation(sRT)
            # based on the scale of the bounding box, construct a bounding box at the origin basically
            bbox_3d = get_3d_bbox(s, shift=0)
            # scale rotate and translate that bounding box
            transformed_bbox_3d = transform_coordinates_3d(bbox_3d, sRT)
            # project that bounding box to 2d from 3d
            projected_bbox = calculate_2d_projections(transformed_bbox_3d, intrinsics)
            # projected bbox has shape 8x2
            projected_bboxes[i, :, :] = projected_bbox

            # write results
        gts = {}
        gts['class_ids'] = class_ids  # int list, 1 to 6
        gts['bboxes'] = bboxes  # np.array, [[y1, x1, y2, x2], ...]
        gts['scales'] = scales.astype(np.float32)  # np.array, scale factor from NOCS model to depth observation
        gts['rotations'] = rotations.astype(np.float32)  # np.array, R
        gts['translations'] = translations.astype(np.float32)  # np.array, T
        gts['instance_ids'] = instance_ids  # int list, start from 1
        gts['model_list'] = model_list  # str list, model id/name
        gts['projected_bbox'] = projected_bboxes.astype(np.float32)
        with open(img_full_path + '_label.pkl', 'wb') as f:
            cPickle.dump(gts, f)
        valid_img_list.append(img_path)
        # write valid img list to file

    # this only contains the valid images
    with open(os.path.join(data_dir, 'Real/train_list.txt'), 'w') as f:
        for img_path in valid_img_list:
            f.write("%s\n" % img_path)

def construct_data_lists(data_dir):
    # creates lists that include the path to color, coord, depth, mask and labels
    real_train = open(os.path.join(data_dir, 'Real/train_list_all.txt')).read().splitlines()

    image_list = []

    for img_path in tqdm(real_train):
        img_full_path = os.path.join(data_dir, 'Real', img_path)
        all_exist = os.path.exists(img_full_path + '_color.png') and \
                    os.path.exists(img_full_path + '_coord.png') and \
                    os.path.exists(img_full_path + '_depth.png') and \
                    os.path.exists(img_full_path + '_mask.png') and \
                    os.path.exists(img_full_path + '_label.pkl')
        if not all_exist:
            continue
        else:  # if they exist
            image_list.append(img_path)

    # save the list
    with open(os.path.join(data_dir, 'Real', 'data_paths_list.pickle'), 'wb') as handle:
        pickle.dump(image_list, handle, protocol=pickle.HIGHEST_PROTOCOL)





if __name__ == "__main__":
    data_dir = r"C:\Users\Cem Okan\Desktop\deep learning\project\data"
    create_img_list(data_dir)
    construct_data_lists(data_dir)
    # annotate_real_train(data_dir)
    with open(os.path.join(r'C:\Users\Cem Okan\Desktop\deep learning\project\data', 'Real', 'data_paths_list.pickle'), 'rb') as handle:
        b = pickle.load(handle)



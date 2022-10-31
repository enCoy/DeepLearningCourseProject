import os
import glob
import cv2
import numpy as np
import math
import pickle
import _pickle as cPickle
from random import sample
import random
from tqdm import tqdm
from utils.utils import load_depth, load_label, load_mask, load_coord, load_colored, align_nocs_to_depth
from utils.utils import get_sRT_mat, align_rotation, get_3d_bbox, transform_coordinates_3d, calculate_2d_projections

def create_img_list(data_dir):
    # CAMERA dataset
    for subset in ['data']:
        img_dir = os.path.join(data_dir, 'CAMERA', subset)
        folder_list = [name for name in os.listdir(img_dir) if os.path.isdir(os.path.join(img_dir, name))]
        print(f"There are {len(folder_list)} folders in CAMERA data set")
        val_number_of_folders = int(len(folder_list) * 0.1)
        test_number_of_folders = int(len(folder_list) * 0.1)
        print(f"10% are reserved for validation which corresponds to {val_number_of_folders} folders")
        print(f"10% are reserved for testing which corresponds to {test_number_of_folders} folders")
        print(f"80% are reserved for training which corresponds to {len(folder_list) - val_number_of_folders - test_number_of_folders} folders")
        val_folders = sample(folder_list, val_number_of_folders)
        train_folders = list(set(folder_list) - set(val_folders))
        test_folders = sample(train_folders, test_number_of_folders)
        train_folders = list(set(train_folders) - set(test_folders))
        train_img_list = []
        val_img_list = []
        test_img_list = []

        for i in range(10 * len(folder_list)):
            folder_id = int(i) // 10
            img_id = int(i) % 10
            img_path = os.path.join(subset, '{:05d}'.format(folder_id), '{:04d}'.format(img_id))
            folder_name = '{:05d}'.format(folder_id)
            if folder_name in train_folders:
                train_img_list.append(img_path)
            elif folder_name in val_folders:
                val_img_list.append(img_path)
            elif folder_name in test_folders:
                test_img_list.append(img_path)
        # save files
        with open(os.path.join(data_dir, 'CAMERA', 'train' + '_list_all.txt'), 'w') as f:
            for img_path in train_img_list:
                f.write("%s\n" % img_path)
        with open(os.path.join(data_dir, 'CAMERA', 'val' + '_list_all.txt'), 'w') as f:
            for img_path in val_img_list:
                f.write("%s\n" % img_path)
        with open(os.path.join(data_dir, 'CAMERA', 'test' + '_list_all.txt'), 'w') as f:
            for img_path in test_img_list:
                f.write("%s\n" % img_path)

    # creates a txt file listing paths from the directory where images are
    print("Note: In Real275-train, Scene 2 and Scene 6 are reserved for validation")
    for subset in ['train', 'test']:
        img_list = []
        img_list_for_val = []
        img_dir = os.path.join(data_dir, 'Real', subset)
        folder_list = [name for name in sorted(os.listdir(img_dir)) if os.path.isdir(os.path.join(img_dir, name))]
        for folder in folder_list:
            img_paths = glob.glob(os.path.join(img_dir, folder, '*_color.png'))
            img_paths = sorted(img_paths)
            for img_full_path in img_paths:
                img_name = os.path.basename(img_full_path)
                img_ind = img_name.split('_')[0]
                img_path = os.path.join(subset, folder, img_ind)
                if subset =='train':
                    if (folder == 'scene_2') or (folder == 'scene_6'):
                        img_list_for_val.append(img_path)
                    else:
                        img_list.append(img_path)
                else:
                    img_list.append(img_path)
        with open(os.path.join(data_dir, 'Real', subset + '_list_all.txt'), 'w') as f:
            for img_path in img_list:
                f.write("%s\n" % img_path)
        # save validation too
        if subset =='train':
            with open(os.path.join(data_dir, 'Real', 'val' + '_list_all.txt'), 'w') as f:
                for img_path in img_list_for_val:
                    f.write("%s\n" % img_path)
    print('Write all data paths to file done!')

def process_data(img_path):

    img_colored = load_colored(img_path)
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
    colored_data = np.zeros((h, w, num_all_inst, 3), dtype=np.float32)

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
            # remove one mug instance in CAMERA train due to improper model
            # if model_id == 'b9be7cfe653740eb7633a2dd89cec754':
            #     continue
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
            # object occupies full image, rendering error, happens in CAMERA dataset
            if np.any(np.logical_or((x2 - x1) > 600, (y2 - y1) > 440)):
                return None, None, None, None, None, None, None, None
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
        return None, None, None, None, None, None, None, None

    masks = masks[:, :, :i]
    coords = np.clip(coords[:, :, :i, :], 0, 1)
    bboxes = bboxes[:i, :]
    return img_colored, masks, coords, class_ids, instance_ids, model_list, bboxes, depth

def annotate_real(data_dir, data_name='train', output_dir_name='train_data'):
    # creates label pickle files for training data set and
    real_train = open(os.path.join(data_dir, f'Real/{data_name}_list_all.txt')).read().splitlines()
    intrinsics = np.array([[591.0125, 0, 322.525], [0, 590.16775, 244.11084], [0, 0, 1]])
    # scale factors for all instances
    scale_factors = {}
    if data_name == 'val':
        path_to_size = glob.glob(os.path.join(data_dir, f'obj_models/real_train', '*_norm.txt'))
    else:
        path_to_size = glob.glob(os.path.join(data_dir, f'obj_models/real_{data_name}', '*_norm.txt'))
    for inst_path in sorted(path_to_size):
        instance = os.path.basename(inst_path).split('.')[0]
        bbox_dims = np.loadtxt(inst_path)
        scale_factors[instance] = bbox_dims / np.linalg.norm(bbox_dims)

    valid_img_list = []

    # create folder for individual ground truths data
    ground_truth_folder = os.path.join(data_dir, f'gts/{output_dir_name}')
    if not os.path.isdir(ground_truth_folder):
        os.makedirs(ground_truth_folder)

    img_counter = 0
    for img_path in tqdm(real_train):
        img_counter += 1
        img_full_path = os.path.join(data_dir, 'Real', img_path)
        all_exist = os.path.exists(img_full_path + '_color.png') and \
                    os.path.exists(img_full_path + '_coord.png') and \
                    os.path.exists(img_full_path + '_depth.png') and \
                    os.path.exists(img_full_path + '_mask.png') and \
                    os.path.exists(img_full_path + '_meta.txt')
        if not all_exist:
            continue
        colored, masks, coords, class_ids, instance_ids, model_list, bboxes, depth = process_data(img_full_path)
        # masks shape HxWxNumInstances
        # coords shape HxWxNumInstancesx3    3 comes from RGB
        # depth shape H x W
        if instance_ids is None:
            continue

        # compute pose
        num_insts = len(class_ids)
        scales = np.zeros((num_insts, 3))
        rotations = np.zeros((num_insts, 3, 3))
        translations = np.zeros((num_insts, 3))
        projected_bboxes = np.zeros((num_insts, 8, 2))


        for i in range(num_insts):
            gts_individual = {}

            # for each instance, we will need # corresponding depth data for the object
            # + translation, rotation, scale, bbox_3d coordinates
            s = scale_factors[model_list[i]]
            mask = masks[:, :, i]   # shape H x W
            idxs = np.where(mask)
            coord = coords[:, :, i, :]  # shape H x W x 3   3 is for RGB I think
            # take the masked points and scale them
            coord_pts = s * (coord[idxs[0], idxs[1], :] - 0.5)  # shape
            coord_pts = coord_pts[:, :, None]  # expands dimension basically
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

            gts_individual['class_id'] = class_ids[i]  # int list, 1 to 6
            # following is the label
            gts_individual['bbox_3d'] = transformed_bbox_3d
            gts_individual['projected_bbox'] = projected_bbox.astype(np.float32)
            gts_individual['scale'] = s.astype(np.float32)  # np.array, scale factor from NOCS model to depth observation
            gts_individual['rotation'] = R.astype(np.float32)  # np.array, R
            gts_individual['translation'] = T.astype(np.float32)  # np.array, T
            gts_individual['instance_id'] = instance_ids[i]  # int list, start from 1
            # include depth data, x and y
            x_depth = idxs[0]
            y_depth = idxs[1]
            z_depth = depth[idxs[0], idxs[1]]
            rgb_masked = colored[idxs[0], idxs[1], :]
            # followings will be features
            gts_individual['x_depth'] = x_depth
            gts_individual['y_depth'] = y_depth
            gts_individual['z_depth'] = z_depth
            gts_individual['rgb'] = rgb_masked

            with open(ground_truth_folder + f'/real_{img_counter}_inst_{instance_ids[i]}_label.pkl', 'wb') as f:
                cPickle.dump(gts_individual, f)


            # write results
        # write results - THIS WAS FOR SCENE BASED DATA LOADER - CURRENTLY WE DO NOT USE THIS
        # gts = {}
        # gts['class_ids'] = class_ids  # int list, 1 to 6
        # gts['bboxes'] = bboxes  # np.array, [[y1, x1, y2, x2], ...]
        # gts['scales'] = scales.astype(np.float32)  # np.array, scale factor from NOCS model to depth observation
        # gts['rotations'] = rotations.astype(np.float32)  # np.array, R
        # gts['translations'] = translations.astype(np.float32)  # np.array, T
        # gts['instance_ids'] = instance_ids  # int list, start from 1
        # gts['model_list'] = model_list  # str list, model id/name
        # gts['projected_bbox'] = projected_bboxes.astype(np.float32)
        # with open(img_full_path + '_label.pkl', 'wb') as f:
        #     cPickle.dump(gts, f)
        valid_img_list.append(img_path)
        # write valid img list to file

    # this only contains the valid images
    with open(os.path.join(data_dir, f'Real/{data_name}_list.txt'), 'w') as f:
        for img_path in valid_img_list:
            f.write("%s\n" % img_path)


def annotate_camera(data_dir, data_name='val', output_dir_name='train_data'):
    """ Generate gt labels for CAMERA train data. """
    camera_train = open(os.path.join(data_dir, 'CAMERA', f'{data_name}_list_all.txt')).read().splitlines()
    intrinsics = np.array([[577.5, 0, 319.5], [0, 577.5, 239.5], [0, 0, 1]])
    # meta info for re-label mug category
    # with open(os.path.join(data_dir, 'obj_models/mug_meta.pkl'), 'rb') as f:
    #     mug_meta = cPickle.load(f)

    valid_img_list = []
    img_counter = 0
    for img_path in tqdm(camera_train):
        img_counter+=1
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
                bbox_txt_file_name = os.path.join(data_dir, 'obj_models', 'camera', model_folder_id, model_id,
                                                  'bbox.txt')
                bbox_dims = np.loadtxt(bbox_txt_file_name)[0, :]
                scale_factors[model_id] = bbox_dims / np.linalg.norm(bbox_dims)

        colored, masks, coords, class_ids, instance_ids, model_list, bboxes, depth = process_data(img_full_path)
        # masks have shape H x W x NumInsts - different from real dataset
        # coords have shape H x W x NumInsts x 3 (rgb)
        # bboxes shape NumInsts x 4
        # depth shape H x W
        if instance_ids is None:
            continue
        # Umeyama alignment of GT NOCS map with depth image
        scales, rotations, translations, error_messages, _ = \
            align_nocs_to_depth(masks, coords, depth, intrinsics, instance_ids, img_path)
        if error_messages:
            continue

        ground_truth_folder = os.path.join(data_dir, f'gts/{output_dir_name}')
        if not os.path.isdir(ground_truth_folder):
            os.makedirs(ground_truth_folder)

        num_insts = len(class_ids)
        projected_bboxes = np.zeros((num_insts, 8, 2))
        for i in range(num_insts):
            gts_individual = {}
        #     # for each instance, we will need # corresponding depth data for the object
        #     # + translation, rotation, scale, bbox_3d coordinates
            s = scale_factors[model_list[i]]
            mask = masks[:, :, i]   # shape H x W
            idxs = np.where(mask)
            coord = coords[:, :, i, :]  # shape H x W x 3   3 is for RGB I think

            R = rotations[i]
            T = translations[i]
            # get the sRT matrix for that object
            sRT = get_sRT_mat(s, R, T)
            # based on the scale of the bounding box, construct a bounding box at the origin basically
            bbox_3d = get_3d_bbox(s, shift=0)
            # scale rotate and translate that bounding box
            transformed_bbox_3d = transform_coordinates_3d(bbox_3d, sRT)
            # project that bounding box to 2d from 3d
            projected_bbox = calculate_2d_projections(transformed_bbox_3d, intrinsics)
            # projected bbox has shape 8x2
            projected_bboxes[i, :, :] = projected_bbox
        #
            gts_individual['class_id'] = class_ids[i]  # int list, 1 to 6
            gts_individual['bbox_3d'] = transformed_bbox_3d
            gts_individual['projected_bbox'] = projected_bbox.astype(np.float32)
            gts_individual['scale'] = scales[i].astype(np.float32)  # np.array, scale factor from NOCS model to depth observation
            gts_individual['rotation'] = rotations[i].astype(np.float32)  # np.array, R
            gts_individual['translation'] = translations[i].astype(np.float32)  # np.array, T
            gts_individual['instance_id'] = instance_ids[i]  # int list, start from 1
            # include depth data, x and y
            x_depth = idxs[0]
            y_depth = idxs[1]
            z_depth = depth[idxs[0], idxs[1]]
            gts_individual['x_depth'] = x_depth
            gts_individual['y_depth'] = y_depth
            gts_individual['z_depth'] = z_depth
            rgb_masked = colored[idxs[0], idxs[1], :]
            gts_individual['rgb'] = rgb_masked

            with open(ground_truth_folder + f'/camera_{img_counter}_inst_{instance_ids[i]}_label.pkl', 'wb') as f:
                cPickle.dump(gts_individual, f)

        # write results - THIS WAS FOR SCENE BASED DATA LOADER - CURRENTLY WE DO NOT USE THIS

        # gts = {}
        # gts['class_ids'] = class_ids    # int list, 1 to 6
        # gts['bboxes'] = bboxes  # np.array, [[y1, x1, y2, x2], ...]
        # gts['scales'] = scales.astype(np.float32)  # np.array, scale factor from NOCS model to depth observation
        # gts['rotations'] = rotations.astype(np.float32)    # np.array, R
        # gts['translations'] = translations.astype(np.float32)  # np.array, T
        # gts['instance_ids'] = instance_ids  # int list, start from 1
        # gts['model_list'] = model_list  # str list, model id/name
        # with open(img_full_path + '_label.pkl', 'wb') as f:
        #     cPickle.dump(gts, f)

        valid_img_list.append(img_path)
    # write valid img list to file
    # with open(os.path.join(data_dir, f'CAMERA/{data_name}_list.txt'), 'w') as f:
    #     for img_path in valid_img_list:
    #         f.write("%s\n" % img_path)

# def construct_data_lists(data_dir, data_name='train'):
#     # creates lists that include the path to color, coord, depth, mask and labels
#     real_train = open(os.path.join(data_dir, f'Real/{data_name}_list_all.txt')).read().splitlines()
#
#     image_list = []
#
#     for img_path in tqdm(real_train):
#         img_full_path = os.path.join(data_dir, 'Real', img_path)
#         all_exist = os.path.exists(img_full_path + '_color.png') and \
#                     os.path.exists(img_full_path + '_coord.png') and \
#                     os.path.exists(img_full_path + '_depth.png') and \
#                     os.path.exists(img_full_path + '_mask.png') and \
#                     os.path.exists(img_full_path + '_label.pkl')
#         if not all_exist:
#             continue
#         else:  # if they exist
#             image_list.append(img_path)
#
#     # # save the list
#     # with open(os.path.join(data_dir, 'Real', f'{data_name}_data_paths_list.pickle'), 'wb') as handle:
#     #     pickle.dump(image_list, handle, protocol=pickle.HIGHEST_PROTOCOL)





if __name__ == "__main__":
    random.seed(16)
    user_dir = r"C:\Users\Cem Okan"  # change this on your computer
    data_dir = user_dir + r"\Dropbox (GaTech)\deep_learning_data"
    # following function constructs the txt files that stores training, validation and test folder names
    create_img_list(data_dir)
    # construct_data_lists(data_dir, data_name='train')  # constructs a list that contains paths of
    # construct_data_lists(data_dir, data_name='test')

    annotate_real(data_dir, data_name='val', output_dir_name='val_data')
    annotate_camera(data_dir, data_name='val', output_dir_name='val_data')
    annotate_real(data_dir, data_name='test', output_dir_name='test_data')
    annotate_camera(data_dir, data_name='test', output_dir_name='test_data')
    annotate_real(data_dir, data_name='train', output_dir_name='train_data')
    annotate_camera(data_dir, data_name='train', output_dir_name='train_data')





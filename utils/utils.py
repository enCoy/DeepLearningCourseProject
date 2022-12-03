import numpy as np
import math
import cv2
import pickle
import torch
from itertools import combinations
import time
intrinsics = np.array([[577.5, 0, 319.5], [0., 577.5, 239.5], [0., 0., 1.]])
Camera_intrinsics = np.array([[577.5, 0, 319.5], [0, 577.5, 239.5], [0, 0, 1]],
                                          dtype=np.float)  # [fx, fy, cx, cy]


# source: https://github.com/mentian/object-deformnet/blob/master/preprocess/pose_data.py

def get_intrinsics(dataset_type):
    if dataset_type == 'CAMERA':
        return Camera_intrinsics
    elif dataset_type == 'Real':
        return intrinsics

def get_sRT_mat(scale, rotation, translation):
    # scale is a scaler
    # rotation is a 3x3 matrix
    # translation is a 3x1 vector
    translation_mat = np.identity(4)
    translation_mat[0:3, -1] = translation


    scale_mat = np.identity(4)
    scale_mat[0, 0] = scale[0]
    scale_mat[1, 1] = scale[1]
    scale_mat[2, 2] = scale[2]


    rotation_mat = np.identity(4)
    rotation_mat[0:3, 0:3] = rotation


    sRT = np.matmul(translation_mat, np.matmul(rotation_mat, scale_mat))
    return sRT  # 4x4 transformation matrix

def get_3d_bbox(scale, shift=0):
    """
    Input:
        scale: [3] or scalar
        shift: [3] or scalar
    Return
        bbox_3d: [3, N]
    """
    if hasattr(scale, "__iter__"):
        bbox_3d = np.array([[scale[0] / 2, +scale[1] / 2, scale[2] / 2],
                            [scale[0] / 2, +scale[1] / 2, -scale[2] / 2],
                            [-scale[0] / 2, +scale[1] / 2, scale[2] / 2],
                            [-scale[0] / 2, +scale[1] / 2, -scale[2] / 2],
                            [+scale[0] / 2, -scale[1] / 2, scale[2] / 2],
                            [+scale[0] / 2, -scale[1] / 2, -scale[2] / 2],
                            [-scale[0] / 2, -scale[1] / 2, scale[2] / 2],
                            [-scale[0] / 2, -scale[1] / 2, -scale[2] / 2]]) + shift
    else:
        bbox_3d = np.array([[scale / 2, +scale / 2, scale / 2],
                            [scale / 2, +scale / 2, -scale / 2],
                            [-scale / 2, +scale / 2, scale / 2],
                            [-scale / 2, +scale / 2, -scale / 2],
                            [+scale / 2, -scale / 2, scale / 2],
                            [+scale / 2, -scale / 2, -scale / 2],
                            [-scale / 2, -scale / 2, scale / 2],
                            [-scale / 2, -scale / 2, -scale / 2]]) + shift

    bbox_3d = bbox_3d.transpose()
    return bbox_3d

def transform_coordinates_3d(coordinates, sRT):
    """
    Args:
        coordinates: [3, N]
        sRT: [4, 4]
    Returns:
        new_coordinates: [3, N]
    """
    assert coordinates.shape[0] == 3
    coordinates = np.vstack([coordinates, np.ones((1, coordinates.shape[1]), dtype=np.float32)])
    new_coordinates = sRT @ coordinates
    new_coordinates = new_coordinates[:3, :] / new_coordinates[3, :]
    return new_coordinates

def calculate_2d_projections(coordinates_3d, intrinsics):
    """
    Args:
        coordinates_3d: [3, N]
        intrinsics: [3, 3]
    Returns:
        projected_coordinates: [N, 2]
    """
    projected_coordinates = intrinsics @ coordinates_3d
    projected_coordinates = projected_coordinates[:2, :] / projected_coordinates[2, :]
    projected_coordinates = projected_coordinates.transpose()
    projected_coordinates = np.array(projected_coordinates, dtype=np.int32)

    return projected_coordinates

def calculate_2d_projections_inv(projected_coordinates, intrinsics):
    # from [N, 2] to [3, N]
    projected_coordinates =  projected_coordinates.transpose()



def align_rotation(sRT):
    """ Align rotations for symmetric objects.
    Args:
        sRT: 4 x 4
    """
    s = np.cbrt(np.linalg.det(sRT[:3, :3]))
    R = sRT[:3, :3] / s
    T = sRT[:3, 3]

    theta_x = R[0, 0] + R[2, 2]
    theta_y = R[0, 2] - R[2, 0]
    r_norm = math.sqrt(theta_x**2 + theta_y**2)
    s_map = np.array([[theta_x/r_norm, 0.0, -theta_y/r_norm],
                      [0.0,            1.0,  0.0           ],
                      [theta_y/r_norm, 0.0,  theta_x/r_norm]])
    rotation = R @ s_map
    aligned_sRT = np.identity(4, dtype=np.float32)
    aligned_sRT[:3, :3] = s * rotation
    aligned_sRT[:3, 3] = T
    return aligned_sRT

def backproject(depth, intrinsics, instance_mask):
    """ Back-projection, use opencv camera coordinate frame.
    """
    cam_fx = intrinsics[0, 0]
    cam_fy = intrinsics[1, 1]
    cam_cx = intrinsics[0, 2]
    cam_cy = intrinsics[1, 2]

    non_zero_mask = (depth > 0)
    final_instance_mask = np.logical_and(instance_mask, non_zero_mask)
    idxs = np.where(final_instance_mask)

    z = depth[idxs[0], idxs[1]]
    x = (idxs[1] - cam_cx) * z / cam_fx
    y = (idxs[0] - cam_cy) * z / cam_fy
    pts = np.stack((x, y, z), axis=1)

    return pts, idxs

def estimateSimilarityTransform(source: np.array, target: np.array, verbose=False):
    """ Add RANSAC algorithm to account for outliers.
    """
    assert source.shape[0] == target.shape[0], 'Source and Target must have same number of points.'
    SourceHom = np.transpose(np.hstack([source, np.ones([source.shape[0], 1])]))
    TargetHom = np.transpose(np.hstack([target, np.ones([target.shape[0], 1])]))
    # Auto-parameter selection based on source heuristics
    # Assume source is object model or gt nocs map, which is of high quality
    SourceCentroid = np.mean(SourceHom[:3, :], axis=1)
    nPoints = SourceHom.shape[1]
    CenteredSource = SourceHom[:3, :] - np.tile(SourceCentroid, (nPoints, 1)).transpose()
    SourceDiameter = 2 * np.amax(np.linalg.norm(CenteredSource, axis=0))
    InlierT = SourceDiameter / 10.0  # 0.1 of source diameter
    maxIter = 128
    confidence = 0.99

    if verbose:
        print('Inlier threshold: ', InlierT)
        print('Max number of iterations: ', maxIter)

    BestInlierRatio = 0
    BestInlierIdx = np.arange(nPoints)
    for i in range(0, maxIter):
        # Pick 5 random (but corresponding) points from source and target
        RandIdx = np.random.randint(nPoints, size=5)
        Scale, _, _, OutTransform = estimateSimilarityUmeyama(SourceHom[:, RandIdx], TargetHom[:, RandIdx])
        PassThreshold = Scale * InlierT    # propagate inlier threshold to target scale
        Diff = TargetHom - np.matmul(OutTransform, SourceHom)
        ResidualVec = np.linalg.norm(Diff[:3, :], axis=0)
        InlierIdx = np.where(ResidualVec < PassThreshold)[0]
        nInliers = InlierIdx.shape[0]
        InlierRatio = nInliers / nPoints
        # update best hypothesis
        if InlierRatio > BestInlierRatio:
            BestInlierRatio = InlierRatio
            BestInlierIdx = InlierIdx
        if verbose:
            print('Iteration: ', i)
            print('Inlier ratio: ', BestInlierRatio)
        # early break
        if (1 - (1 - BestInlierRatio ** 5) ** i) > confidence:
            break

    if(BestInlierRatio < 0.1):
        print('[ WARN ] - Something is wrong. Small BestInlierRatio: ', BestInlierRatio)
        return None, None, None, None

    SourceInliersHom = SourceHom[:, BestInlierIdx]
    TargetInliersHom = TargetHom[:, BestInlierIdx]
    Scale, Rotation, Translation, OutTransform = estimateSimilarityUmeyama(SourceInliersHom, TargetInliersHom)

    if verbose:
        print('BestInlierRatio:', BestInlierRatio)
        print('Rotation:\n', Rotation)
        print('Translation:\n', Translation)
        print('Scale:', Scale)

    return Scale, Rotation, Translation, OutTransform

def estimateSimilarityUmeyama(SourceHom, TargetHom):
    # Copy of original paper is at: http://web.stanford.edu/class/cs273/refs/umeyama.pdf
    SourceCentroid = np.mean(SourceHom[:3, :], axis=1)
    TargetCentroid = np.mean(TargetHom[:3, :], axis=1)
    nPoints = SourceHom.shape[1]
    CenteredSource = SourceHom[:3, :] - np.tile(SourceCentroid, (nPoints, 1)).transpose()
    CenteredTarget = TargetHom[:3, :] - np.tile(TargetCentroid, (nPoints, 1)).transpose()
    CovMatrix = np.matmul(CenteredTarget, np.transpose(CenteredSource)) / nPoints
    if np.isnan(CovMatrix).any():
        print('nPoints:', nPoints)
        print(SourceHom.shape)
        print(TargetHom.shape)
        raise RuntimeError('There are NANs in the input.')

    U, D, Vh = np.linalg.svd(CovMatrix, full_matrices=True)
    d = (np.linalg.det(U) * np.linalg.det(Vh)) < 0.0
    if d:
        D[-1] = -D[-1]
        U[:, -1] = -U[:, -1]
    # rotation
    Rotation = np.matmul(U, Vh)
    # scale
    varP = np.var(SourceHom[:3, :], axis=1).sum()
    Scale = 1 / varP * np.sum(D)
    # translation
    Translation = TargetHom[:3, :].mean(axis=1) - SourceHom[:3, :].mean(axis=1).dot(Scale*Rotation.T)
    # transformation matrix
    OutTransform = np.identity(4)
    OutTransform[:3, :3] = Scale * Rotation
    OutTransform[:3, 3] = Translation

    return Scale, Rotation, Translation, OutTransform

def align_nocs_to_depth(masks, coords, depth, intrinsics, instance_ids, img_path, verbose=False):
    num_instances = len(instance_ids)
    error_messages = ''
    elapses = []
    scales = np.zeros(num_instances)
    rotations = np.zeros((num_instances, 3, 3))
    translations = np.zeros((num_instances, 3))

    for i in range(num_instances):
        mask = masks[:, :, i]
        coord = coords[:, :, i, :]
        pts, idxs = backproject(depth, intrinsics, mask)
        coord_pts = coord[idxs[0], idxs[1], :] - 0.5
        try:
            start = time.time()
            s, R, T, outtransform = estimateSimilarityTransform(coord_pts, pts, False)
            elapsed = time.time() - start
            if verbose:
                print('elapsed: ', elapsed)
            elapses.append(elapsed)
        except Exception as e:
            message = '[ Error ] aligning instance {} in {} fails. Message: {}.'.format(instance_ids[i], img_path, str(e))
            print(message)
            error_messages += message + '\n'
            s = 1.0
            R = np.eye(3)
            T = np.zeros(3)
            outtransform = np.identity(4, dtype=np.float32)

        scales[i] = s / 1000.0
        rotations[i, :, :] = R
        translations[i, :] = T / 1000.0

    return scales, rotations, translations, error_messages, elapses

def align_nocs_to_depth_v2(mask, coord, depth, intrinsics, instance_id, img_path, verbose=False):
    error_messages = ''
    elapses = []
    scales = None
    rotations = np.zeros((3, 3))
    translations = np.zeros((3))

    pts, idxs = backproject(depth, intrinsics, mask)
    coord_pts = coord[idxs[0], idxs[1], :] - 0.5
    try:
        start = time.time()
        s, R, T, outtransform = estimateSimilarityTransform(coord_pts, pts, False)
        elapsed = time.time() - start
        if verbose:
            print('elapsed: ', elapsed)
        elapses.append(elapsed)
    except Exception as e:
        message = '[ Error ] aligning instance {} in {} fails. Message: {}.'.format(instance_id, img_path, str(e))
        print(message)
        error_messages += message + '\n'
        s = 1.0
        R = np.eye(3)
        T = np.zeros(3)
        outtransform = np.identity(4, dtype=np.float32)

    scale = s / 1000.0
    rotation = R
    translation = T / 1000.0

    return scale, rotation, translation, error_messages, elapses

def load_depth(img_path):
    # returns depth data
    depth_path = img_path + '_depth.png'
    depth = cv2.imread(depth_path, -1)
    if len(depth.shape) == 3:  #THIS IS PROBABLY FOR CAMERA DATASET
        # This is encoded depth image, let's convert
        # NOTE: RGB is actually BGR in opencv
        depth16 = depth[:, :, 1] * 256 + depth[:, :, 2]
        depth16 = np.where(depth16 == 32001, 0, depth16)
        depth16 = depth16.astype(np.uint16)
    elif len(depth.shape) == 2 and depth.dtype == 'uint16':  # THIS IS PROBABLY FOR REAL275
        depth16 = depth
    else:
        assert False, '[ Error ]: Unsupported depth type.'
    return depth16

def load_coord(img_path):
    coord_path = img_path + '_coord.png'
    coord_map = cv2.imread(coord_path)[:, :, :3]
    # B G R to R G B or R G B to B G R
    coord_map = coord_map[:, :, (2, 1, 0)]
    # flip z axis of coord map - no idea why?
    coord_map = np.array(coord_map, dtype=np.float32) / 255
    coord_map[:, :, 2] = 1 - coord_map[:, :, 2]
    return coord_map

def load_mask(img_path):
    mask_path = img_path + '_mask.png'
    mask = cv2.imread(mask_path)[:, :, 2]
    mask = np.array(mask, dtype=np.int32)
    return mask

def load_label(img_path, label_type='projected_bbox'):
    label_path = img_path + '_label.pkl'
    # load the label
    with open(label_path, 'rb') as handle:
        label = pickle.load(handle)
    label = label[label_type]  # Shape : Num Instances x 8 x 2
    return label

def load_colored(img_path):
    img = cv2.imread(img_path + '_color.png')
    return img

def scene_to_point_cloud(depth, intrinsic):
    """
    Converts depth map to point cloud.
    :param ndarray depth: depth image
    :param tuple focal_length: horizontal and vertical focal length (fx, fy)
    :param tuple center: optical center of depth camera (cx, cy)
    :param tuple bounding_box: (x,y,width,height)
    :return: point cloud list of [x,y,z] coordinates
    """
    pc = []
    fx = intrinsic[0, 0]
    fy = intrinsic[1, 1]
    cx = intrinsic[0, 2]
    cy = intrinsic[1, 2]

    height = depth.shape[0]
    width = depth.shape[1]

    # following is a matrix like [[0, 0, 0, 0, ...], [1, 1, 1, 1, 1...]]
    U_matrix = np.repeat(np.expand_dims(np.arange(height), axis=1), width, axis=1)
    # following is a matrix like [[0, 1, 2, ... width-1], [0, 1, 2, width-1]]
    V_matrix = np.repeat(np.expand_dims(np.arange(width), axis=0), height, axis=0)

    X = np.multiply((V_matrix - cx), depth) / fx
    Y = np.multiply((U_matrix - cy), depth) / fy
    output = np.concatenate((
        np.expand_dims(X, axis=-1),
        np.expand_dims(Y, axis=-1),
        np.expand_dims(depth, axis=-1),
    ), axis=-1)
    return output


def split_into_boxes(image_data, height_sep=2, width_sep=3):
    #
    # image data torch shape is N x 256 x H_fix x W_fix  (256 is geo embed dimension)
    # output torch shape will be N x 256 x num_boxes x (H_fix / sqrt(num_boxes)) x ((W_fix / sqrt(num_boxes)))
    h = image_data.size()[-2]
    w = image_data.size()[-1]
    N = image_data.size()[0]
    embedding_dim = image_data.size()[1]

    # followings are the sizes of small boxes
    height_sep_fac = h // height_sep
    width_sep_fac = w // width_sep
    output_num_boxes = height_sep * width_sep
    output = torch.zeros((N, embedding_dim, output_num_boxes, height_sep_fac, width_sep_fac))

    for i in range(output_num_boxes):
        row_index = i // width_sep
        column_index = i % width_sep
        slice = image_data[:, :, row_index*height_sep_fac: (row_index+1)*height_sep_fac,
                column_index*width_sep_fac:(column_index+1)*width_sep_fac]
        output[:, :, i, :, :] = slice
    return output

def get_relationship_from_splits(split_data, rel_type='subtraction'):
    # input data is N x 256 x NumBoxes(height_sep x width_sep)  x H_fix x W_fix
    # output data shape: N x 256 x Comb(Num_boxes, 2)  x H_fix x W_fix
    h = split_data.size()[-2]
    w = split_data.size()[-1]
    N = split_data.size()[0]
    embedding_dim = split_data.size()[1]

    num_boxes = split_data.size()[2]
    # Get all combinations of [1, 2, 3]
    # and length 2
    comb = list(combinations([j for j in range(num_boxes)], 2))
    output = torch.zeros((N, embedding_dim, len(comb), h, w))
    for i in range(len(comb)):
        # take those slices
        slice_1 = split_data[:, :, comb[i][0], :, :]
        slice_2 = split_data[:, :, comb[i][1], :, :]
        if rel_type == 'subtraction':
            output[:, :, i, :, :] = slice_1 - slice_2
        else:
            pass

    return output


# source: https://github.com/nghiaho12/rigid_transform_3D/blob/master/rigid_transform_3D.py
# Input: expects 3xN matrix of points
# Returns R,t
# R = 3x3 rotation matrix
# t = 3x1 column vector

def rigid_transform_3D(A, B):
    assert A.shape == B.shape

    num_rows, num_cols = A.shape
    if num_rows != 3:
        raise Exception(f"matrix A is not 3xN, it is {num_rows}x{num_cols}")

    num_rows, num_cols = B.shape
    if num_rows != 3:
        raise Exception(f"matrix B is not 3xN, it is {num_rows}x{num_cols}")

    # find mean column wise
    centroid_A = np.mean(A, axis=1)
    centroid_B = np.mean(B, axis=1)

    # ensure centroids are 3x1
    centroid_A = centroid_A.reshape(-1, 1)
    centroid_B = centroid_B.reshape(-1, 1)

    # subtract mean
    Am = A - centroid_A
    Bm = B - centroid_B

    H = Am @ np.transpose(Bm)

    # sanity check
    #if linalg.matrix_rank(H) < 3:
    #    raise ValueError("rank of H = {}, expecting 3".format(linalg.matrix_rank(H)))

    # find rotation
    U, S, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T

    # special reflection case
    if np.linalg.det(R) < 0:
        print("det(R) < R, reflection detected!, correcting for it ...")
        Vt[2,:] *= -1
        R = Vt.T @ U.T
    t = -R @ centroid_A + centroid_B
    return R, t
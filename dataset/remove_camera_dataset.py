import os

if __name__ == "__main__":
    user_dir = r'C:\Users\Cem Okan'  # change this on your computer
    data_dir = user_dir + r'\Dropbox (GaTech)\deep_learning_data'

    dataset_name = 'val'
    full_datapath = data_dir + '/' + dataset_name + '_processed_list_all.txt'
    full_metapath = data_dir + '/' + dataset_name + '_processed_meta.txt'

    full_data = open(full_datapath, 'r').readlines()
    meta_data = open(full_metapath, 'r').readlines()
    print("len full data: ", len(full_data))
    indices_to_remove = []
    index = 0
    with open(full_datapath, 'r') as f:
        for line in f:
            # lets say for image 0000 our line is 1 6 mug2_scene3_norm
            line_info = line.strip().split(' ')  # ['1', '6', 'mug2_scene3_norm']
            if line_info[0][0:4] == "Real":
                indices_to_remove.append(index)
            index += 1
    new_datalist = [i for j, i in enumerate(full_data) if j not in indices_to_remove]
    new_metadata = [i for j, i in enumerate(meta_data) if j not in indices_to_remove]
    print("len new data: ", len(new_datalist))

    with open(os.path.join(data_dir, dataset_name + '_processed_list_camera.txt'), 'w') as f:
        for img_path in new_datalist:
            f.write("%s" % img_path)
    with open(os.path.join(data_dir, dataset_name + '_processed_meta_camera.txt'), 'w') as f:
        for meta in new_metadata:
            f.write("%s" % meta)
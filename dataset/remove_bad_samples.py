from dataset.dataloader import BadSampleRemover
from torch.utils.data import DataLoader
import os
import time

if __name__ == "__main__":
    user_dir = r'C:\Users\Cem Okan'  # change this on your computer
    data_dir = user_dir + r'\Dropbox (GaTech)\deep_learning_data'
    # sampling size does not work right now

    extracted_feature_size = (32, 48)

    train_dataset = BadSampleRemover(data_dir, data_name='train', apply_normalization=True, resize=extracted_feature_size)
    val_dataset = BadSampleRemover(data_dir, data_name='val', apply_normalization=True, resize=extracted_feature_size)
    test_dataset = BadSampleRemover(data_dir, data_name='test', apply_normalization=True, resize=extracted_feature_size)

    batch_size = 1  # due to proposed approach, this should be 1
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)


    # print("test loader size before: ", len(test_loader))
    # counter = 0
    # prev_time = time.time()
    # for data in test_loader:
    #     counter += 1
    #     if counter % 1000 == 0:
    #         current_time = time.time()
    #         print(f"counter: {counter},  time passes: {current_time - prev_time}")
    #         prev_time = current_time
    # test_dataset.data_list = [i for j, i in enumerate(test_dataset.data_list) if j not in test_dataset.remove_list]
    # test_dataset.meta_list = [i for j, i in enumerate(test_dataset.meta_list) if j not in test_dataset.remove_list]
    # print("test loader size after: ", len(test_dataset.data_list))
    #
    # # save files
    # for _ in range(len(test_dataset.data_list)):
    #     with open(os.path.join(data_dir, 'test' + '_processed_list_all.txt'), 'w') as f:
    #         for img_path in test_dataset.data_list:
    #             f.write("%s\n" % img_path)
    #     with open(os.path.join(data_dir, 'test' + '_processed_meta.txt'), 'w') as f:
    #         for meta in test_dataset.meta_list:
    #             f.write("%s\n" % meta)

    # counter = 0
    # print("val loader size before: ", len(val_loader))
    # prev_time = time.time()
    # for data in val_loader:
    #     counter += 1
    #     if counter % 1000 == 0:
    #         current_time = time.time()
    #         print(f"counter: {counter},  time passes: {current_time - prev_time}")
    #         prev_time = current_time
    # val_dataset.data_list = [i for j, i in enumerate(val_dataset.data_list) if j not in val_dataset.remove_list]
    # val_dataset.meta_list = [i for j, i in enumerate(val_dataset.meta_list) if j not in val_dataset.remove_list]
    # print("val loader size after: ", len(val_dataset.data_list))
    # # save files
    # for _ in range(len(val_dataset.data_list)):
    #     with open(os.path.join(data_dir, 'val' + '_processed_list_all.txt'), 'w') as f:
    #         for img_path in val_dataset.data_list:
    #             f.write("%s\n" % img_path)
    #     with open(os.path.join(data_dir, 'val' + '_processed_meta.txt'), 'w') as f:
    #         for meta in val_dataset.meta_list:
    #             f.write("%s\n" % meta)



    print("train loader size before: ", len(train_loader))
    counter = 0
    prev_time = time.time()
    for data in train_loader:
        counter += 1
        if counter % 1000 == 0:
            current_time = time.time()
            print(f"counter: {counter},  time passes: {current_time - prev_time}")
            prev_time = current_time
    # save files
    train_dataset.data_list = [i for j, i in enumerate(train_dataset.data_list) if j not in train_dataset.remove_list]
    train_dataset.meta_list = [i for j, i in enumerate(train_dataset.meta_list) if j not in train_dataset.remove_list]
    print("train loader size after: ", len(train_dataset.data_list))
    for _ in range(len(train_dataset.data_list)):
        with open(os.path.join(data_dir, 'train' + '_processed_list_all.txt'), 'w') as f:
            for img_path in train_dataset.data_list:
                f.write("%s\n" % img_path)
        with open(os.path.join(data_dir, 'train' + '_processed_meta.txt'), 'w') as f:
            for meta in train_dataset.meta_list:
                f.write("%s\n" % meta)



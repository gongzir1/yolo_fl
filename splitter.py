import os
import shutil
from src.utils import folder_data_splitter_iid,folder_data_splitter_non_iid,create_groups,split_dataset_val_train,rename_images,folder_data_splitter_n_class_non_iid

if __name__ == "__main__":

    # move_files_to_directory(data/og_datasets,data/combine-data)

    # split_dataset_val_train('data/combine-data', img_ext="jpg", train_split=0.8)

    # folder_data_splitter_non_iid('datasets/meat_3/non-iid', n_splits=3, agent=1)
    # for i in range(9):
    #     agent = i
    #     ext = f"_iid_{agent}"  # Modify ext based on the agent value
    #     folder_data_splitter_iid('data/train', n_splits=9, agent=agent, img_extension="jpg", ext=ext)

    # folder_data_splitter_non_iid('data/train', n_splits=6, agent=0, img_extension="jpg")

    # split_dataset_val_train('data/datasets/meat_10/iid/client_1/val', img_ext="jpg", train_split=0.6)
    # split_dataset_val_train('data/val', img_ext="txt", train_split=0.6)

    # n-class non-iid
    for i in range(3):
        agent = i
        ext = f"_non_iid_{agent}"  # Modify ext based on the agent value
        folder_data_splitter_n_class_non_iid('data/train',n_splits=3, agent=agent,num_classes_per_client=2,total_classes=6, img_extension="jpg", ext=ext,seed=42)



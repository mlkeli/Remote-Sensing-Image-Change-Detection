import os
import random
import shutil

# 原始数据文件夹路径和新的训练集/验证集文件夹路径
folders = {
    'original': {
        'im1': 'D:/变化监测/im1',
        'im2': 'D:/变化监测/im2',
        'label1': 'D:/变化监测/label1',
        'label2': 'D:/变化监测/label2',
        'label3': 'D:/变化监测/label3'
    },
    'train': {
        'im1': 'D:/变化监测/train/im1',
        'im2': 'D:/变化监测/train/im2',
        'label1': 'D:/变化监测/train/label1',
        'label2': 'D:/变化监测/train/label2',
        'label3': 'D:/变化监测/train/label3'
    },
    'val': {
        'im1': 'D:/变化监测/val/im1',
        'im2': 'D:/变化监测/val/im2',
        'label1': 'D:/变化监测/val/label1',
        'label2': 'D:/变化监测/val/label2',
        'label3': 'D:/变化监测/val/label3'
    }
}

# 创建文件夹
def create_folders(folder_paths):
    for folder_path in folder_paths:
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

# 复制文件
def copy_files(selected_files, original_folder, train_folder, val_folder):
    for file in os.listdir(original_folder):
        original_file_path = os.path.join(original_folder, file)
        if file in selected_files:
            shutil.copy(original_file_path, os.path.join(val_folder, file))
        else:
            shutil.copy(original_file_path, os.path.join(train_folder, file))

# 获取原始文件夹中的所有文件
file_list = os.listdir(folders['original']['im1'])

# 计算20%的数据量
num_samples = int(len(file_list) * 0.2)

# 随机选择20%的数据
selected_files = random.sample(file_list, num_samples)

# 创建训练集和验证集文件夹
create_folders([folders['train']['im1'], folders['train']['im2'], 
                folders['train']['label1'], folders['train']['label2'], folders['train']['label3']])
create_folders([folders['val']['im1'], folders['val']['im2'], 
                folders['val']['label1'], folders['val']['label2'], folders['val']['label3']])

# 复制选中的文件到新的文件夹中
copy_files(selected_files, folders['original']['im1'], folders['train']['im1'], folders['val']['im1'])
copy_files(selected_files, folders['original']['im2'], folders['train']['im2'], folders['val']['im2'])
copy_files(selected_files, folders['original']['label1'], folders['train']['label1'], folders['val']['label1'])
copy_files(selected_files, folders['original']['label2'], folders['train']['label2'], folders['val']['label2'])
copy_files(selected_files, folders['original']['label3'], folders['train']['label3'], folders['val']['label3'])

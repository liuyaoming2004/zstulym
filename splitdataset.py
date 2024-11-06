import os
import shutil
import random

# 设置原始数据集路径
data_dir = 'voc_data'
images_dir = os.path.join(data_dir, 'voc_images')
labels_dir = os.path.join(data_dir, 'voc_labels_txt')

# 设置划分后的数据集输出路径
output_dir = 'datasets/pvn'
train_images_dir = os.path.join(output_dir, 'images/train')
test_images_dir = os.path.join(output_dir, 'images/test')
train_labels_dir = os.path.join(output_dir, 'labels/train')
test_labels_dir = os.path.join(output_dir, 'labels/test')

# 如果输出目录已存在，则删除并重新创建
if os.path.exists(output_dir):
    shutil.rmtree(output_dir)
os.makedirs(output_dir, exist_ok=True)
os.makedirs(train_images_dir, exist_ok=True)
os.makedirs(test_images_dir, exist_ok=True)
os.makedirs(train_labels_dir, exist_ok=True)
os.makedirs(test_labels_dir, exist_ok=True)

# 设置测试集划分比例
test_split = 0.2

# 获取所有有对应标签的图像文件列表，并删除无对应标签的图片
image_files = []
for f in os.listdir(images_dir):
    if f.endswith('.jpg') or f.endswith('.png'):
        label_file = os.path.splitext(f)[0] + '.txt'
        if os.path.exists(os.path.join(labels_dir, label_file)):
            image_files.append(f)
        else:
            img_path = os.path.join(images_dir, f)
            os.remove(img_path)

# 随机打乱文件列表
random.shuffle(image_files)

# 按照比例划分数据
num_test = int(len(image_files) * test_split)
num_train = len(image_files) - num_test
test_files = image_files[:num_test]
train_files = image_files[num_test:]

# 定义移动文件的函数
def move_files(files, src_dir_images, src_dir_labels, dest_dir_images, dest_dir_labels):
    for file in files:
        img_src_path = os.path.join(src_dir_images, file)
        img_dest_path = os.path.join(dest_dir_images, file)
        label_file = os.path.splitext(file)[0] + '.txt'
        label_src_path = os.path.join(src_dir_labels, label_file)
        label_dest_path = os.path.join(dest_dir_labels, label_file)
        shutil.copy(img_src_path, img_dest_path)
        shutil.copy(label_src_path, label_dest_path)

# 分别划分并存储训练集和测试集
move_files(train_files, images_dir, labels_dir, train_images_dir, train_labels_dir)
move_files(test_files, images_dir, labels_dir, test_images_dir, test_labels_dir)

print("数据集划分完成！")
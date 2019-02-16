clc; clear all;
seed = 95461354;
rng(seed)

img_name = 'test_'
img_path = ['data/original/'];
output_path = 'data/formatted_trainval/';

dataset_name = ['icanteen_patches'];
path = ['data/original/images'];
output_path = 'data/formatted_trainval/';
train_path_img = strcat(output_path, dataset_name,'/train/');
train_path_den = strcat(output_path, dataset_name,'/train_den/');
val_path_img = strcat(output_path, dataset_name,'/val/');
val_path_den = strcat(output_path, dataset_name,'/val_den/');
gt_path = ['data/original/ground_truth'];

mkdir(output_path)
mkdir(train_path_img);
mkdir(train_path_den);
mkdir(val_path_img);
mkdir(val_path_den);

num_images = 25;
num_val = ceil(num_images*0.1);
indices = randperm(num_images);


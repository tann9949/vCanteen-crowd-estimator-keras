clc; clear all;

% data/original/ground_truth/
path = ['data/original/images/'];
gt_path = 'data/original/ground_truth/';

num_images = 50;

for idx = 47:num_images
    figure;
    index = num2str(idx);
    img_name = ['IMG_' index];
    img = strcat(path,img_name,'.jpg');
    disp(img);
    imshow(img);
    [x,y] = getpts;
    
    location = [x y];
    number = size(location,1);
    
    lo_num.location = location;
    lo_num.number = number;
    image_info = {lo_num};
    
    file_name = strcat(gt_path, 'GT_',img_name,'.mat');
    save(file_name, 'image_info');
    close all;
end
close all;

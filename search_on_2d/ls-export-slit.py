import argparse
import glob
import os
import shutil

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--input", required=False,
    default="project-1-at",
    help="path of exported label studio result")
args = vars(ap.parse_args())
input_path = args["input"]

output_path = input_path + "-split"
if os.path.exists(output_path):
    shutil.rmtree(output_path)
path_image_train=output_path+"/images/train"
os.makedirs(path_image_train)
path_image_val=output_path+"/images/val"
os.makedirs(path_image_val)
path_label_train=output_path+"/labels/train"
os.makedirs(path_label_train)
path_label_val=output_path+"/labels/val"
os.makedirs(path_label_val)
images = glob.glob(input_path + "/images/*.*")
images = sorted(images)
labels = glob.glob(input_path + "/labels/*.*")
labels = sorted(labels)

split = int(len(images)*0.8)
src_train_images = images[:split]
src_train_labels = labels[:split]
src_val_images = images[split:]
src_val_labels = labels[split:]
    
for file in src_train_images: shutil.copy2(file, path_image_train)
for file in src_train_labels: shutil.copy2(file, path_label_train)
for file in src_val_images: shutil.copy2(file, path_image_val)
for file in src_val_labels: shutil.copy2(file, path_label_val)

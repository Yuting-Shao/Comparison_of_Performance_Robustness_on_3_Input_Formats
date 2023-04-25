# Yuting Shao
# shao.yut@northeastern.edu
# CS 5330 Spring 2023
# Final Project - Check the robustness of the trained ResNet-34: intensity, color balance, and orientation

# import statements
import os
os.environ['OPENCV_IO_ENABLE_OPENEXR']="1"
import cv2
import torch
import sys
import time
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torchvision.models import resnet34
from torchvision.models import vgg16
from torch.utils.data import DataLoader
from torchvision.models.resnet import ResNet34_Weights
from torchvision.models.vgg import VGG16_Weights
from torchsummary import summary
from torchviz import make_dot


# class definitions

# custom dataset class that inherits from PyTorch's Dataset class
# and reads the image data using OpenCV, retaining the original data type.
class CustomImageFolder(ImageFolder):
    def __init__(self, root, custom_transforms=None, *args, **kwargs):
        super(CustomImageFolder, self).__init__(root, is_valid_file=is_valid_img_file, *args, **kwargs)
        self.custom_transforms = custom_transforms

    def __getitem__(self, index):
        path, target = self.samples[index]
        image = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if self.custom_transforms:
            for transform in self.custom_transforms:
                image = transform(image)

                if image is None:
                    return None, None

        image = torch.from_numpy(np.transpose(image, (2, 0, 1))).float()
        return image, target

# custom dataset class that inherits from PyTorch's Dataset class
# and reads the image data using OpenCV, retaining the original data type
# applying variations to intensity, color balance, and orientation.
class CustomVariationImageFolder(ImageFolder):
    def __init__(self, root, custom_transforms=None, *args, **kwargs):
        super(CustomVariationImageFolder, self).__init__(root, is_valid_file=is_valid_img_file, *args, **kwargs)
        self.custom_transforms = custom_transforms

    def __getitem__(self, index):
        path, target = self.samples[index]
        image = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # variations in intensity
        intensity_variation = 50
        img_intensity_variation = image.copy()
        img_intensity_variation = cv2.add(img_intensity_variation, intensity_variation)
        
        # variations in color balance
        blue, green, red = cv2.split(img_intensity_variation)
        blue = cv2.add(blue, 10)
        green = cv2.add(green, 20)
        red = cv2.add(red, 30)
        img_color_balance_variation = cv2.merge((blue, green, red))
        
        # variations in orientation
        rows, cols, _ = img_color_balance_variation.shape
        rotation_matrix = cv2.getRotationMatrix2D((cols / 2, rows / 2), 30, 1)
        img_orientation_variation = cv2.warpAffine(img_color_balance_variation, rotation_matrix, (cols, rows))

        if self.custom_transforms:
            for transform in self.custom_transforms:
                img_orientation_variation = transform(img_orientation_variation)

                if img_orientation_variation is None:
                    return None, None

        image = torch.from_numpy(np.transpose(img_orientation_variation, (2, 0, 1))).float()
        return image, target

# function definitions

# load a pre-trained ResNet-34 model and modify the last layer to match 10 categories.
def create_resnet_model(num_categories=10):
    model = resnet34(weights=ResNet34_Weights.IMAGENET1K_V1)
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, num_categories)
    return model

# initialize a VGG16 model.
def init_vgg16(num_categories=10):
    model = vgg16(weights=VGG16_Weights.IMAGENET1K_V1)
    num_features = model.classifier[6].in_features
    model.classifier[6] = nn.Linear(num_features, num_categories)
    return model

# Load a saved model
def load_model(path, modeltype, num_categories=10):
    if modeltype == "resnet34":
        model = create_resnet_model(num_categories)
    elif modeltype == "vgg16":
        model = init_vgg16(num_categories=10)

    model.load_state_dict(torch.load(path))
    model.eval()  # Set the model to evaluation mode
    return model

# determine if a file should be included in the dataset.
def is_valid_img_file(filename):
    return filename.lower().endswith('.exr') or filename.lower().endswith('.jpg') or filename.lower().endswith('.tiff')

# crops the central area of the image with a specified size
def center_crop(image, size=(63, 63)):
    height, width = image.shape[:2]
    crop_height, crop_width = size

    if height < crop_height or width < crop_width:
        print(height, width)
        return None

    top = (height - crop_height) // 2
    left = (width - crop_width) // 2
    bottom = top + crop_height
    right = left + crop_width

    return image[top:bottom, left:right]

# normalize the input image to the range [0, 1].
def normalize_0_1(image):
    min_val = np.min(image)
    max_val = np.max(image)
    return (image - min_val) / (max_val - min_val)

# a custom collate function that filters out the None values
def custom_collate(batch):
    batch = list(filter(lambda x: x[0] is not None, batch))
    return torch.utils.data.dataloader.default_collate(batch)

# function to evaluate the performace on testset
def evaluate_model(model, test_loader):
    model.eval()
    running_corrects = 0
    for inputs, labels in test_loader:
        with torch.no_grad():
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            running_corrects += torch.sum(preds == labels.data)

    test_accuracy = running_corrects.double() / len(test_loader.dataset)
    print(f"Test Accuracy: {test_accuracy}")
    return test_accuracy

# Calculate testing accuracy on the original testing dataset and the variated testing dataset
def calculate_test_accuracy(modelpath, test_data_path, img_type, model_type):
    # define the custom transforms for each type of image
    custom_transforms = {
        'log': [center_crop],
        'linear': [center_crop, normalize_0_1],
        'jpg': [center_crop, normalize_0_1]
    }


    # load the model
    print(f"Loading model from {modelpath}")
    model = load_model(modelpath, model_type)

    # create data loaders for the original and variated test datasets
    print(f"Creating test datasets from {test_data_path}")
    original_test_dataset = CustomImageFolder(test_data_path, custom_transforms=custom_transforms[img_type])
    variated_test_dataset = CustomVariationImageFolder(test_data_path, custom_transforms=custom_transforms[img_type])

    original_test_loader = DataLoader(original_test_dataset, batch_size=32, shuffle=True)
    variated_test_loader = DataLoader(variated_test_dataset, batch_size=32, shuffle=True)

    # evaluate the model on the original test dataset
    original_test_accuracy = evaluate_model(model, original_test_loader)
    print(f"Original Test Accuracy: {original_test_accuracy}")

    # evaluate the model on the variated test dataset
    variated_test_accuracy = evaluate_model(model, variated_test_loader)
    print(f"Variated Test Accuracy: {variated_test_accuracy}")

    return original_test_accuracy, variated_test_accuracy

# main function
def main(argv):
    # handle any command line arguments in argv

    # main function code
    calculate_test_accuracy("trained_resnet34_jpg64db.pth", "Database/jpg64DB/testset/", "jpg", "resnet34")
    calculate_test_accuracy("trained_resnet34_linear64db.pth", "Database/ln64db/testset/", "linear", "resnet34")
    calculate_test_accuracy("trained_resnet34_log64db.pth", "Database/log64db/testset/", "log", "resnet34")
    calculate_test_accuracy("trained_vgg16_jpg64db.pth", "Database/jpg64DB/testset/", "jpg", "vgg16")
    calculate_test_accuracy("trained_vgg16_linear64db.pth", "Database/ln64db/testset/", "linear", "vgg16")
    calculate_test_accuracy("trained_vgg16_log64db.pth", "Database/log64db/testset/", "log", "vgg16")
    
    return

if __name__ == "__main__":
    main(sys.argv)
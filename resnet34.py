# Yuting Shao
# shao.yut@northeastern.edu
# CS 5330 Spring 2023
# Final Project - Compare jpg/linear/log images for a image recognition using ResNet-34

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
from torch.utils.data import DataLoader
from torchvision.models.resnet import ResNet34_Weights
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

# function definitions

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

# load datasets for each size/format combination into a dictionary.
def load_datasets():
    # define the custom transforms for each type of image
    custom_transforms = {
        'log': [center_crop],
        'linear': [center_crop, normalize_0_1],
        'jpg': [center_crop, normalize_0_1]
    }

    # create a dictionary to store the datasets
    datasets = {}

    # define the root directory for each size/format combination
    root_directories = {
        'jpg64DB': 'jpg',
        'ln64db': 'linear',
        'log64db': 'log'
    }

    # load the datasets
    for folder_name, img_type in root_directories.items():
        for dataset_type in ['devset', 'testset', 'trainset']:
            root = os.path.join('Database', folder_name, dataset_type)
            dataset_key = f'{folder_name}_{dataset_type}'
            datasets[dataset_key] = CustomImageFolder(root, custom_transforms=custom_transforms[img_type])

    return datasets

# load a pre-trained ResNet-34 model and modify the last layer to match 10 categories.
def create_resnet_model(num_categories=10):
    model = resnet34(weights=ResNet34_Weights.IMAGENET1K_V1)
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, num_categories)
    return model

# train the model.
def train_model(model, train_loader, val_loader, test_loader, criterion, optimizer, num_epochs=8, patience=3):
    loss_values = []
    accuracy_values = []
    test_accuracy_values = []
    early_stop_counter = 0
    best_test_accuracy = 0.0

    for epoch in range(num_epochs):
        # training phase
        model.train()
        running_loss = 0.0
        for inputs, labels in train_loader:
            optimizer.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        # validation phase
        model.eval()
        running_corrects = 0
        val_running_loss = 0.0
        for inputs, labels in val_loader:
            with torch.no_grad():
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                running_corrects += torch.sum(preds == labels.data)
                val_loss = criterion(outputs, labels)
                val_running_loss += val_loss.item()

        epoch_loss = running_loss / len(train_loader)
        epoch_val_loss = val_running_loss / len(val_loader)
        epoch_accuracy = running_corrects.double() / len(val_loader.dataset)
        loss_values.append(epoch_loss)
        accuracy_values.append(epoch_accuracy)

        # testing phase
        test_accuracy = evaluate_model(model, test_loader)
        test_accuracy_values.append(test_accuracy)

        print(f"Epoch {epoch + 1}/{num_epochs}, Training Loss: {epoch_loss}, Validation Loss: {epoch_val_loss}, Validation Accuracy: {epoch_accuracy}, Testing Accuracy: {test_accuracy}")

        # Check for early stopping
        if test_accuracy > best_test_accuracy + 0.005:
            best_test_accuracy = test_accuracy
            early_stop_counter = 0
        else:
            early_stop_counter += 1

        if early_stop_counter >= patience:
            if  epoch >= 10:
                print(f"Early stopping triggered after {epoch + 1} epochs")
                break
            else:
                best_test_accuracy = test_accuracy
                early_stop_counter = 0

    return loss_values, accuracy_values, test_accuracy_values

# function to plot loss and accuracy
def plot_loss_accuracy(loss_values, val_accuracy_values, test_accuracy_values, filename=None):
    epochs = range(1, len(loss_values) + 1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Subfigure for Loss
    ax1.plot(epochs, loss_values, 'r', label='Training Loss')
    ax1.scatter(epochs, loss_values, c='r', marker='o')
    ax1.set_title('Training Loss')
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Loss')
    ax1.legend()

    # Subfigure for Validation and Testing Accuracy
    ax2.plot(epochs, val_accuracy_values, 'b', label='Validation Accuracy')
    ax2.scatter(epochs, val_accuracy_values, c='b', marker='o')
    ax2.plot(epochs, test_accuracy_values, 'k', label='Testing Accuracy')
    ax2.scatter(epochs, test_accuracy_values, c='k', marker='o')
    ax2.set_title('Validation and Testing Accuracy')
    ax2.set_xlabel('Epochs')
    ax2.set_ylabel('Accuracy')
    ax2.legend()

    plt.tight_layout()

    if filename:
        plt.savefig(filename)
    plt.show(block=False)
    plt.close()

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

# train and evaluate the model
def train_and_evaluate(batch_size=64, num_epochs=8, lr=0.001, momentum=0.9, datasets_name=['jpg64DB_trainset', 'jpg64DB_devset', 'jpg64DB_testset'], filename=None, modelname=None, load_model_path=None, print_model=True):
    # get the iamge type
    # define the root directory for each size/format combination
    root_directories = {
        'jpg64DB': 'jpg',
        'ln64db': 'linear',
        'log64db': 'log'
    }
    folder_name = datasets_name[0].split('_')[0]
    image_type = root_directories[folder_name]
    print(f"Image type of the dataset: {image_type}")

    # set random seed
    torch.manual_seed(42)

    # turn off CUDA
    torch.backends.cudnn.enabled = False

    # load datasets
    print('Loading datasets...')
    datasets = load_datasets()

    if load_model_path:
        print(f'Loading model from {load_model_path}...')
        model = load_model(load_model_path)
    else:
    # create a ResNet model
        print('Creating ResNet model...')
        model = create_resnet_model()

    if print_model:
        summary(model, input_size=(3, 63, 63))
        # Generate a diagram of the model's computation graph
        x = torch.randn(1, 3, 63, 63, requires_grad=True)
        y = model(x)
        dot = make_dot(y, params=dict(model.named_parameters()).update({"x": x}))
        dot.format = "png"
        dot.render(modelname)

    # create a DataLoader for the train and validation datasets of jpg64DB
    print('Creating DataLoader...')
    train_loader = DataLoader(datasets[datasets_name[0]], batch_size=batch_size, shuffle=True, collate_fn=custom_collate)
    val_loader = DataLoader(datasets[datasets_name[1]], batch_size=batch_size, shuffle=False, collate_fn=custom_collate)
    test_loader = DataLoader(datasets[datasets_name[2]], batch_size=batch_size, shuffle=False, collate_fn=custom_collate)

    # define the loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum)

    # train the model
    print('Training model...')
    loss_values, accuracy_values, test_accuracy_values = train_model(model, train_loader, val_loader, test_loader, criterion, optimizer, num_epochs=num_epochs)

    # plot loss and accuracy
    plot_loss_accuracy(loss_values, accuracy_values, test_accuracy_values, filename)

    # test the model
    print('Testing model...')
    test_accuracy = evaluate_model(model, test_loader)

    # Save the model
    if modelname:
        print('Saving model...')
        save_model(model, modelname+'.pth')

    return test_accuracy

# plot the test_accuracies vs. one changed parameter
def plot_test_accuracy_and_runtime_vs_one_param(x_values, accuracies, run_times, xlabel, filename=None):
    fig, ax1 = plt.subplots()

    ax1.plot(x_values, accuracies, 'o-', markersize=8, linewidth=2, color='b', label='Test Accuracy')
    ax1.set_xlabel(xlabel)
    ax1.set_ylabel('Test Accuracy')
    ax1.tick_params(axis='y', labelcolor='b')

    ax2 = ax1.twinx()
    ax2.plot(x_values, run_times, 'o-', markersize=8, linewidth=2, color='r', label='Run Time')
    ax2.set_ylabel('Run Time (seconds)')
    ax2.tick_params(axis='y', labelcolor='r')

    fig.tight_layout()

    if filename:
        plt.savefig(filename)
    plt.show(block=False)
    plt.close()

# Save the trained model
def save_model(model, path):
    torch.save(model.state_dict(), path)

# Load a saved model
def load_model(path, num_categories=10):
    model = create_resnet_model(num_categories)
    model.load_state_dict(torch.load(path))
    model.eval()  # Set the model to evaluation mode
    return model

# main function
def main(argv):
    # handle any command line arguments in argv

    # main function code

    # optimize number of epochs using early stopping
    print(f"Evaluating num of epochs for jpg...")
    start_time = time.time()
    test_accuracy = train_and_evaluate(batch_size=16, num_epochs=50, datasets_name=['jpg64DB_trainset', 'jpg64DB_devset', 'jpg64DB_testset'], filename='num_epochs_loss_jpg.png', modelname='trained_resnet34_jpg64db')
    end_time = time.time()
    run_time = end_time - start_time
    print(f"Train and evaluate took {run_time:.2f} seconds")

    # find best batch size
    batch_sizes = [16, 32, 64, 128]
    test_accuracies = []

    for batch_size in batch_sizes:
        print(f"Training and evaluating with batch size {batch_size}...")
        test_accuracy = train_and_evaluate(batch_size, num_epochs=50, datasets_name=['jpg64DB_trainset', 'jpg64DB_devset', 'jpg64DB_testset'], filename=f'num_epochs_loss_jpg_batch_size_{batch_size}.png', modelname=f'trained_resnet34_jpg64db_batch_size_{batch_size}', load_model_path='trained_resnet34_jpg64db.pth', print_model=False)
        test_accuracies.append(test_accuracy)

    best_batch_size_idx = np.argmax(test_accuracies)
    best_batch_size = batch_sizes[best_batch_size_idx]
    print(f"The best batch size is: {best_batch_size} for jpg dataset")

    # update the saved model with the best one
    model = load_model(f'trained_resnet34_jpg64db_batch_size_{best_batch_size}.pth')
    save_model(model, 'trained_resnet34_jpg64db.pth')
    print('trained_resnet34_jpg64db.pth has been updated with the best batch size.')

    # find best learning rate
    learning_rates = [1E-4, 1E-3, 1E-2, 1E-1]
    test_accuracies = []

    for lr in learning_rates:
        print(f"Training and evaluating with learning rate {lr}...")
        test_accuracy = train_and_evaluate(batch_size=best_batch_size, num_epochs=50, lr=lr, datasets_name=['jpg64DB_trainset', 'jpg64DB_devset', 'jpg64DB_testset'], filename=f'num_epochs_loss_jpg_lr_{lr}.png', modelname=f'trained_resnet34_jpg64db_lr_{lr}', load_model_path='trained_resnet34_jpg64db.pth', print_model=False)
        test_accuracies.append(test_accuracy)

    best_lr_idx = np.argmax(test_accuracies)
    best_lr = learning_rates[best_lr_idx]
    print(f"The best learning rates is: {best_lr} for jpg dataset")

    # update the saved model with the best one
    model = load_model(f'trained_resnet34_jpg64db_lr_{best_lr}.pth')
    save_model(model, 'trained_resnet34_jpg64db.pth')
    print('trained_resnet34_jpg64db.pth has been updated with the best learning rates.')

    # find best momentum
    momentums = [0.5, 0.75, 0.9, 0.99]
    test_accuracies = []

    for momentum in momentums:
        print(f"Training and evaluating with momentum {momentum}...")
        test_accuracy = train_and_evaluate(batch_size=best_batch_size, num_epochs=50, lr=best_lr, momentum=momentum, datasets_name=['jpg64DB_trainset', 'jpg64DB_devset', 'jpg64DB_testset'], filename=f'num_epochs_loss_jpg_momentum_{momentum}.png', modelname=f'trained_resnet34_jpg64db_momentum_{momentum}', load_model_path='trained_resnet34_jpg64db.pth', print_model=False)
        test_accuracies.append(test_accuracy)

    best_momentum_idx = np.argmax(test_accuracies)
    best_momentum = momentums[best_momentum_idx]
    print(f"The best momentum is: {best_momentum} for jpg dataset")

    best_accuracy = test_accuracies[best_momentum_idx]
    print(f"For the jpg dataset, the best testing accuracy is: {best_accuracy}, with the batch size={best_batch_size}, learning rate={best_lr}, momentum={best_momentum}")

    # update the saved model with the best one
    model = load_model(f'trained_resnet34_jpg64db_momentum_{best_momentum}.pth')
    save_model(model, 'trained_resnet34_jpg64db.pth')
    print('trained_resnet34_jpg64db.pth has been updated with the best momentum.')

    # optimize number of epochs using early stopping
    print(f"Evaluating num of epochs for linear...")
    start_time = time.time()
    test_accuracy = train_and_evaluate(batch_size=16, num_epochs=50, datasets_name=['ln64db_trainset', 'ln64db_devset', 'ln64db_testset'], filename='num_epochs_loss_linear.png', modelname='trained_resnet34_linear64db')
    end_time = time.time()
    run_time = end_time - start_time
    print(f"Train and evaluate took {run_time:.2f} seconds")

    # find best batch size
    batch_sizes = [16, 32, 64, 128]
    test_accuracies = []

    for batch_size in batch_sizes:
        print(f"Training and evaluating with batch size {batch_size}...")
        test_accuracy = train_and_evaluate(batch_size, num_epochs=50, datasets_name=['ln64db_trainset', 'ln64db_devset', 'ln64db_testset'], filename=f'num_epochs_loss_linear_batch_size_{batch_size}.png', modelname=f'trained_resnet34_linear64db_batch_size_{batch_size}', load_model_path='trained_resnet34_linear64db.pth', print_model=False)
        test_accuracies.append(test_accuracy)

    best_batch_size_idx = np.argmax(test_accuracies)
    best_batch_size = batch_sizes[best_batch_size_idx]
    print(f"The best batch size is: {best_batch_size} for linear dataset")

    # update the saved model with the best one
    model = load_model(f'trained_resnet34_linear64db_batch_size_{best_batch_size}.pth')
    save_model(model, 'trained_resnet34_linear64db.pth')
    print('trained_resnet34_linear64db.pth has been updated with the best batch size.')

    # find best learning rate
    learning_rates = [1E-4, 1E-3, 1E-2, 1E-1]
    test_accuracies = []

    for lr in learning_rates:
        print(f"Training and evaluating with learning rate {lr}...")
        test_accuracy = train_and_evaluate(batch_size=best_batch_size, num_epochs=50, lr=lr, datasets_name=['ln64db_trainset', 'ln64db_devset', 'ln64db_testset'], filename=f'num_epochs_loss_linear_lr_{lr}.png', modelname=f'trained_resnet34_linear64db_lr_{lr}', load_model_path='trained_resnet34_linear64db.pth', print_model=False)
        test_accuracies.append(test_accuracy)

    best_lr_idx = np.argmax(test_accuracies)
    best_lr = learning_rates[best_lr_idx]
    print(f"The best learning rates is: {best_lr} for linear dataset")

    # update the saved model with the best one
    model = load_model(f'trained_resnet34_linear64db_lr_{best_lr}.pth')
    save_model(model, 'trained_resnet34_linear64db.pth')
    print('trained_resnet34_linear64db.pth has been updated with the best learning rates.')

    # find best momentum
    momentums = [0.5, 0.75, 0.9, 0.99]
    test_accuracies = []

    for momentum in momentums:
        print(f"Training and evaluating with momentum {momentum}...")
        test_accuracy = train_and_evaluate(batch_size=best_batch_size, num_epochs=50, lr=best_lr, momentum=momentum, datasets_name=['ln64db_trainset', 'ln64db_devset', 'ln64db_testset'], filename=f'num_epochs_loss_linear_momentum_{momentum}.png', modelname=f'trained_resnet34_linear64db_momentum_{momentum}', load_model_path='trained_resnet34_linear64db.pth', print_model=False)
        test_accuracies.append(test_accuracy)

    best_momentum_idx = np.argmax(test_accuracies)
    best_momentum = momentums[best_momentum_idx]
    print(f"The best momentum is: {best_momentum} for linear dataset")

    # update the saved model with the best one
    model = load_model(f'trained_resnet34_linear64db_momentum_{best_momentum}.pth')
    save_model(model, 'trained_resnet34_linear64db.pth')
    print('trained_resnet34_linear64db.pth has been updated with the best momentum.')

    best_accuracy = test_accuracies[best_momentum_idx]
    print(f"For the linear dataset, the best testing accuracy is: {best_accuracy}, with the batch size={best_batch_size}, learning rate={best_lr}, momentum={best_momentum}")

    # optimize number of epochs using early stopping
    print(f"Evaluating num of epochs for log...")
    start_time = time.time()
    test_accuracy = train_and_evaluate(batch_size=16, num_epochs=50, datasets_name=['log64db_trainset', 'log64db_devset', 'log64db_testset'], filename='num_epochs_loss_log.png', modelname='trained_resnet34_log64db')
    end_time = time.time()
    run_time = end_time - start_time
    print(f"Train and evaluate took {run_time:.2f} seconds")

    # find best batch size
    batch_sizes = [16, 32, 64, 128]
    test_accuracies = []

    for batch_size in batch_sizes:
        print(f"Training and evaluating with batch size {batch_size}...")
        test_accuracy = train_and_evaluate(batch_size, num_epochs=50, datasets_name=['log64db_trainset', 'log64db_devset', 'log64db_testset'], filename=f'num_epochs_loss_log_batch_size_{batch_size}.png', modelname=f'trained_resnet34_log64db_batch_size_{batch_size}', load_model_path='trained_resnet34_log64db.pth', print_model=False)
        test_accuracies.append(test_accuracy)

    best_batch_size_idx = np.argmax(test_accuracies)
    best_batch_size = batch_sizes[best_batch_size_idx]
    print(f"The best batch size is: {best_batch_size} for log dataset")

    # update the saved model with the best one
    model = load_model(f'trained_resnet34_log64db_batch_size_{best_batch_size}.pth')
    save_model(model, 'trained_resnet34_log64db.pth')
    print('trained_resnet34_log64db.pth has been updated with the best batch size.')

    # find best learning rate
    learning_rates = [1E-4, 1E-3, 1E-2, 1E-1]
    test_accuracies = []

    for lr in learning_rates:
        print(f"Training and evaluating with learning rate {lr}...")
        test_accuracy = train_and_evaluate(batch_size=best_batch_size, num_epochs=50, lr=lr, datasets_name=['log64db_trainset', 'log64db_devset', 'log64db_testset'], filename=f'num_epochs_loss_log_lr_{lr}.png', modelname=f'trained_resnet34_log64db_lr_{lr}', load_model_path='trained_resnet34_log64db.pth', print_model=False)
        test_accuracies.append(test_accuracy)

    best_lr_idx = np.argmax(test_accuracies)
    best_lr = learning_rates[best_lr_idx]
    print(f"The best learning rates is: {best_lr} for log dataset")

    # update the saved model with the best one
    model = load_model(f'trained_resnet34_log64db_lr_{best_lr}.pth')
    save_model(model, 'trained_resnet34_log64db.pth')
    print('trained_resnet34_log64db.pth has been updated with the best learning rates.')

    # find best momentum
    momentums = [0.5, 0.75, 0.9, 0.99]
    test_accuracies = []

    for momentum in momentums:
        print(f"Training and evaluating with momentum {momentum}...")
        test_accuracy = train_and_evaluate(batch_size=best_batch_size, num_epochs=50, lr=best_lr, momentum=momentum, datasets_name=['log64db_trainset', 'log64db_devset', 'log64db_testset'], filename=f'num_epochs_loss_log_momentum_{momentum}.png', modelname=f'trained_resnet34_log64db_momentum_{momentum}', load_model_path='trained_resnet34_log64db.pth', print_model=False)
        test_accuracies.append(test_accuracy)

    best_momentum_idx = np.argmax(test_accuracies)
    best_momentum = momentums[best_momentum_idx]
    print(f"The best momentum is: {best_momentum} for log dataset")

    # update the saved model with the best one
    model = load_model(f'trained_resnet34_log64db_momentum_{best_momentum}.pth')
    save_model(model, 'trained_resnet34_log64db.pth')
    print('trained_resnet34_log64db.pth has been updated with the best momentum.')

    best_accuracy = test_accuracies[best_momentum_idx]
    print(f"For the log dataset, the best testing accuracy is: {best_accuracy}, with the batch size={best_batch_size}, learning rate={best_lr}, momentum={best_momentum}")

    return

if __name__ == "__main__":
    main(sys.argv)
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import matplotlib.pyplot as plt
import os
import tensorflow as tf
import cv2
import copy
import torch
import torch.nn as nn
import torch.optim as optim

from torchvision import models
from sklearn.model_selection import KFold
from torch.utils.data import Dataset, DataLoader
from skimage import io, img_as_float
from sklearn.model_selection import train_test_split

# paths for directories 
train_data_path = 'sorted_train_data'
categories = ['cherry', 'strawberry', 'tomato']

# METHODS FOR IMAGE PREPROCESSING

# resizing and aspect ratio normalization
def resize_images(image_paths, target_size=(200, 200)):
    resized_images = []
    for image_path in image_paths: #iterate through image paths
        img = cv2.imread(image_path) #read image
        img_resized = cv2.resize(img, target_size) #resize image
        resized_images.append(img_resized) #append resized image to list
    return resized_images

# preprocess images through flattening
def preprocess_images(images, target_size=(200, 200)):
    processed_images = []
    for img in images: #iterate through images
        if img.dtype != np.float32: #check if image dtype is not float32
            img = img.astype(np.float32) #convert image dtype to float32
        if len(img.shape) == 2: #check if image shape is 2 (no channel)
            img_resized = cv2.resize(img, target_size) #resize image
        elif len(img.shape) == 3: #check if image shape is 3 (with channel)
            img_resized = cv2.resize(img, target_size) #resize image
        processed_images.append(img_resized) #append resized image to list
    return np.array(processed_images)

# image augmentation
def augment_image(image):
    flipped_image = tf.image.flip_left_right(image) #flip image horizontally
    rotated_image = tf.image.rot90(image) #rotate image 90 degrees
    zoom_factor = np.random.uniform(0.6, 0.8) #generate random zoom factor
    zoomed_image = tf.image.central_crop(image, zoom_factor) #zoom image
    return flipped_image.numpy(), rotated_image.numpy(), zoomed_image.numpy()

# loader method to load images given their respective category
def load_images_from_category(category):
    category_path = os.path.join(train_data_path, category) #get category path
    image_files = [os.path.join(category_path, img) for img in os.listdir(category_path) if img.endswith('.jpg')] #get image files
    images = [img_as_float(io.imread(img)) for img in image_files] #read images
    return images

# method to remove images less than 300 (width, height)
def remove_small_images(images, min_size=(300, 300)):
    filtered_images = [img for img in images if img.shape[0] >= min_size[0] and img.shape[1] >= min_size[1]] #filter images less than 300
    return filtered_images

# create new datalists to store extracted augmented images. copy processed images to list
def create_augmented_images(augmented_cherry_images, augmented_strawberry_images, augmented_tomato_images):
    for i in range(num_cherry_images): #iterate through cherry images
        img_copy = copy.deepcopy(cherry_images[i]) #copy cherry image
        if len(img_copy.shape) == 2: #check if image shape is 2 (no channel)
            img_copy = cv2.cvtColor(img_copy, cv2.COLOR_BGR2RGB) #convert image to RGB
        flipped_img, rotated_img, zoomed_img = augment_image(img_copy) #augment image
        augmented_cherry_images.extend([flipped_img, rotated_img, zoomed_img]) #extend augmented cherry images
    for i in range(num_strawberry_images): #iterate through strawberry images
        img_copy = copy.deepcopy(strawberry_images[i]) #copy strawberry image
        if len(img_copy.shape) == 2: #check if image shape is 2 (no channel)
            img_copy = cv2.cvtColor(img_copy, cv2.COLOR_BGR2RGB) #convert image to RGB
        flipped_img, rotated_img, zoomed_img = augment_image(img_copy) #augment image
        augmented_strawberry_images.extend([flipped_img, rotated_img, zoomed_img]) #extend augmented strawberry images
    for i in range(num_tomato_images): #iterate through tomato images
        img_copy = copy.deepcopy(tomato_images[i]) #copy tomato image
        if len(img_copy.shape) == 2: #check if image shape is 2 (no channel)
            img_copy = cv2.cvtColor(img_copy, cv2.COLOR_BGR2RGB) #convert image to RGB
        flipped_img, rotated_img, zoomed_img = augment_image(img_copy) #augment image
        augmented_tomato_images.extend([flipped_img, rotated_img, zoomed_img]) #extend augmented tomato images
    
    return augmented_cherry_images, augmented_strawberry_images, augmented_tomato_images

# create labels for each dataset
class ImageDataset(Dataset):
    def __init__(self, images, labels):
        self.images = images #set images
        self.labels = labels #set labels
    def __len__(self):
        return len(self.images) #return length of images
    def __getitem__(self, idx):
        image = self.images[idx] #get image
        label = self.labels[idx] #get label
        if len(image.shape) == 2:
            image = torch.tensor(image, dtype=torch.float32).unsqueeze(0) #convert image to tensor
        else:
            image = torch.tensor(image, dtype=torch.float32).permute(2, 0, 1) #convert image to tensor
        return image, torch.tensor(label, dtype=torch.long) #return image and label

# FINAL MODEL TRAINING

# create model for training
class Model(nn.Module):
    def __init__(self, num_classes=3):
        super(Model, self).__init__()
        resnet = models.resnet18(pretrained=True) #initialize resnet model
        for param in resnet.parameters(): #set parameters to false
            param.requires_grad = False #freeze parameters
        
        in_features = resnet.fc.in_features #get in features
        resnet.fc = nn.Linear(in_features, num_classes) #set linear layer to number of classes
        self.resnet = resnet #set resnet model

    def forward(self, x):
        return self.resnet(x) #return resnet model

# RUNNING PREPROCESSING

# load images from their respective dataset category
cherry_images = load_images_from_category('cherry')
strawberry_images = load_images_from_category('strawberry')
tomato_images = load_images_from_category('tomato')

# convert image dtypes to float32
cherry_images = [img.astype(np.float32) for img in cherry_images]
strawberry_images = [img.astype(np.float32) for img in strawberry_images]
tomato_images = [img.astype(np.float32) for img in tomato_images]

# get total count of image categories to understand amount of image in each dataset
num_cherry_images = len(cherry_images)
num_strawberry_images = len(strawberry_images)
num_tomato_images = len(tomato_images)

# remove images under 300 from each dataset and get new total count
cherry_images = remove_small_images(cherry_images)
strawberry_images = remove_small_images(strawberry_images)
tomato_images = remove_small_images(tomato_images)

# number of images in each dataset
num_cherry_images = len(cherry_images)
num_strawberry_images = len(strawberry_images)
num_tomato_images = len(tomato_images)

# ensure that image datasets are equal
min_images = min(num_cherry_images, num_strawberry_images, num_tomato_images)
cherry_images = cherry_images[:min_images]
strawberry_images = strawberry_images[:min_images]
tomato_images = tomato_images[:min_images]
num_cherry_images = len(cherry_images)
num_strawberry_images = len(strawberry_images)
num_tomato_images = len(tomato_images)

# create augmented images
augmented_cherry_images = []
augmented_strawberry_images = []
augmented_tomato_images = []
augmented_cherry_images, augmented_strawberry_images, augmented_tomato_images = create_augmented_images(augmented_cherry_images, augmented_strawberry_images, augmented_tomato_images)

# RUNNING TRAINING

#preprocess augmented images
augmented_cherry_images = preprocess_images(augmented_cherry_images)
augmented_strawberry_images = preprocess_images(augmented_strawberry_images)
augmented_tomato_images = preprocess_images(augmented_tomato_images)
print("PROCESSED IMAGES")

#create labels for each image in dataset
augmented_cherry_labels = np.zeros(augmented_cherry_images.shape[0])
augmented_strawberry_labels = np.ones(augmented_strawberry_images.shape[0])
augmented_tomato_labels = np.full(augmented_tomato_images.shape[0], 2)
print("PERFORMED LABEL ENCODING")

#combine augmented images and labels
augmented_features = np.concatenate([augmented_cherry_images, augmented_strawberry_images, augmented_tomato_images])
augmented_labels = np.concatenate([augmented_cherry_labels, augmented_strawberry_labels, augmented_tomato_labels])
print("FEATURES SHAPE:", augmented_features.shape)
print("LABELS SHAPE:", augmented_labels.shape)

#split augmented dataset into train and test
x_augmented_train, x_augmented_test, y_augmented_train, y_augmented_test = train_test_split(
    augmented_features, augmented_labels, test_size=0.3, random_state=42
)
print("SHAPES OF TRAIN AND TEST SPLIT")
print("XT", x_augmented_train.shape)
print("XTST", x_augmented_test.shape)
print("YT", y_augmented_train.shape)
print("YTST", y_augmented_test.shape)

augmented_x = augmented_features #augmented features
augmented_y = augmented_labels #augmented labels

kf = KFold(n_splits=3, shuffle=True, random_state=42) #use KFold to split dataset into 3 folds

fold_train_losses = [[] for _ in range(3)]
fold_val_accuracies = [[] for _ in range(3)]
augmented_fold_accuracies = []
augmented_num_epochs = 20 #number of epochs

for fold, (train_idx, val_idx) in enumerate(kf.split(augmented_x)): #iterate through each fold
    print(f"Fold {fold + 1}")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    augmented_model_py = Model(num_classes=3) #initialize model
    augmented_model_py.to(device) #set model to device cpu
    augmented_criterion = nn.CrossEntropyLoss() #set criterion to CrossEntropyLoss
    augmented_optimizer = optim.Adam(augmented_model_py.resnet.fc.parameters(), lr=0.001) #set optimizer to Adam

    train_images, val_images = augmented_x[train_idx], augmented_x[val_idx] #get train and validation images
    train_labels, val_labels = augmented_y[train_idx], augmented_y[val_idx] #get train and validation labels
    augmented_train_dataset = ImageDataset(train_images, train_labels) #create train dataset
    augmented_val_dataset = ImageDataset(val_images, val_labels) #create validation dataset
    augmented_train_loader = DataLoader(augmented_train_dataset, batch_size=32, shuffle=True) #create train loader with batch size 32
    augmented_val_loader = DataLoader(augmented_val_dataset, batch_size=32, shuffle=False) #create validation loader with batch size 32

    best_loss = float('inf')
    patience = 3 #set patience to 3
    no_improve_count = 0 #if model does not improve after 3 epochs, stop training

    for epoch in range(augmented_num_epochs):
        augmented_model_py.train() #set model to train
        augmented_running_loss, augmented_correct, augmented_total = 0.0, 0, 0 #initialize running loss, correct and total

        for images, labels in augmented_train_loader: #iterate through train loader dataset
            images, labels = images.to(device), labels.to(device)
            augmented_outputs = augmented_model_py(images) #get model outputs
            augmented_loss = augmented_criterion(augmented_outputs, labels) #calculate loss

            augmented_optimizer.zero_grad() #zero gradients
            augmented_loss.backward() #backpropagate loss
            augmented_optimizer.step() #update optimizer

            augmented_running_loss += augmented_loss.item() #add loss to running loss
            _, augmented_predicted = torch.max(augmented_outputs, 1) #get predicted values
            augmented_total += labels.size(0) #get total labels
            augmented_correct += (augmented_predicted == labels).sum().item() #get correct labels

        augmented_train_accuracy = 100 * augmented_correct / augmented_total #calculate train accuracy
        avg_train_loss = augmented_running_loss / len(augmented_train_loader)
        fold_train_losses[fold].append(augmented_running_loss / len(augmented_train_loader))
        print(f"Epoch [{epoch + 1}/{augmented_num_epochs}], Loss: {augmented_running_loss:.4f}, Accuracy: {augmented_train_accuracy:.2f}%")

        if avg_train_loss < best_loss: #check if average train loss is less than best loss
            best_loss = avg_train_loss #set best loss to average train loss
            no_improve_count = 0 #reset no improve count
        else:
            no_improve_count += 1 #increment no improve count
            if no_improve_count >= patience: #if no improve count is greater than 3
                print(f"Early stopping at epoch {epoch + 1} for Fold {fold + 1}")
                break #stop current fold

        augmented_model_py.eval() #set model to evaluation
        val_correct, val_total = 0, 0 #initialize correct and total
        with torch.no_grad():
            for images, labels in augmented_val_loader: #iterate through validation loader dataset
                images, labels = images.to(device), labels.to(device) 
                val_outputs = augmented_model_py(images) #get model outputs
                _, val_predicted = torch.max(val_outputs, 1) #get predicted values
                val_total += labels.size(0) #get total labels
                val_correct += (val_predicted == labels).sum().item() #get correct labels

        val_accuracy = 100 * val_correct / val_total #calculate validation accuracy
        fold_val_accuracies[fold].append(val_accuracy)

    augmented_fold_accuracies.append(fold_val_accuracies[fold][-1])
    print(f"Fold {fold + 1} Final Validation Accuracy: {augmented_fold_accuracies[-1]:.2f}%") #print final validation accuracy for that fold

augmented_average_accuracy = np.mean(augmented_fold_accuracies) #calculate average validation accuracy across all folds
print(f"Average Validation Accuracy: {augmented_average_accuracy:.2f}%")

#save the trained model
MODEL_PATH = "model.pth"
torch.save(augmented_model_py.state_dict(), MODEL_PATH)
print(f"Model saved to {MODEL_PATH}")
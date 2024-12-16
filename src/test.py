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
from torch.utils.data import Dataset, DataLoader
from skimage import io, img_as_float
from sklearn.model_selection import train_test_split

# paths for directories
test_data_path = 'sorted_train_data'
categories = ['cherry', 'strawberry', 'tomato']

# METHODS FOR IMAGE PREPROCESSING

# loader method to load images given their respective category
def load_images_from_category(category):
    category_path = os.path.join(test_data_path, category) #get category path
    image_files = [os.path.join(category_path, img) for img in os.listdir(category_path) if img.endswith('.jpg')] #get image files
    images = [img_as_float(io.imread(img)) for img in image_files] #read images
    return images
    
# resizing and aspect ratio normalization
def resize_images(image_paths, target_size=(300, 300)):
    resized_images = []
    for image_path in image_paths: #iterate through image paths
        img = cv2.imread(image_path) #read image
        img_resized = cv2.resize(img, target_size) #resize image
        resized_images.append(img_resized) #append resized image to list
    return resized_images

# preprocess images through flattening
def preprocess_images(images, target_size=(300, 300)):
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

# MODEL

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

# create model
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

#load images from their respective dataset category
cherry_images = load_images_from_category('cherry')
strawberry_images = load_images_from_category('strawberry')
tomato_images = load_images_from_category('tomato')

print(f"Total cherry images: {len(cherry_images)}")
print(f"Total strawberry images: {len(strawberry_images)}")
print(f"Total tomato images: {len(tomato_images)}")

# RUNNING PREPROCESSING
print("STARTING PREPROCESSING")

#convert image dtypes to float32
cherry_images = [img.astype(np.float32) for img in cherry_images]
strawberry_images = [img.astype(np.float32) for img in strawberry_images]
tomato_images = [img.astype(np.float32) for img in tomato_images]

#get total count of image categories to understand amount of image in each dataset
num_cherry_images = len(cherry_images)
num_strawberry_images = len(strawberry_images)
num_tomato_images = len(tomato_images)

#remove images under 300 from each dataset and get new total count
cherry_images = remove_small_images(cherry_images)
strawberry_images = remove_small_images(strawberry_images)
tomato_images = remove_small_images(tomato_images)

#number of images in each dataset
num_cherry_images = len(cherry_images)
num_strawberry_images = len(strawberry_images)
num_tomato_images = len(tomato_images)

#ensure that image datasets are equal
min_images = min(num_cherry_images, num_strawberry_images, num_tomato_images)
cherry_images = cherry_images[:min_images]
strawberry_images = strawberry_images[:min_images]
tomato_images = tomato_images[:min_images]
num_cherry_images = len(cherry_images)
num_strawberry_images = len(strawberry_images)
num_tomato_images = len(tomato_images)

#create augmented images
augmented_cherry_images = []
augmented_strawberry_images = []
augmented_tomato_images = []
augmented_cherry_images, augmented_strawberry_images, augmented_tomato_images = create_augmented_images(augmented_cherry_images, augmented_strawberry_images, augmented_tomato_images)

print("FINISHED PREPROCESSING")

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
x_test = np.concatenate([augmented_cherry_images, augmented_strawberry_images, augmented_tomato_images])
y_test = np.concatenate([augmented_cherry_labels, augmented_strawberry_labels, augmented_tomato_labels])

#create test dataset and loader
test_dataset = ImageDataset(x_test, y_test) #create test dataset
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False) #create test loader with batch size 32

# MODEL

MODEL_PATH = "model.pth"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = Model()
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.to(device)
model.eval()
print("MODEL LOADED")

# test model
correct, total = 0, 0 #initialize correct and total
with torch.no_grad(): 
    for images, labels in test_loader: #iterate through test loader (images, labels)
        images, labels = images.to(device), labels.to(device) #send images and labels to device
        outputs = model(images) #get model outputs
        _, predicted = torch.max(outputs, 1) #get predicted labels
        total += labels.size(0) #add labels size to total
        correct += (predicted == labels).sum().item() #add correct predictions to correct
test_accuracy = 100 * correct / total #calculate test accuracy
print(f"Test Accuracy: {test_accuracy:.2f}%")
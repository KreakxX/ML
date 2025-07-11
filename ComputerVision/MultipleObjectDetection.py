import keras 
import tensorflow as tf
import numpy as np
import pandas as pd
from keras import layers,models
import xml.etree.ElementTree as ET
import os
from PIL import ImageDraw, ImageFont, Image
import matplotlib.pyplot as plt
import torchvision.datasets as datasets
import random

voc_dataset = datasets.VOCDetection(
    root='C:/Users/Henri/Videos/VOC_data', 
    year='2012',
    image_set='train',
    download=True
)

input_shape = (256, 256, 3)

multipleBoundingBoxesModel = keras.Sequential([
    layers.Input(shape=input_shape),             # Input layer takes images 256x 256 with np dimensions
    layers.Conv2D(64, 3, activation="relu", padding="same"),            
    layers.Conv2D(64, 3, activation="relu", padding="same"),         # first convolutional layer with 64 filters each 3 
    layers.MaxPooling2D(),                  # pooling for performance
    layers.Dropout(0.2),                    # dropout for preventing overfitting
    
    layers.Conv2D(128, 3, activation="relu", padding="same"),
    layers.Conv2D(128, 3, activation="relu", padding="same"),        # second convolutional layer with 128 filers each 3
    layers.MaxPooling2D(),                  # pooling for performance
    layers.Dropout(0.2),                    # dropout for preventing overfitting
    
    layers.Conv2D(256, 3, activation="relu", padding="same"),          
    layers.Conv2D(256, 3, activation="relu", padding="same"),       # third convolutional layer with 256 filters each 3
    layers.MaxPooling2D(),                  # pooling for performance
    layers.Dropout(0.2),                    # dropout for preventing overfitting

    layers.Conv2D(512, 3, activation="relu", padding="same"),       # fourth convolutional layer with 512 filters each 3
    layers.Conv2D(512, 3, activation="relu", padding="same"),
    layers.GlobalAveragePooling2D(),        # pooling for performance
    
    layers.Dense(512, activation="relu"),                           # Neuron layer with 512 Neurons
    layers.Dropout(0.4),                    # dropout for preventing overfitting
    layers.Dense(256, activation="relu"),                           # Neuron layer with 256 Neurons
    layers.Dropout(0.3),                    # dropout for preventing overfitting
    layers.Dense(10, activation="linear")                           # Output layer
])

TARGET_SIZE = (256, 256)

# Model compiling
multipleBoundingBoxesModel.compile(
     optimizer="adam",  
    loss="mse"
)


# Data paths
img_dir = r"C:\Users\Henri\Videos\VOC_data\VOCdevkit\VOC2012\JPEGImages"
xml = r"C:\Users\Henri\Videos\VOC_data\VOCdevkit\VOC2012\Annotations"

# Method for parsing XML files
def parse_annotation(xml_file):
    tree = ET.parse(xml_file)       # open xml file
    root = tree.getroot()       # get <annotation> tag
    
    filename = root.find("filename").text   # filename
    size = root.find("size")        # and size
    orig_w = int(size.find("width").text)       # also original size etc
    orig_h = int(size.find("height").text)
    
    boxes = []      # initialize arrays
    labels = []
    
    for obj in root.findall("object"):      # find all Objects 2x more
        name = obj.find("name").text.lower()
        bbox = obj.find("bndbox")           # get a box

        xmin = int(float(bbox.find("xmin").text))
        ymin = int(float(bbox.find("ymin").text))       
        xmax = int(float(bbox.find("xmax").text))       # also all the coordinates
        ymax = int(float(bbox.find("ymax").text))
        
        center_x = (xmin + xmax) / 2            # and center_x, center_y aswell as the width
        center_y = (ymin + ymax) / 2
        width = (xmax - xmin)
        height = (ymax - ymin)          # implement resizing if not working properly
        
        new_xmin = center_x - width / 2
        new_ymin = center_y - height / 2           # for resizing cases if needed (not primarly)
        new_xmax = center_x + width / 2
        new_ymax = center_y + height / 2

        x_center = ((new_xmin + new_xmax) / 2) / orig_w     # normalization for nn to handle better
        y_center = ((new_ymin + new_ymax) / 2) / orig_h         # and end
        width = (new_xmax - new_xmin) / orig_w
        height = (new_ymax - new_ymin) / orig_h
        
        boxes.append([x_center, y_center, width, height, 1.0])      # append also wit 1.0 confidence   
        labels.append(name)
    
    if len(boxes) < 2:          # if there arent enough boxes return None at least 2 for good training
        return None
    
    combined = list(zip(boxes, labels))
    combined.sort(key=lambda x: x[0][0])        # sort by x_center was a real fix for the second box, because else the second box was offset
    boxes, labels = zip(*combined)


    boxes = boxes[:2]
    labels = labels[:2]

    flattened_boxes = []
    for box in boxes:
        flattened_boxes.extend(box)

    return filename, flattened_boxes, labels

# loading Data 80 % of the Dataset
def load_data(percentage=0.8):  
    global img_dir, xml
    X = []
    y = []
    
    xml_files = [f for f in os.listdir(xml) if f.endswith(".xml")]
    
    num_files = int(len(xml_files) * percentage)    # only load a specific amount of data not all 
    xml_files = xml_files[:num_files]
    
    print(f"Loading {num_files} files ({percentage*100}% of total {len(os.listdir(xml))} files)")
    
    for i, xml_file in enumerate(xml_files):
        if i % 1000 == 0:                   
            print(f"Processed {i}/{num_files} files")
            
        full_xml = os.path.join(xml, xml_file)       # loop trhoguh and apply parsing of the boxes
        try:
            result  = parse_annotation(full_xml)
            if result is None:              # continue if there arent enough boxes
                continue
            filename, bbox, label = result
            img_path = os.path.join(img_dir, filename)
            if not os.path.exists(img_path):
                continue

            img = Image.open(img_path).convert("RGB")       # preprocessing  and normalizing the img
            img = img.resize(TARGET_SIZE)
            img_arr = np.array(img) / 255.0

            X.append(img_arr)           # appending to array
            y.append(bbox)
        except Exception as e:
            print(f"Error processing {xml_file}: {e}")
            continue

    print(f"Successfully loaded {len(X)} images")
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32) # return normalized dataset as np.array

X,Y = load_data()


# earlystopping for preventing overfitting in this case not really optimized
earlyStopping = keras.callbacks.EarlyStopping(
    monitor="val_loss",
    patience=10,
    restore_best_weights = True
)

# Training
multipleBoundingBoxesModel.fit(X,Y,epochs=80, validation_split=0.2, callbacks=[earlyStopping])
# Saving
multipleBoundingBoxesModel.save("Multiple_Bounding_Boxes_Model.keras")


# Denormalization to make the box relative to the Img size, for processing and showing
def denormalize_box(box, orig_size, target_size=(256, 256)):
    ow, oh = orig_size  
    x, y, w, h = box
    x *= target_size[0]; y *= target_size[1]  
    w *= target_size[0]; h *= target_size[1]
    xmin, ymin = x - w/2, y - h/2     
    xmax, ymax = x + w/2, y + h/2
    scale_x, scale_y = ow / target_size[0], oh / target_size[1]  
    return int(xmin*scale_x), int(ymin*scale_y), int(xmax*scale_x), int(ymax*scale_y)


def testMultiObjectDetection(img_path):

    # Model loading
    classificationModel = keras.models.load_model("Cat_Dog_Classifier4.keras")
    boxModel = keras.models.load_model("Multiple_Bounding_Boxes_Model.keras")

    
    img = Image.open(img_path).convert("RGB")
    img_copy = img.copy()  

    # preprocessing
    img_resized = img.resize((256,256))
    img_array = np.array(img_resized) / 255.0       # normalize
    img_array = np.expand_dims(img_array, axis=0)   # expand to 3 dims

    pred = boxModel.predict(img_array)[0]       # predict the boxes
    box1 = pred[0:4]            # first box coordinates
    conf1 = pred[4]         # confidence neccessary to identify important boxes
    box2 = pred[5:9]            # second box coordinates
    conf2 = pred[9]         # confidence neccessary to identify important boxes
    
    draw = ImageDraw.Draw(img_copy)

    if conf1 > 0.3:     # if the model is serious about the box 
        box_coords_1 = denormalize_box(box1, img.size)      # scaling it relative to img
        print(f"Box 1 denormalized: {box_coords_1}")
        
        box_img1 = img.crop(box_coords_1).resize((256,256))     # crop out and preprocess for classification model
        box_img1_arr = np.expand_dims(np.array(box_img1) / 255.0, axis=0)

        classification_img_1 = classificationModel.predict(box_img1_arr)[0] # predict the class

        draw.rectangle(box_coords_1, outline="red", width=3)        # draw the box with the coordinates
        label_idx1 = np.argmax(classification_img_1)
        label_names = ["Cat", "Dog"]
        label1 = f"{label_names[label_idx1]}: {classification_img_1[label_idx1]:.2f}"   # and label it

        font_size = max(20, min(img.size) // 15)
        try:
            font = ImageFont.truetype("arial.ttf", font_size)
        except:
            font = ImageFont.load_default()

        text_x_1 = box_coords_1[0]
        text_y_1 = max(box_coords_1[1] - font_size, 0)
        draw.text((text_x_1, text_y_1), label1, fill="red", font=font)

    # Same for the second box
    if conf2 > 0.3:
        box_coords_2 = denormalize_box(box2, img.size)

        print(f"Box 2 denormalized: {box_coords_2}")
        
        box_img2 = img.crop(box_coords_2).resize((256,256))
        box_img2_arr = np.expand_dims(np.array(box_img2) / 255.0, axis=0)
        classification_img_2 = classificationModel.predict(box_img2_arr)[0]

        draw.rectangle(box_coords_2, outline="blue", width=3)
        label_idx2 = np.argmax(classification_img_2)
        label_names = ["Cat", "Dog"]
        label2 = f"{label_names[label_idx2]}: {classification_img_2[label_idx2]:.2f}"

        font_size = max(20, min(img.size) // 15)
        try:
            font = ImageFont.truetype("arial.ttf", font_size)
        except:
            font = ImageFont.load_default()

        text_x_2 = box_coords_2[0]
        text_y_2 = max(box_coords_2[1] - font_size, 0)
        draw.text((text_x_2, text_y_2), label2, fill="blue", font=font)

    # display end result
    plt.imshow(img_copy)
    plt.axis("off")
    plt.show()


# testMultiObjectDetection("Hund_katze2.jpeg")


# Summary

# For mutliple object detection this is neccessary to know if not using YOLO (You only look once)
# - Bigger and deeper Model architecture with more outputs  up the 3/4 conv layer (512 filters)
# - Confidence as Output to let the model decided which boxes are important (should be shown)
# - Data preparation, more data esspecially with images and multiple boxes
# - filtering boxes by x_center in this case helped with the second box
# - more learning and micro adjustments like training model with resized boxes for better understanding
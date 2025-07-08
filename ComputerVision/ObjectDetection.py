import keras 
import tensorflow as tf
import numpy as np
import pandas as pd
from keras import layers,models
import xml.etree.ElementTree as ET
import os
from PIL import ImageDraw, ImageFont, Image
import matplotlib.pyplot as plt


# Region Proposals generating (where is an Object)
# Region classification = like an Apple

# Model for Classifcation and model for Image Box making so every box get put in the classifcation model 
# -> models bauen und loaden 

# basic CNN for bounding boxes with the algorythm
input_shape = (128, 128, 3)

# Model is training on relative to original size based images  thats why we need to denormalize
# Model for predicting the shapes   basic CNN with Pooling layers for performance minimize the inputs into the neural net
boxModel = keras.Sequential([
    layers.Input(shape=input_shape),
    layers.Conv2D(32, 3, activation='relu'),       # Convolutional layer with 32 filters each with a size of 3
    layers.MaxPooling2D(),          # Max Pooling for better performance
    layers.Conv2D(64, 3, activation='relu'),   # another convolutional layer bigger to find better shapes
    layers.MaxPooling2D(),            # Max Pooling for better performance   
    layers.Conv2D(128, 3, activation='relu'),      # and the Last big layer for finding shapes
    layers.GlobalAveragePooling2D(),      # prevents overfitting etc
    layers.Dense(128, activation='relu'),   # neuron layer
    layers.Dense(4, activation='sigmoid')  # and output layer because we need x_min, x_max, y_min y_max   # this is for only on detection
])

boxModel.compile(optimizer="adam",loss="mse")  # compile it

# model for classify the boxes(objects)
# load Dataset and train 
filepath = r"C:\Users\Henri\Videos\Dog_Cat_Object"
xml = r"C:\Users\Henri\Videos\Dog_Cat_Object\annotations"
img_dir = r"C:\Users\Henri\Videos\Dog_Cat_Object\images"

# target size for both models
TARGET_SIZE = (128, 128)

def parse_annotation(xml_file):
    tree = ET.parse(xml_file)
    root = tree.getroot()  # opens the XML file the <annotation> element

    filename = root.find("filename").text

    size = root.find("size")
    orig_w = int(size.find("width").text)   # gets original size of the image
    orig_h = int(size.find("height").text)  

    for obj in root.findall("object"):
        name = obj.find("name").text.lower() # gets the name of the box like label like dog or cat
        bbox = obj.find("bndbox")   # gets the bndBox element

        xmin = int(bbox.find("xmin").text)
        ymin = int(bbox.find("ymin").text)     # and from this elemtn get all teh positionx left top corner and bottom right corner
        xmax = int(bbox.find("xmax").text)
        ymax = int(bbox.find("ymax").text)

        x_center = ((xmin + xmax) / 2) / orig_w     # normalizinh the box like 0 - 1 so neural nets can train better with it and get the center point of the Box
        y_center = ((ymin + ymax) / 2) / orig_h   # here also     
        width = (xmax - xmin) / orig_w      # and the width and height relative to the img size
        height = (ymax - ymin) / orig_h

        return filename, [x_center, y_center, width, height], name
        #         (
        #   "dog1.jpg",                            # example output
        #   [0.45, 0.60, 0.30, 0.40],
        #   "dog"
        # )
       
def load_data():
    global img_dir,xml
    X = []
    y = []
    
    for xml_file in os.listdir(xml):      # looping through each xml File
        if not xml_file.endswith(".xml"):
            continue

        full_xml = os.path.join(xml, xml_file)     # get the xml
        filename, bbox, label = parse_annotation(full_xml)  # finding fileName like dog.jpg, get the bounding box for the detetion and the label of the bounding box
        img_path = os.path.join(img_dir, filename)  # and the image

        if not os.path.exists(img_path):
            continue

        img = Image.open(img_path).convert("RGB")
        img = img.resize(TARGET_SIZE)               # preprocess it normalize it and resize aswell as convert to RGB
        img_arr = np.array(img) / 255.0

        X.append(img_arr)      # add the images
        y.append(bbox)    # and the bouding box

    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)  # get the data of the images as numpy arrays and the bounding boxes


earlyStopping = keras.callbacks.EarlyStopping(
    monitor="val_loss",
    patience=4,
    restore_best_weights = True
)

# X,Y = load_data()   # get bounding boxes and the imgae
# boxModel.fit(X, Y, epochs=30, batch_size=16, validation_split=0.1, callbacks=earlyStopping) # train model to predict the bounding boxes and use earlystopping to prevent overfitting
# boxModel.save("BoxModel.keras") # and save it


def denormalize_box(box, orig_size, target_size=(128, 128)):
    ow, oh = orig_size  # original data
    x, y, w, h = box
    x *= target_size[0]; y *= target_size[1]  # scale back down
    w *= target_size[0]; h *= target_size[1]
    xmin, ymin = x - w/2, y - h/2      # from center based back to corner beased
    xmax, ymax = x + w/2, y + h/2
    scale_x, scale_y = ow / target_size[0], oh / target_size[1]  # scaling it back to original size with the ratio
    return int(xmin*scale_x), int(ymin*scale_y), int(xmax*scale_x), int(ymax*scale_y)



def testObjectDetection(img_path):
  # load models
  classification_model = keras.models.load_model("Cat_Dog_Classifier3.keras")   
  boxModel = keras.models.load_model("BoxModel.keras")

  # preprocess imgs
  img = Image.open(img_path).convert("RGB")
  img_resized = img.resize(TARGET_SIZE)
  img_array = np.array(img_resized) / 255.0  # resizing and normalising
  img_array = np.expand_dims(img_array, axis=0) # add a dimension because its expected in the input everytime when handling images for preprocessing

  # predict the boxes
  pred = boxModel.predict(img_array)[0] # predict the bouding boxes
  box_coords = denormalize_box(pred,img.size,TARGET_SIZE) # and denormalize box
  box_img = img.crop(box_coords).resize((128,128)) # than get the image from the specific box
  box_img_arr = np.expand_dims(np.array(box_img)/255., axis=0) # add dimension and normalize 
  classification = classification_model.predict(box_img_arr)[0] # and classify

  #visualize
  draw = ImageDraw.Draw(img)
  draw.rectangle(box_coords, outline="red", width=3)
  label_idx = np.argmax(classification)     # draw and visualize
  label_names = ["Cat", "Dog"] 
  label = f"{label_names[label_idx]}: {classification[label_idx]:.2f}"
  print(label)
  

  img_width, img_height = img.size
  font_size = max(20, min(img_width, img_height) // 15)
  try:
    font = ImageFont.truetype("arial.ttf", font_size)  
    
  except:
    font = ImageFont.load_default()
  text_x = box_coords[0]
  text_y = max(box_coords[1] - font_size , 0)
  draw.text((text_x, text_y), label, fill="red", font=font)

  plt.imshow(img)
  plt.axis("off")
  plt.show()


testObjectDetection("Hund.jpg")

  
  




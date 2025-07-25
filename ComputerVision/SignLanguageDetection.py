import keras 
import tensorflow as tf
import numpy as np
import pandas as pd
from keras import layers,models
import xml.etree.ElementTree as ET
import os
from PIL import ImageDraw, ImageFont, Image
import matplotlib.pyplot as plt
from keras.preprocessing.image import ImageDataGenerator

inputShape = (50,50,3)
filepath = r"C:\Users\Henri\Videos\SignLanguageNotIndian"
class_names = [d for d in os.listdir(filepath) if os.path.isdir(os.path.join(filepath, d))]
class_names_length = len(class_names)

# basic CNN for Image Classification
model = keras.Sequential([
  layers.Input(shape=inputShape),
  layers.Conv2D(32,3,activation="relu"),
  layers.MaxPooling2D(),
  layers.Conv2D(64,3,activation="relu"),
  layers.MaxPooling2D(),
  layers.Conv2D(50,3,activation="relu"),
  layers.GlobalAveragePooling2D(),
  layers.Dense(50,activation="relu"),
  layers.Dense(class_names_length, activation="softmax")
])

model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']   
)

# loading dataset 
datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2,
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,     # neccessary if the data same is
    zoom_range=0.1, 
    brightness_range=[0.8,1.2],
    horizontal_flip=False  # nur wenn sinnvoll
)

# training Dataset
train_generator = datagen.flow_from_directory(
    filepath,
    target_size=(50, 50),   # Shape Input size
    batch_size=32,
    class_mode='categorical',  # for more classes
    subset='training',
    shuffle=True
)

# validation Dataset
val_generator = datagen.flow_from_directory(
    filepath,
    target_size=(50, 50),
    batch_size=32,
    class_mode='categorical',
    subset='validation'
)

earlyStopping = keras.callbacks.EarlyStopping(
    monitor="val_loss",
    patience=5,
    restore_best_weights= True
)

# model.fit(
#     train_generator,
#     validation_data=val_generator,
#     epochs=10,
#     callbacks=[earlyStopping]
# )

# model.save("SignLanguageClassifier.keras")

def testSignClassifierModel(img_path):
      classificationModel = keras.models.load_model("SignLanguageClassifier.keras")
      img = Image.open(img_path).convert("RGB") # open the image
      img = img.resize((50, 50))     
      img_array = np.array(img) / 255.0
      img_array = np.expand_dims(img_array,axis=0)

      prediction = classificationModel.predict(img_array)[0]
      for i, name in enumerate(class_names):
        print(i, name)
      print(prediction)
      predicted_index = np.argmax(prediction)
      result = class_names[predicted_index]
      print(result)

# testSignClassifierModel("1.jpg")

# Box Model
BoxModel = models.Sequential([
    layers.Input(shape=inputShape),
    layers.Conv2D(32,3,activation="relu"),
    layers.MaxPooling2D(),
    layers.Conv2D(64,3,activation="relu"),
    layers.MaxPooling2D(),
    layers.Conv2D(50,3,activation="relu"),
    layers.GlobalAveragePooling2D(),
    layers.Dense(50,activation="relu"),
    layers.Dense(4, activation="sigmoid")
])

BoxModel.compile(optimizer="adam", loss="mse")

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
        
def load_data():
    X = []
    y = []

    filepath = r"C:\Users\Henri\Videos\train"
    for file in os.listdir(filepath):    # loop through all files
        if file.endswith(".xml"):        #checks if the file in xml
            full_xml = os.path.join(filepath, file) # join the file name with the path
            filename, bbox, label = parse_annotation(full_xml) 
            img_path = os.path.join(filepath, filename) # join the file name with the path
            if not os.path.exists(img_path): # checks if the file exists
                continue
            img = Image.open(img_path).convert("RGB") # open the image and covert it to RGB
            img = img.resize((50,50)) # resize the image
            img = np.array(img) # convert the image to numpy array
            img = img/255.0 # normalize the image
            
            X.append(img) 
            y.append(bbox)

    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)  

earlyStopping = keras.callbacks.EarlyStopping(
    monitor="val_loss",
    patience=5,
    restore_best_weights = True
)

# X,Y = load_data()
# print(len(X))
# print(len(Y))
# BoxModel.fit(X,Y, epochs=20, callbacks=[earlyStopping], validation_split = 0.2)
# BoxModel.save("BoxModel_sign_language.keras")


def denormalize_box(box, orig_size, target_size=(50, 50)):
    ow, oh = orig_size  # original data
    x, y, w, h = box
    x *= target_size[0]; y *= target_size[1]  # scale back down
    w *= target_size[0]; h *= target_size[1]
    xmin, ymin = x - w/2, y - h/2      # from center based back to corner beased
    xmax, ymax = x + w/2, y + h/2
    scale_x, scale_y = ow / target_size[0], oh / target_size[1]  # scaling it back to original size with the ratio
    return int(xmin*scale_x), int(ymin*scale_y), int(xmax*scale_x), int(ymax*scale_y)

def testModels(img_path):

    boxModel = keras.models.load_model("BoxModel_sign_language.keras")
    ClassificationModel = keras.models.load_model("SignLanguageClassifier.keras")
    img = Image.open(img_path).convert("RGB")
    img_resized = img.resize((50,50))
    img_array = np.array(img_resized) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    pred = boxModel.predict(img_array)[0]
    box_coords = denormalize_box(pred,img.size)
    box_img =  img.crop(box_coords).resize((50,50))
    box_img_array = np.expand_dims(np.array(box_img)/ 255.0, axis = 0)

    classification = ClassificationModel.predict(box_img_array)[0]
    draw = ImageDraw.Draw(img)
    draw.rectangle(box_coords, outline="red", width=1)
    label_idx = np.argmax(classification)  
    label_names = class_names 
    label = f"{label_names[label_idx]}: {classification[label_idx]:.2f}"
    print(label)
    
    try:
        font = ImageFont.truetype("arial.ttf", 10)  
        
    except:
        font = ImageFont.load_default()
    text_x = box_coords[0]
    text_y = max(box_coords[1] - 10 , 0)
    draw.text((text_x, text_y), label, fill="red", font=font)

    plt.imshow(img)
    plt.axis("off")
    plt.show()
        

testModels("1.jpg")

# Evaluation what to do better for next time, and how to improve the model

# 1 Step: Better Dataset
# - Dataset with more variety, like multiple people and a more advanced data augmentation
# - better Dataset for the Box modelling and better training parameters
# - works 100% for the Dataset images (obviously) but bounding boxes model works also
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
    layers.Input(shape=input_shape),
    layers.Conv2D(64, 3, activation="relu"),
    layers.MaxPooling2D(),
    layers.Conv2D(128, 3, activation="relu"),
    layers.MaxPooling2D(),
    layers.Conv2D(256, 3, activation="relu"),
    layers.MaxPooling2D(),
    layers.Conv2D(512, 3, activation="relu"),
    layers.GlobalAveragePooling2D(),
    layers.Dense(256, activation="relu"),
    layers.Dropout(0.3),
    layers.Dense(10, activation="linear")  
])
TARGET_SIZE = (256, 256)

multipleBoundingBoxesModel.compile(
     optimizer="adam",
    loss="mse"
)

img_dir = r"C:\Users\Henri\Videos\VOC_data\VOCdevkit\VOC2012\JPEGImages"
xml = r"C:\Users\Henri\Videos\VOC_data\VOCdevkit\VOC2012\Annotations"

def parse_annotation(xml_file):
    tree = ET.parse(xml_file)
    root = tree.getroot()
    
    filename = root.find("filename").text
    size = root.find("size")
    orig_w = int(size.find("width").text)
    orig_h = int(size.find("height").text)
    
    boxes = []
    labels = []
    
    for obj in root.findall("object"):
        name = obj.find("name").text.lower()
        bbox = obj.find("bndbox")
        
        xmin = int(float(bbox.find("xmin").text))
        ymin = int(float(bbox.find("ymin").text))
        xmax = int(float(bbox.find("xmax").text))
        ymax = int(float(bbox.find("ymax").text))
        
        center_x = (xmin + xmax) / 2
        center_y = (ymin + ymax) / 2
        width = (xmax - xmin) * 0.6
        height = (ymax - ymin) * 1.2           # implement resizing if not working properly
        
        new_xmin = center_x - width / 2
        new_ymin = center_y - height / 2
        new_xmax = center_x + width / 2
        new_ymax = center_y + height / 2

        x_center = ((new_xmin + new_xmax) / 2) / orig_w
        y_center = ((new_ymin + new_ymax) / 2) / orig_h
        width = (new_xmax - new_xmin) / orig_w
        height = (new_ymax - new_ymin) / orig_h
        
        boxes.append([x_center, y_center, width, height, 1.0])
        labels.append(name)
    
    if len(boxes) < 2:
        return None
    
    combined = list(zip(boxes, labels))
    combined.sort(key=lambda x: x[0][0])  
    boxes, labels = zip(*combined)


    boxes = boxes[:2]
    labels = labels[:2]

    flattened_boxes = []
    for box in boxes:
        flattened_boxes.extend(box)

    return filename, flattened_boxes, labels
def load_data(percentage=0.6):  
    global img_dir, xml
    X = []
    y = []
    
    xml_files = [f for f in os.listdir(xml) if f.endswith(".xml")]
    
    num_files = int(len(xml_files) * percentage)
    xml_files = xml_files[:num_files]
    
    print(f"Loading {num_files} files ({percentage*100}% of total {len(os.listdir(xml))} files)")
    
    for i, xml_file in enumerate(xml_files):
        if i % 1000 == 0:
            print(f"Processed {i}/{num_files} files")
            
        full_xml = os.path.join(xml, xml_file)
        try:
            result  = parse_annotation(full_xml)
            if result is None:
                continue
            filename, bbox, label = result
            img_path = os.path.join(img_dir, filename)
            if not os.path.exists(img_path):
                continue

            img = Image.open(img_path).convert("RGB")
            img = img.resize(TARGET_SIZE)
            img_arr = np.array(img) / 255.0

            X.append(img_arr)
            y.append(bbox)
        except Exception as e:
            print(f"Error processing {xml_file}: {e}")
            continue

    print(f"Successfully loaded {len(X)} images")
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)

# X,Y = load_data()

# earlyStopping = keras.callbacks.EarlyStopping(
#     monitor="val_loss",
#     patience=10,
#     restore_best_weights = True
# )

# multipleBoundingBoxesModel.fit(X,Y,epochs=70, validation_split=0.2, callbacks=[earlyStopping])
# multipleBoundingBoxesModel.save("Multiple_Bounding_Boxes_Model.keras")

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
    classificationModel = keras.models.load_model("Cat_Dog_Classifier4.keras")
    boxModel = keras.models.load_model("Multiple_Bounding_Boxes_Model.keras")
    
    img = Image.open(img_path).convert("RGB")
    img_copy = img.copy()  
    
    img_resized = img.resize((256,256))
    img_array = np.array(img_resized) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    pred = boxModel.predict(img_array)[0]
    box1 = pred[0:4]
    conf1 = pred[4]
    box2 = pred[5:9]
    conf2 = pred[9]
    
    draw = ImageDraw.Draw(img_copy)

    print(f"Box 1 - Confidence: {conf1:.3f}, Coordinates: {box1}")
    print(f"Box 2 - Confidence: {conf2:.3f}, Coordinates: {box2}")

    if conf1 > 0.3:
        box_coords_1 = denormalize_box(box1, img.size)
        print(f"Box 1 denormalized: {box_coords_1}")
        
        box_img1 = img.crop(box_coords_1).resize((256,256))
        box_img1_arr = np.expand_dims(np.array(box_img1) / 255.0, axis=0)
        classification_img_1 = classificationModel.predict(box_img1_arr)[0]

        draw.rectangle(box_coords_1, outline="red", width=3)
        label_idx1 = np.argmax(classification_img_1)
        label_names = ["Cat", "Dog"]
        label1 = f"{label_names[label_idx1]}: {classification_img_1[label_idx1]:.2f}"

        font_size = max(20, min(img.size) // 15)
        try:
            font = ImageFont.truetype("arial.ttf", font_size)
        except:
            font = ImageFont.load_default()

        text_x_1 = box_coords_1[0]
        text_y_1 = max(box_coords_1[1] - font_size, 0)
        draw.text((text_x_1, text_y_1), label1, fill="red", font=font)

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

    plt.imshow(img_copy)
    plt.axis("off")
    plt.show()


testMultiObjectDetection("Hund_katze2.jpeg")


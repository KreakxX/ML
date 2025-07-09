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
    layers.Dense(8, activation="linear")  
])
TARGET_SIZE = (256, 256)

# model compile
multipleBoundingBoxesModel.compile(
     optimizer="adam",
    loss="mse"
)



img_dir = r"C:\Users\Henri\Videos\VOC_data\VOCdevkit\VOC2012\JPEGImages"
xml = r"C:\Users\Henri\Videos\VOC_data\VOCdevkit\VOC2012\Annotations"

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

        # Convert to float first, then to int to handle decimal values
        xmin = int(float(bbox.find("xmin").text))
        ymin = int(float(bbox.find("ymin").text))     # and from this elemtn get all teh positionx left top corner and bottom right corner
        xmax = int(float(bbox.find("xmax").text))
        ymax = int(float(bbox.find("ymax").text))

        x_center = ((xmin + xmax) / 2) / orig_w     # normalizinh the box like 0 - 1 so neural nets can train better with it and get the center point of the Box
        y_center = ((ymin + ymax) / 2) / orig_h   # here also     
        width = (xmax - xmin) / orig_w      # and the width and height relative to the img size
        height = (ymax - ymin) / orig_h

        return filename, [x_center, y_center, width, height], name

def load_data(percentage=0.3):  
    global img_dir, xml
    X = []
    y = []
    
    xml_files = [f for f in os.listdir(xml) if f.endswith(".xml")]
    
    # Nur einen Teil der Dateien nehmen
    num_files = int(len(xml_files) * percentage)
    xml_files = xml_files[:num_files]
    
    print(f"Loading {num_files} files ({percentage*100}% of total {len(os.listdir(xml))} files)")
    
    for i, xml_file in enumerate(xml_files):
        if i % 1000 == 0:  # Progress indicator
            print(f"Processed {i}/{num_files} files")
            
        full_xml = os.path.join(xml, xml_file)
        try:
            filename, bbox, label = parse_annotation(full_xml)
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

X,Y = load_data()

earlyStopping = keras.callbacks.EarlyStopping(
    monitor="val_loss",
    patience=15,
    restore_best_weights = True
)

multipleBoundingBoxesModel.fit(X,Y,epochs=100, validation_split=0.2, callbacks=[earlyStopping])
multipleBoundingBoxesModel.save("Multiple_Bounding_Boxes_Model")

def denormalize_box(box, orig_size, target_size=(256, 256)):
    ow, oh = orig_size  # original data
    x, y, w, h = box
    x *= target_size[0]; y *= target_size[1]  # scale back down
    w *= target_size[0]; h *= target_size[1]
    xmin, ymin = x - w/2, y - h/2      # from center based back to corner beased
    xmax, ymax = x + w/2, y + h/2
    scale_x, scale_y = ow / target_size[0], oh / target_size[1]  # scaling it back to original size with the ratio
    return int(xmin*scale_x), int(ymin*scale_y), int(xmax*scale_x), int(ymax*scale_y)


def testMultiObjectDetection(img_path):
  classificationModel = keras.models.load_model("Cat_Dog_Classifier4.keras")
  boxModel = keras.models.load_model("Multiple_Bounding_Boxes_Model")
  img = Image.open(img_path).convert("RGB")
  img_resized = img.resize((256,256))
  img_array = np.array(img_resized) / 255.0
  img_array = np.expand_dims(img_array, axis=0)

  pred = boxModel.predict(img_array)[0]
  box1 = pred[0:4] 
  box2 = pred[4:8]  

  box_coords_1 = denormalize_box(box1,img.size)
  box_coords_2 = denormalize_box(box2,img.size)

  box_img1 = img.crop(box_coords_1).resize((256,256))
  box_img2 = img.crop(box_coords_2).resize((256,256))
  

  box_img1_arr = np.expand_dims(np.array(box_img1) / 255.0, axis = 0)
  box_img2_arr = np.expand_dims(np.array(box_img2) / 255.0, axis = 0)

  classification_img_1 = classificationModel.predict(box_img1_arr)[0]
  classification_img_2 = classificationModel.predict(box_img2_arr)[0]

  draw = ImageDraw.Draw(img)
  print(box_coords_1)
  print(box_coords_2)
  draw.rectangle(box_coords_1, outline="red", width=3)
  draw.rectangle(box_coords_2, outline="red", width=3)

  label_idx1 = np.argmax(classification_img_1)     
  label_idx2 = np.argmax(classification_img_2)
  label_names = ["Cat", "Dog"] 
  label1 = f"{label_names[label_idx1]}: {classification_img_1[label_idx1]:.2f}"
  label2 = f"{label_names[label_idx2]}: {classification_img_2[label_idx2]:.2f}"

  img_width, img_height = img.size
  font_size = max(20, min(img_width, img_height) // 15)
  try:
    font = ImageFont.truetype("arial.ttf", font_size)  
    
  except:
    font = ImageFont.load_default()
  text_x_1 = box_coords_1[0]
  text_y_1 = max(box_coords_1[1] - font_size , 0)
  text_x_2 = box_coords_2[0]
  text_y_2 = max(box_coords_2[1] - font_size , 0)
  draw.text((text_x_1, text_y_1), label1, fill="red", font=font)
  draw.text((text_x_2, text_y_2), label2, fill="red", font=font)

  plt.imshow(img)
  plt.axis("off")
  plt.show()

# testMultiObjectDetection("Dog_Cat.png")


import keras
from keras import layers
import numpy as np
import tensorflow_datasets as tfds
import tensorflow as tf
from PIL import Image
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

dataset, info = tfds.load('oxford_iiit_pet', with_info=True, as_supervised=True)
(train_raw, val_raw, test_raw), ds_info = tfds.load(name='oxford_iiit_pet',
                                                    split=['train[:90%]',
                                                          'train[90%:]',
                                                          'test'],
                                                    shuffle_files=True,
                                                    as_supervised=True,
                                                    with_info=True #
                                                    )

data_augmentation = keras.Sequential(
    [layers.RandomFlip('horizontal'),
      layers.RandomRotation(factor=(-0.025, 0.025)),
      layers.RandomTranslation(height_factor=0.1, width_factor=0.1),
      layers.RandomContrast(factor=0.1),
     ])
num_classes = ds_info.features['label'].num_classes

def one_hot_encode(image, label):
    label = tf.one_hot(label, num_classes)
    return image, label

train_ds = train_raw.map(lambda x, y: (tf.image.resize(x, (224, 224)), y))
val_ds = val_raw.map(lambda x, y: (tf.image.resize(x, (224, 224)), y))
test_ds = test_raw.map(lambda x, y: (tf.image.resize(x, (224, 224)), y))

train_ds = train_ds.map(one_hot_encode)
val_ds = val_ds.map(one_hot_encode)
test_ds = test_ds.map(one_hot_encode)

train_ds = train_ds.batch(batch_size=32,
                          drop_remainder=True).prefetch(tf.data.AUTOTUNE)

val_ds = val_ds.batch(batch_size=32,
                      drop_remainder=True).prefetch(tf.data.AUTOTUNE)

test_ds = test_ds.batch(batch_size=32,
                        drop_remainder=True).prefetch(tf.data.AUTOTUNE)

base_model = keras.applications.ResNet50V2(
                            include_top=False,
                            weights="imagenet",
                            input_shape=(224, 224, 3)
                            )

base_model.trainable = False

inputs = keras.Input(shape=(224, 224, 3))
x = data_augmentation(inputs)
x = keras.applications.resnet_v2.preprocess_input(x)
x = base_model(x, training=False)

x = layers.GlobalAveragePooling2D()(x)
x = layers.BatchNormalization()(x)
x = layers.Dropout(0.2)(x)
#
# outputs = layers.Dense(37, activation="softmax", name="pred")(x)
#
# model = keras.Model(inputs, outputs)
#
# model.compile(
#     optimizer=keras.optimizers.Adam(),
#     loss=keras.losses.CategoricalCrossentropy(),
#     metrics=[
#        keras.metrics.CategoricalAccuracy(),
#        keras.metrics.TopKCategoricalAccuracy(k=3, name='top_3_accuracy')
#      ]
# )
#
# earlystopping = keras.callbacks.EarlyStopping(monitor='val_loss',
#                                         mode='min',
#                                         patience=10,
#                                         restore_best_weights=True)
#
# history = model.fit(train_ds, epochs=25, validation_data=val_ds, verbose=1,
#                     callbacks =[earlystopping])
#
# base_model.trainable = True
# for layer in base_model.layers[:-100]:
#     layer.trainable = False
#
# model.compile(
#     optimizer=keras.optimizers.Adam(1e-4),
#     loss=keras.losses.CategoricalCrossentropy(),
#     metrics=[
#         keras.metrics.CategoricalAccuracy(),
#         keras.metrics.TopKCategoricalAccuracy(k=3, name='top_3_accuracy')
#     ]
# )
#
# history2 = model.fit(train_ds, epochs=25, validation_data=val_ds, verbose=1,
#                     callbacks =[earlystopping])
#
# model.save("Pet_Classifier2Remakev2.keras")


def confusionMatrix():
    class_names = ds_info.features['label'].names
    model = keras.models.load_model("Pet_Classifier2Remake.keras")
    y_true = []
    y_pred = []

    for images, labels in test_ds:
        preds = model.predict(images)
        y_pred.extend(np.argmax(preds, axis=1))
        y_true.extend(np.argmax(labels.numpy(), axis=1))

    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    disp.plot(xticks_rotation='vertical')
    plt.tight_layout()
    plt.show()


def checkImage(img_path):
    class_names = ds_info.features['label'].names
    model = keras.models.load_model("Pet_Classifier2Remake.keras")
    img = Image.open(img_path).convert("RGB")
    img = img.resize((224, 224))
    img_array = np.array(img).astype("float32")

    img_tensor = tf.convert_to_tensor(img_array)
    img_tensor = tf.expand_dims(img_tensor, axis=0)
    img_array = data_augmentation(img_tensor).numpy()[0]

    preprocessed_img = img_array

    input_batch = np.expand_dims(preprocessed_img, axis=0)

    predictions = model.predict(input_batch)
    pred_label = np.argmax(predictions[0])
    confidence = np.max(predictions[0])

    plt.figure(figsize=(8, 6))
    plt.imshow(img)
    plt.title(f"Predicted: {class_names[pred_label]} ({confidence:.2f})")
    plt.axis("off")
    plt.show()

    top_indices = np.argsort(predictions[0])[-3:][::-1]  # Sortieren der Vorhersagen und Top 3 nehmen
    print("Top predictions:")
    for idx in top_indices:
        print(f"  {class_names[idx]}: {predictions[0][idx]:.4f}")


#checkImage("Jan2jpg.jpg")

def evaluateModel():
    class_names = ds_info.features['label'].names
    model = keras.models.load_model("Pet_Classifier2Remake.keras")
    for images, labels in test_ds.take(1):
        predictions = model.predict(images)

        for i in range(min(5, len(images))):
            true_label = np.argmax(labels[i])

            pred_label = np.argmax(predictions[i])
            confidence = np.max(predictions[i])

            plt.figure(figsize=(8, 4))

            display_img = images[i].numpy()
            display_img = display_img + [123.68, 116.78, 103.94]
            display_img = np.clip(display_img, 0, 255).astype('uint8')

            plt.imshow(display_img)
            plt.title(f"True: {class_names[true_label]}, Pred: {class_names[pred_label]} ({confidence:.2f})")
            plt.axis('off')
            plt.show()

            top_indices = np.argsort(predictions[i])[-3:][::-1]
            print(f"True class: {class_names[true_label]}")
            print("Top predictions:")
            for idx in top_indices:
                print(f"  {class_names[idx]}: {predictions[i][idx]:.4f}")
            print("\n")

#evaluateModel()
def FaceRecognition():
    train_ds = keras.preprocessing.image_dataset_from_directory(
        "dataset",
        validation_split=0.2,
        subset="training",
        seed=123,
        image_size=(224, 224),
        batch_size=32,
        label_mode='int'
    )

    val_ds = keras.preprocessing.image_dataset_from_directory(
        "dataset",
        validation_split=0.2,
        subset="validation",
        seed=123,
        image_size=(224, 224),
        batch_size=32,
        label_mode='int'
    )
    num_classes = len(train_ds.class_names)

    AUTOTUNE = tf.data.AUTOTUNE
    train_ds = train_ds.prefetch(buffer_size=AUTOTUNE)
    val_ds = val_ds.prefetch(buffer_size=AUTOTUNE)


    def one_hot_encode(image, label):
        label = tf.one_hot(label, num_classes)
        return image, label
    train_ds = train_ds.map(one_hot_encode)
    val_ds = val_ds.map(one_hot_encode)

    data_augmentation = keras.Sequential(
        [layers.RandomFlip('horizontal'),
         layers.RandomRotation(factor=(-0.025, 0.025)),  # For better generalisation of the images == higher val_accuracy
         layers.RandomTranslation(height_factor=0.1, width_factor=0.1),
         layers.RandomContrast(factor=0.1),
         ])

    base_model = keras.applications.ResNet50V2(
        include_top=False,
        weights="imagenet",
        input_shape=(224, 224, 3)
    )

    base_model.trainable = False
    inputs = keras.Input(shape=(224, 224, 3))
    x = data_augmentation(inputs)
    x = keras.applications.resnet_v2.preprocess_input(x)
    x = base_model(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.2)(x)
    outputs = layers.Dense(2, activation="softmax", name="pred")(x)
    model = keras.Model(inputs, outputs)
    model.compile(
        optimizer=keras.optimizers.Adam(),
        loss=keras.losses.CategoricalCrossentropy(),
        metrics=[
           keras.metrics.CategoricalAccuracy(),
           keras.metrics.TopKCategoricalAccuracy(k=3, name='top_3_accuracy')
         ]
    )

    history = model.fit(train_ds, epochs=25, validation_data=val_ds, verbose=1)

    base_model.trainable = True
    for layer in base_model.layers[:-100]:
        layer.trainable = False

    model.compile(
        optimizer=keras.optimizers.Adam(1e-4),
        loss=keras.losses.CategoricalCrossentropy(),
        metrics=[
            keras.metrics.CategoricalAccuracy(),
            keras.metrics.TopKCategoricalAccuracy(k=3, name='top_3_accuracy')
        ]
    )

    history2 = model.fit(train_ds, epochs=25, validation_data=val_ds, verbose=1,)

    model.save("FaceRecognition.keras")


#FaceRecognition()


def checkFaceRecognition(img_path):
    model = keras.models.load_model("FaceRecognition.keras")
    img = Image.open(img_path).convert("RGB")
    img = img.resize((224, 224))
    img_array = np.array(img).astype("float32")

    img_tensor = tf.convert_to_tensor(img_array)
    img_tensor = tf.expand_dims(img_tensor, axis=0)
    img_array = data_augmentation(img_tensor).numpy()[0]

    preprocessed_img = img_array

    input_batch = np.expand_dims(preprocessed_img, axis=0)
    class_names = ["henrik", "not_henrik"]

    predictions = model.predict(input_batch)
    pred_label = np.argmax(predictions[0])
    confidence = np.max(predictions[0])

    plt.figure(figsize=(8, 6))
    plt.imshow(img)
    plt.title(f"Predicted: {class_names[pred_label]} with {confidence*100:.2f}% confidence")
    plt.axis("off")
    plt.show()

#checkFaceRecognition("Henrik.jpg")

def generateMathData(n_samples=10000):
    a = np.random.uniform(-10, 10, size=(n_samples,)) # generating a random number between -10 and 10 for n_samples times in this case 10k
    b = np.random.uniform(-10, 10, size=(n_samples,)) # generating a random number for b and so on
    x_true = np.random.uniform(-10, 10, size=(n_samples,)) # aswell as for the x_true, which we are going to sovle
    c = a * x_true + b  # now we are building out equation
    X = np.stack([a, b, c], axis=1)
    y = x_true
    return X, y  # so we get an array of the inputs a b c and the thing we are solving for x_true


def MathSolver():
    X_train, y_train = generateMathData(100000)
    X_val, y_val = generateMathData(10000)

    model = keras.Sequential([ # this aswell
        keras.Input(shape=(3,)),  # this is necessary
        layers.Dense(256, activation='relu'),
        layers.Dense(256, activation='relu'),
        layers.Dense(256, activation='relu'),

        layers.Dense(1)
    ])

    model.compile(optimizer="adam", loss="mse")

    model.fit(X_train, y_train, epochs=65, batch_size=32,validation_data=(X_val, y_val))
    model.save("MathEquationSolver.keras")

#MathSolver()

def testMath():
    model = keras.models.load_model("MathEquationSolver.keras")
    test_equation = np.array([[4, 6, 3]]) # 2x +3 = 3
    prediction = model.predict(test_equation)
    print(f"Lösung: x = {prediction[0, 0]:.1f}")


#testMath()
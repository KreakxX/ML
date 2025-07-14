import keras
from keras import layers, models
import tensorflow as tf


# Generator model
generator = keras.Sequential([
    layers.Input(shape=(100,)),  # z_dim = 100   Random noise vector with 100 values like [0.5,-1,-2,-2.3]
    layers.Dense(4 * 4 * 512), # makes the 100 values to 8192 
    layers.Reshape((4, 4, 512)), # makes the 8192 values to a 4x4 image with 512 canals
    layers.BatchNormalization(),      # Batch Normalizations is important for outbreaking features to minimize loss
    layers.ReLU(),
    
    layers.Conv2DTranspose(256, 5, 2, padding='same'),  # 4x4 -> 8x8    # resizing the image from 4x4 to 8x8
    layers.BatchNormalization(),
    layers.ReLU(),                              # and so on .....
      
    layers.Conv2DTranspose(128, 5, 2, padding='same'),  # 8x8 -> 16x16
    layers.BatchNormalization(),
    layers.ReLU(),  # makes negative values to zero, for better and brighter features
    
    layers.Conv2DTranspose(64, 5, 2, padding='same'),   # 16x16 -> 32x32
    layers.BatchNormalization(),
    layers.ReLU(),
    
    layers.Conv2DTranspose(3, 5, 2, padding='same', activation='tanh'),  # 32x32 -> 64x64
])

# output format of the generator
input_shape = (64,64,3)


# Discriminator model
discriminator = keras.Sequential([
        layers.Input(shape=input_shape),
        layers.Conv2D(64, 4, strides=2, padding='same'),  # a Convolutional layer with 64 filters with a 4x4 size
        layers.LeakyReLU(alpha=0.2),      # is there for making the values not null but smaller for example it takes a negative input -2 and than times the alpha value to minimize loss
        
        layers.Conv2D(128, 4, strides=2, padding='same'),
        layers.BatchNormalization(), # normalizing the values
        layers.LeakyReLU(alpha=0.2),
        
        layers.Conv2D(256, 4, strides=2, padding='same'),   # strides makes the difference for the steps, for example =2 makes jumping 2px worth
        layers.BatchNormalization(),
        layers.LeakyReLU(alpha=0.2),
        
        layers.Conv2D(512, 4, strides=2, padding='same'), # padding adds zeros, so the output size is right 
        layers.BatchNormalization(),
        layers.LeakyReLU(alpha=0.2),
        
        layers.Conv2D(1024, 4, strides=2, padding='same'),
        layers.BatchNormalization(),
        layers.LeakyReLU(alpha=0.2),
        
        layers.Flatten(),
        layers.Dense(1, activation='sigmoid')
])

# The Difference between LeakyReLu and Relu is the generator should only generate good features thats 0 ... 1 etc and the
# Discriminator should know and rate all informations like negative but we make it smaller for better performance

# method for preprocessing and normalising the images
def preprocessImages(image):
    image = tf.cast(image, tf.float32)
    image = (image / 127.5) - 1.0
    return image


# path for the dataset
path = r"C:\Users\Henri\Downloads\archive\img_align_celeba\img_align_celeba"


# load and generate the Dataset
celeba = tf.keras.utils.image_dataset_from_directory(
    path,
    labels=None,
    image_size=(64, 64),
    batch_size=32,
    shuffle=True
).map(preprocessImages)


# Model compiling
generator.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.0002, beta_1=0.5),
    loss='binary_crossentropy',
    metrics=['accuracy']
)

discriminator.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.0002, beta_1=0.5),
    loss='binary_crossentropy',
    metrics=['accuracy']
)


def train_gan(generator, discriminator, dataset, epochs=100, z_dim=100):
    # looping each epoch
    for epoch in range(epochs):
        print(f"Epoch {epoch+1}/{epochs}")
        
        for batch in dataset:   # and iterate through each batch in the dataset
            batch_size = tf.shape(batch)[0] # take tthe first batch
            
            # Training of the discriminator to rate if the image looks real or not real images
            real_images = batch   # and take the images
            real_labels = tf.ones((batch_size, 1))  
            
            # generating fake images
            noise = tf.random.normal([batch_size, z_dim])
            fake_images = generator(noise, training=False)
            fake_labels = tf.zeros((batch_size, 1))

            # training the discriminator
            d_loss_real = discriminator.train_on_batch(real_images, real_labels)
            d_loss_fake = discriminator.train_on_batch(fake_images, fake_labels)
            

            # Training of the generator the generator trys to fool the discriminator
            noise = tf.random.normal([batch_size, z_dim])
            misleading_labels = tf.ones((batch_size, 1))
            
            # freeze the discriminator training when training the generator
            discriminator.trainable = False
            
            # training the generator
            g_loss = discriminator.train_on_batch(generator(noise), misleading_labels)
            
            # and unfreezing after
            discriminator.trainable = True
        
        print(f"D_loss_real: {d_loss_real[0]:.4f}, D_loss_fake: {d_loss_fake[0]:.4f}, G_loss: {g_loss[0]:.4f}")


# training
train_gan(generator, discriminator, celeba, epochs=20)

# saving the model
discriminator.save("models/discriminator")
generator.save("models/generator")

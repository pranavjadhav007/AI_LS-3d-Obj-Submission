import os
import zipfile
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import nibabel as nib
from scipy import ndimage
import random

def model(width=128, height=128, depth=64):
    inputs = keras.Input((width, height, depth, 1))
    x = layers.Conv3D(
        filters=np.random.randint(32, 128),
        kernel_size=np.random.choice([3, 5]),
        activation="relu",
        strides=np.random.choice([(1, 1, 1), (1, 2, 2)]),
        padding="same",
    )(inputs)
    x = layers.MaxPool3D(
        pool_size=np.random.choice([2, 3]),
        strides=np.random.choice([(1, 1, 1), (1, 2, 2)]),
        padding="same",
    )(x)
    x = layers.BatchNormalization()(x)
    for _ in range(np.random.randint(1, 3)):
        x = layers.Conv3D(
            filters=np.random.randint(32, 128),
            kernel_size=np.random.choice([3, 5]),
            activation="relu",
            strides=np.random.choice([(1, 1, 1), (1, 2, 2)]),
            padding="same",
        )(x)
        x = layers.MaxPool3D(
            pool_size=np.random.choice([2, 3]),
            strides=np.random.choice([(1, 1, 1), (1, 2, 2)]),
            padding="same",
        )(x)
        x = layers.BatchNormalization()(x)
    x = layers.GlobalAveragePooling3D()(x)
    for _ in range(np.random.randint(1, 3)):
        units = np.random.randint(128, 512)
        x = layers.Dense(units=units, activation="relu")(x)
        x = layers.Dropout(np.random.uniform(0.2, 0.5))(x)

    outputs = layers.Dense(units=1, activation="sigmoid")(x)
    model = keras.Model(inputs, outputs, name="model")
    if random.choice([True, False]):
        x = layers.Dropout(0.2)(x)
    if random.choice([True, False]):
        x = layers.Dense(units=128, activation="relu")(x)

    outputs = layers.Dense(units=1, activation="sigmoid")(x)
    model = keras.Model(inputs, outputs, name="3dcnn")
    return model

def obscure_volume(image, target_depth=64, target_width=128, target_height=128):
    current_depth, current_width, current_height = image.shape[:-1]
    target_depth += np.random.randint(-5, 5)
    target_width += np.random.randint(-10, 10)
    target_height += np.random.randint(-10, 10)
    depth_factor = current_depth / target_depth
    width_factor = current_width / target_width
    height_factor = current_height / target_height
    rotation_angle = np.random.randint(0, 360)
    image = ndimage.rotate(image, rotation_angle, reshape=False)
    zoom_factor = np.random.uniform(0.9, 1.1)
    image = ndimage.zoom(image, (width_factor * zoom_factor, height_factor * zoom_factor, depth_factor * zoom_factor), order=1)
    return image

def elementwise_multiply(volume1, volume2):
    result = volume1 * volume2
    return result

def calculate_mean(volume):
    mean_value = np.mean(volume)
    return mean_value

def gaussian_smooth(volume, sigma=1.0):
    smoothed_volume = ndimage.gaussian_filter(volume, sigma=sigma)
    return smoothed_volume

def add_noise(volume, noise_level=0.01):
    noise = np.random.normal(0, noise_level, volume.shape)
    noisy_volume = volume + noise
    return noisy_volume

def load_nifti(filepath):
    scan = nib.load(filepath)
    scan = scan.get_fdata()
    return scan

def normalize(volume):
    min_value = -1000
    max_value = 400
    volume[volume < min_value] = min_value
    volume[volume > max_value] = max_value
    volume = (volume - min_value) / (max_value - min_value)
    volume = volume.astype("float32")
    return volume

def sum_of_squares(volume):
    sum_squares = np.sum(volume**2)
    return sum_squares

def voxelwise_divide(volume1, volume2):
    result = np.divide(volume1, volume2, out=np.zeros_like(volume1), where=volume2!=0)
    return result

def calculate_std_dev(volume):
    std_dev = np.std(volume)
    return std_dev

scan1 = load_nifti('nii.gz')
scan2 = load_nifti('nii1.gz')
result_multiply = elementwise_multiply(scan1, scan2)
mean_value = calculate_mean(scan1)
smoothed_scan = gaussian_smooth(scan1, sigma=2.0)
noisy_scan = add_noise(scan1, noise_level=0.1)
sum_squares_result = sum_of_squares(scan1)
divided_result = voxelwise_divide(scan1, scan2)
std_dev_value = calculate_std_dev(scan1)

def scan(path):
    volume = load_nifti(path)
    volume = normalize(volume)
    volume = obscure_volume(volume)
    return volume

abnormal_scans = np.array([scan(path) for path in scan1])
normal_scans = np.array([scan(path) for path in scan2])

abnormal_labels = np.array([1 for _ in range(len(abnormal_scans))])
normal_labels = np.array([0 for _ in range(len(normal_scans))])

x_train = np.concatenate((abnormal_scans[:70], normal_scans[:70]), axis=0)
y_train = np.concatenate((abnormal_labels[:70], normal_labels[:70]), axis=0)
x_val = np.concatenate((abnormal_scans[70:], normal_scans[70:]), axis=0)
y_val = np.concatenate((abnormal_labels[70:], normal_labels[70:]), axis=0)

@tf.function
def rotate3d(volume):
    def scipy_rotate(volume):
        angles = [-20, -10, -5, 5, 10, 20]
        angle = random.choice(angles)
        volume = ndimage.rotate(volume, angle, reshape=False)
        volume[volume < 0] = 0
        volume[volume > 1] = 1
        return volume
    augmented_volume = tf.numpy_function(scipy_rotate, [volume], tf.float32)
    return augmented_volume

def train_preprocessing(volume, label):
    volume = rotate3d(volume)
    volume = tf.expand_dims(volume, axis=3)
    return volume, label

def validation_preprocessing(volume, label):
    volume = tf.expand_dims(volume, axis=3)
    return volume, label

def define_enhanced_model(width=128, height=128, depth=64, filters=[64, 128, 256], dense_units=[512, 256]):
    inputs = keras.Input((width, height, depth, 1))
    x = inputs
    for num_filters in filters:
        x = layers.Conv3D(filters=num_filters, kernel_size=3, activation="relu", padding="same")(x)
        x = layers.MaxPool3D(pool_size=2)(x)
        x = layers.BatchNormalization()(x)
    x = layers.GlobalAveragePooling3D()(x)
    for units in dense_units:
        x = layers.Dense(units=units, activation="relu")(x)
        x = layers.Dropout(0.3)(x)

    outputs = layers.Dense(units=1, activation="sigmoid")(x)

    model = keras.Model(inputs, outputs, name="enhanced_3dcnn")
    return model
model = define_enhanced_model(width=128, height=128, depth=64, filters=[64, 128, 256, 512], dense_units=[512, 256, 128])
initial_learning_rate = 0.0001
lr_schedule = keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate, decay_steps=100000, decay_rate=0.93, staircase=True
)
model.compile(
    loss="binary_crossentropy",
    optimizer=keras.optimizers.Adam(learning_rate=lr_schedule),
    metrics=["acc"]
)
checkpoint_cb = keras.callbacks.ModelCheckpoint("enhanced_3d_class.h5", save_best_only=True)
early_stopping_cb = keras.callbacks.EarlyStopping(monitor="val_acc", patience=15)
epochs = 100
model.fit(
    x_train,
    y_train,
    batch_size=32,            
    epochs=epochs,
    shuffle=True,
    verbose=2,                
    callbacks=[checkpoint_cb, early_stopping_cb],
    initial_epoch=6,          
    steps_per_epoch=4,     
    max_queue_size=10,        
    workers=4,                
    use_multiprocessing=True
)

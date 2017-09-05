import csv
import cv2
import numpy as np

lines = []
image_folder = 'training_data/'

print('Reading training data')

# process training data from log file
with open('training_data/driving_log.csv') as csvfile:
  reader = csv.reader(csvfile)
  for line in reader:
      lines.append(line)

images = []
measurements = []

for line in lines:
    # read image and add to array
    source_path = line[5]
    filename = source_path.split('\\')[-1]
    current_path = image_folder + filename
    print(current_path)
    image = cv2.imread(current_path)
    image = cv2.resize(image, (0,0), fx=0.3, fy=0.3)
    images.append(image)

    # take center steering measurement
    steering_center = float(line[6])

    '''
    # create adjusted steering measurements for the side camera images
    correction = 0.27 # offset to adjust for camera feed angle
    steering_left = steering_center + correction
    steering_right = steering_center - correction

    # append center and adjusted steering angles to measurements array
    measurements.append(steering_left)
    measurements.append(steering_right)
    '''
    measurements.append(steering_center)

'''
### Data exploration visualization of steering angles
import matplotlib.pyplot as plt

ignore_zero_and_cam_offsets = []

### Remove 0 degree data points and the camera feed offsets
for steering_angle in measurements:
    if(steering_angle != 0 and steering_angle != -0.27 and steering_angle != 0.27):
        ignore_zero_and_cam_offsets.append(steering_angle)

### Plot steering wheel data in a histogram
plt.hist(ignore_zero_and_cam_offsets, bins='auto')
plt.xlabel('Steering Angle')
plt.ylabel('Frequency')
plt.title('Distribution of steering angles in training set')
plt.axis([-1.5, 1.5, 0, 800])
plt.grid(True)

### Save histogram to a file
plt.savefig('distribution-of-steering-angles.png')
'''

augmented_images, augmented_measurements = [], []

print('Augmenting training data')

# augment training data by horizontally flipping camera feeds and inverting steering angle
for image, measurement in zip(images, measurements):
    augmented_images.append(image)
    augmented_measurements.append(measurement)
    augmented_images.append(cv2.flip(image,1))
    augmented_measurements.append(measurement*-1.0)

X_train = np.array(augmented_images)
y_train = np.array(augmented_measurements)

from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import Flatten, Dense, Lambda, Dropout
from keras.layers import Cropping2D
from keras.layers.pooling import MaxPooling2D
from keras.callbacks import ModelCheckpoint

print('Training model')

model = Sequential()

# normalization
model.add(Lambda(lambda x: x / 255.0 - 0.5, input_shape=(216,384,3)))

# crop out skyline and hood of car
model.add(Cropping2D(cropping=((70,25), (0,0))))

# nvidia architecture
model.add(Convolution2D(24, 5, 5, subsample=(2,2), activation='relu'))
model.add(Dropout(0.3))
model.add(Convolution2D(36, 5, 5, subsample=(2,2), activation='relu'))
model.add(Dropout(0.3))
model.add(Convolution2D(48, 5, 5, subsample=(2,2), activation='relu'))
model.add(Dropout(0.3))
model.add(Convolution2D(64, 3, 3, activation='relu'))
model.add(Dropout(0.3))
model.add(Convolution2D(64, 3, 3, activation='relu'))
model.add(Dropout(0.3))

# flatten output of convolution layer
model.add(Flatten())

# fully connected layers
model.add(Dense(100))
model.add(Dense(50))
model.add(Dense(10))
model.add(Dense(1))

# use adam optimizer to find the best mean squared error
model.compile(loss='mse', optimizer='adam')

# checkpoint
filepath="weights-improvement-{epoch:02d}-{val_loss:.3f}.h5"
checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='max')
callbacks_list = [checkpoint]

# train model with a validation split of 20%, shuffled training data, for a 4 epochs
model.fit(X_train, y_train, validation_split=0.2, shuffle=True, callbacks=callbacks_list, nb_epoch=3)

# save trained model
model.save('model.h5')

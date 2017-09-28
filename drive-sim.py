import csv
import cv2
import h5py
import numpy as np
import argparse

from keras.models import load_model
from keras import __version__ as keras_version

parser = argparse.ArgumentParser(description='Test drive a model against a dataset')
parser.add_argument('-m', '--model', type=str, nargs='?', default='model.h5', help='saved model')

args = parser.parse_args()
model = args.model

history_size = 5

lines = []
input_folder = 'data_training/'
output_folder = 'data_testdrive/'

steering_hist = np.zeros(history_size)

# process training data from log file
with open(input_folder + 'driving_log.csv') as csvfile:
  reader = csv.reader(csvfile)
  for line in reader:
      lines.append(line)

model = load_model(model)

font = cv2.FONT_HERSHEY_SIMPLEX

for line in lines:
    # read image and add to array
    source_path = line[5]
    filename = source_path.split('\\')[-1]
    current_path = input_folder + filename
    output_path = output_folder + filename

    image_raw = cv2.imread(current_path)
    image_raw = cv2.resize(image_raw, (0,0), fx=0.3, fy=0.3)

    # focus on area model was trained on
    image_pred = image_raw[86:216, 84:384]

    steering_angle = float(model.predict(image_pred[None, :, :, :], batch_size=1))

    steering_hist = np.roll(steering_hist, -1)
    steering_hist[history_size - 1] = steering_angle
    steering_smoothed = np.median(steering_hist)
    print('preditected', steering_smoothed)

    image_out = cv2.putText(image_raw, "{0:.3f}".format(steering_smoothed), (20, 40), font, 0.7, (255,255,255), 2)

    cv2.imwrite(output_path, image_out)

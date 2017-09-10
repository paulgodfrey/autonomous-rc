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

lines = []
input_folder = 'data_training/'
output_folder = 'data_testdrive/'

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

    # print('filename', filename)
    # print('input image', current_path)
    # print('output image', output_path)

    image = cv2.imread(current_path)
    image = cv2.resize(image, (0,0), fx=0.3, fy=0.3)
    image = image[86:216, 84:384]

    steering_angle = float(model.predict(image[None, :, :, :], batch_size=1))
    print('preditected', steering_angle)

    image = cv2.putText(image, str(steering_angle), (20, 40), font, 1, (255,255,255), 2)


    cv2.imwrite(output_path, image)

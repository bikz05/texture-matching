#!/usr/bin/python
# OpenCV bindings
import cv2
# To performing path manipulations 
import os
# Local Binary Pattern function
from skimage.feature import local_binary_pattern
# To calculate a normalized histogram 
from scipy.stats import itemfreq
from sklearn.preprocessing import normalize
# To read class from file
import csv
# For plotting
import matplotlib.pyplot as plt
# For array manipulations
import numpy as np
# For saving histogram values
from sklearn.externals import joblib
# For command line input
import argparse as ap
# Utility Package
import cvutils

# Get the path of the training set
parser = ap.ArgumentParser()
parser.add_argument("-t", "--trainingSet", help="Path to Training Set", required="True")
parser.add_argument("-l", "--imageLabels", help="Path to Image Label Files", required="True")
args = vars(parser.parse_args())

# Store the path of training images in train_images
train_images = cvutils.imlist(args["trainingSet"])
# Dictionary containing image paths as keys and corresponding label as value
train_dic = {}
with open(args['imageLabels'], 'rb') as csvfile:
    reader = csv.reader(csvfile, delimiter=' ')
    for row in reader:
        train_dic[row[0]] = int(row[1])

# List for storing the LBP Histograms, address of images and the corresponding label 
X_test = []
X_name = []
y_test = []

# For each image in the training set calculate the LBP histogram
# and update X_test, X_name and y_test
for train_image in train_images:
    # Read the image
    im = cv2.imread(train_image)
    # Convert to grayscale as LBP works on grayscale image
    im_gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    radius = 3
    # Number of points to be considered as neighbourers 
    no_points = 8 * radius
    # Uniform LBP is used
    lbp = local_binary_pattern(im_gray, no_points, radius, method='uniform')
    # Calculate the histogram
    x = itemfreq(lbp.ravel())
    # Normalize the histogram
    hist = x[:, 1]/sum(x[:, 1])
    # Append image path in X_name
    X_name.append(train_image)
    # Append histogram to X_name
    X_test.append(hist)
    # Append class label in y_test
    y_test.append(train_dic[os.path.split(train_image)[1]])

# Dump the  data
joblib.dump((X_name, X_test, y_test), "lbp.pkl", compress=3)
    
# Display the training images
nrows = 2
ncols = 3
fig, axes = plt.subplots(nrows,ncols)
for row in range(nrows):
    for col in range(ncols):
        axes[row][col].imshow(cv2.imread(X_name[row*ncols+col]))
        axes[row][col].axis('off')
        axes[row][col].set_title("{}".format(os.path.split(X_name[row*ncols+col])[1]))

# Convert to numpy and display the image
fig.canvas.draw()
im_ts = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
im_ts = im_ts.reshape(fig.canvas.get_width_height()[::-1] + (3,))
cv2.imshow("Training Set", im_ts)
cv2.waitKey()

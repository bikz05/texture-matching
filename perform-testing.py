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
parser.add_argument("-t", "--testingSet", help="Path to Testing Set", required="True")
parser.add_argument("-l", "--imageLabels", help="Path to Image Label Files", required="True")
args = vars(parser.parse_args())

# Store the path of training images in train_images
train_images = cvutils.imlist(args["testingSet"])
# Dictionary containing image paths as keys and corresponding label as value
train_dic = {}
with open('../data/lbp/class_test.txt', 'rb') as csvfile:
    reader = csv.reader(csvfile, delimiter=' ')
    for row in reader:
        train_dic[row[0]] = int(row[1])

# Load the List for storing the LBP Histograms, address of images and the corresponding label 
X_name, X_test, y_test = joblib.load("lbp.pkl")

# Store the path of testing images in test_images
test_images = cvutils.imlist("../data/lbp/test/")
# Dictionary containing image paths as keys and corresponding label as value
test_dic = {}
with open('../data/lbp/class_test.txt', 'rb') as csvfile:
    reader = csv.reader(csvfile, delimiter=' ')
    for row in reader:
        test_dic[row[0]] = int(row[1])

for test_image in test_images:
     # Read the image
    im = cv2.imread(test_image)
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
    # Display the query image
    cv2.imshow("** Query Image -> {}**".format(test_image), im)
    results = []
    # For each image in the training dataset
    # Calculate the chi-squared distance and the sort the values
    for index, x in enumerate(X_test):
        score = cv2.compareHist(np.array(x, dtype=np.float32), np.array(hist, dtype=np.float32), cv2.cv.CV_COMP_CHISQR)
        results.append((X_name[index], round(score, 3)))
    results = sorted(results, key=lambda score: score[1])
    # Display the results
    nrows = 2
    ncols = 3
    fig, axes = plt.subplots(nrows,ncols)
    fig.suptitle("** Scores for -> {}**".format(test_image))
    for row in range(nrows):
        for col in range(ncols):
            axes[row][col].imshow(cv2.imread(results[row*ncols+col][0]))
            axes[row][col].axis('off')
            axes[row][col].set_title("Score {}".format(results[row*ncols+col][1]))
    fig.canvas.draw()
    im_ts = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
    im_ts = im_ts.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    cv2.imshow("** Scores for -> {}**".format(test_image), im_ts)
    cv2.waitKey()
    cv2.destroyAllWindows()

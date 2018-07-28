# lbp_features.py
# Author: Neha Bhagwat
# Last updated: November 29, 2017

from skimage import feature
import numpy as np
# from sklearn.svm import learnSVC
# from imutils import paths
# import argparse
import cv2
import os
import csv


def writeToCSVFile(features, fileName):
    with open(fileName, 'a') as csv_file:
        writer = csv.writer(csv_file, dialect='excel')
        csv.field_size_limit(100000)
        # writer = csv.writer(csv_file)
        fileEmpty = os.stat(os.path.realpath(fileName)).st_size == 0
        if fileEmpty:
            writer.writerow(["Image", "f1", "f2", "f3","f4", "f5", "f6", "f7", "f8", "f9", "f10", "f11","f12","f13","f14","f15","f16",\
                             "f17","f18","f19","f20","f21","f22","f23","f24","f25","f26","age_class"])
            for feature in features:
                writer.writerow(feature)
        else:
            for feature in features:
                writer.writerow(feature)

class LBP:
    def __init__(self, num_of_points, radius):
        self.num_of_points = num_of_points
        self.radius = radius

    def create_histogram(self, image, eps = 1e-7):
        extracted_features = feature.local_binary_pattern(image, self.num_of_points, self.radius, method = "uniform" )
        print "Features extracted"
        hist, unused_arg = np.histogram(extracted_features.ravel(), bins = np.arange(0, self.num_of_points + 3), range = (0, self.num_of_points + 2))

        hist = hist.astype("float")

        hist = hist/(hist.sum() + eps)

        print "Histogram created"
        return hist


LBP_extract = LBP(24,8)
data = []
labels = []
folder_path = "C:\Users\\bhagw\Desktop\SJSU - SEM I\Topics_in_AI\Project\wiki\\test"

image_features = []
for path, subdirs, files in os.walk(folder_path):
    print files
    for filename in files:
        try:
            print filename
            num_values = filename.split('_')
            year_of_birth = int(num_values[1][0:4])
            year_of_image = int(num_values[2][0:4])
            age = int(year_of_image - year_of_birth)
            f = os.path.join(path, filename)
            # f = path + "\\" + filename
            print f
            ext = os.path.splitext(filename)[1]
            if ext.lower().find("jpeg") != -1 or ext.lower().find("jpg") != -1:
                print "here"
                image = cv2.imread(f)
                # print image.shape
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                hist = LBP_extract.create_histogram(gray)
                labels.append(filename)
                data.append(hist)
                age_class = 2
                if age <= 10:
                    age_class = 0
                elif age > 10 and age <= 20:
                    age_class = 1
                elif age > 20 and age <= 30:
                    age_class = 2
                elif age > 30 and age <= 40:
                    age_class = 3
                elif age > 40 and age <= 50:
                    age_class = 4
                elif age > 50 and age <= 60:
                    age_class = 5
                elif age > 60 and age <= 70:
                    age_class = 6
                elif age > 70 and age <= 80:
                    age_class = 7
                elif age > 80:
                    age_class = 8

                temp_list = []
                temp_list.append(filename)
                for ele in hist:
                    temp_list.append(ele)
                temp_list.append(age_class)
                # print temp_list
                image_features.append(temp_list)

        except Exception as e:
            print "Skipping image"


# print data
# for datum in data:
    # print len(datum)
# image = cv2.imread("C:\Users\\bhagw\Desktop\SJSU - SEM I\Topics_in_AI\Project\wiki\small_subset\\37500_1944-01-23_2010.jpg")
# print image

writeToCSVFile(image_features, "LBP_test.csv")
print image_features

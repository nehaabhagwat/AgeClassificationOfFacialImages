# USAGE
# python facial_features3.py --shape-predictor shape_predictor_68_face_landmarks.dat

# import the necessary packages
from imutils import face_utils
import numpy as np
import argparse
import imutils
import dlib
import cv2
import PIL
import math
import os
from operator import itemgetter
from os import listdir
import csv
import os.path

def calculate_distance(coord_1, coord_2):
    c1 = np.ndarray.tolist(coord_1)
    c2 = np.ndarray.tolist(coord_2)
    x1 = float(c1[0])
    y1 = float(c1[1])
    x2 = float(c2[0])
    y2 = float(c2[1])
    distance = math.sqrt(math.pow((x1 - x2), 2) + math.pow((y1 - y2), 2))
    return distance


def writeToCSVFile(features,fileName):
        with open(fileName, 'a') as csv_file:
           writer = csv.writer(csv_file, dialect='excel')
           csv.field_size_limit(100000) 
           #writer = csv.writer(csv_file)
           fileEmpty = os.stat(os.path.realpath(fileName)).st_size == 0
           if fileEmpty:
            writer.writerow(["Image", "Left_Eye_Right_Eye/Eye_Nose", "Left_Eye_Right_Eye/Eye_Mouth", "Left_Eye_Right_Eye/Eye_Chin", "Eye_Nose/Eye_Mouth", "Eye_Mouth/Eye_Chin", "Virtual_Top_Head/Eye_Chin", "Age_Class"])
            for feature in features:
                writer.writerow(feature)
           else:
            for feature in features:
                writer.writerow(feature)

features_list_train = []
source_folder_path = ""
class_type = 2
# folder_path = raw_input("Kindly enter the folder name from where the training images set has to be read --> ")
folder_path = "/Users/jishavarun/Documents/Python/CS256/project/Age_Classification/facial-landmarks/wiki_test"
image_features = []
for path, subdirs, files in os.walk(folder_path):
    for filename in files:
        f = os.path.join(path, filename)
        ext = os.path.splitext(filename)[1]
        if ext.lower().find("jpeg") != -1 or ext.lower().find("jpg") != -1:
         try:
            num_values = filename.split('_')
            year_of_birth = int(num_values[1][0:4])
            year_of_image = int(num_values[2][0:4])
            age = int(year_of_image - year_of_birth)
            ap = argparse.ArgumentParser()
            ap.add_argument("-p", "--shape-predictor", required=True,
                            help="path to facial landmark predictor")
            # ap.add_argument("-i", "--image", required=True,
            # help="path to input image")
            args = vars(ap.parse_args())

            shapePredictor = args["shape_predictor"]
            # print "shape_predictor", shapePredictor

            # initialize dlib's face detector (HOG-based) and then create
            # the facial landmark predictor
            detector = dlib.get_frontal_face_detector()
            predictor = dlib.shape_predictor(args["shape_predictor"])

            # load the input image, resize it, and convert it to grayscale
            # image = cv2.imread(args["image"])
            image = cv2.imread(f)
            image = imutils.resize(image, width=500)
            image_save = cv2.imwrite("savedimg.jpg", image)

            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

            # detect faces in the grayscale image
            rects = detector(gray, 1)

            # print "rects",rects

            # out_file = open("features.txt", 'w')

            # loop over the face detections
            for (i, rect) in enumerate(rects):
                # determine the facial landmarks for the face region, then
                # convert the facial landmark (x, y)-coordinates to a NumPy
                # array
                shape = predictor(gray, rect)
                shape = face_utils.shape_to_np(shape)
                np.ndarray.tofile(shape, "\n", "%s")
                # out_file.write(shape.tobytes())
                # print(type(shape))
                # print "shape",shape

                # convert dlib's rectangle to a OpenCV-style bounding box
                # [i.e., (x, y, w, h)], then draw the face bounding box
                (x, y, w, h) = face_utils.rect_to_bb(rect)
                cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
                # img = Image.open("image1.jpg")
                # img2 = img.crop((x, y, x+w, y+h))

                # img2 = img.crop((x, y), (x + w, y + h))
                # img2.show()
                img = cv2.imread("savedimg.jpg")

                crop_img = img[y:y + h, x:x + w]
                cv2.imshow("cropped", crop_img)

                image_resized = imutils.resize(crop_img, width=500)
                gray_resized = cv2.cvtColor(image_resized, cv2.COLOR_BGR2GRAY)
                rects_cropped = detector(gray_resized, 1)
                for (i, rect_cropped) in enumerate(rects_cropped):
                    # determine the facial landmarks for the face region, then
                    # convert the facial landmark (x, y)-coordinates to a NumPy
                    # array
                    shape_cropped = predictor(gray_resized, rect_cropped)
                    shape_cropped = face_utils.shape_to_np(shape_cropped)

                #Feature set2

                left_eye_right_eye = (calculate_distance(shape_cropped[37], shape_cropped[44]) + calculate_distance(shape_cropped[38], shape_cropped[43]))/2

                eye_nose = shape_cropped[35][1] - shape_cropped[45][1]

                ratio1 = left_eye_right_eye/abs(eye_nose)

                eye_mouth = shape_cropped[45][1] - shape_cropped[54][1]

                ratio2 = left_eye_right_eye/abs(eye_mouth)

                eye_chin = shape_cropped[45][1] - shape_cropped[8][1]

                ratio3 = left_eye_right_eye/abs(eye_chin)

                ratio4 = abs(eye_nose/eye_mouth)

                ratio5 = abs(eye_mouth/eye_chin)

                virtual_top_head = shape_cropped[24][1] - shape_cropped[8][1]

                ratio6 = abs(virtual_top_head/eye_chin)


                age_class = 2
                if age<=10:
                    age_class = 0
                elif age > 10 and age <=20:
                    age_class = 1
                elif age >20 and age <=30:
                    age_class = 2
                elif age > 30 and age <=40:
                    age_class = 3
                elif age >40 and age <=50:
                    age_class = 4
                elif age > 50 and age <=60:
                    age_class = 5
                elif age >60 and age <=70:
                    age_class = 6
                elif age > 70 and age <=80:
                    age_class = 7
                elif age >80:
                    age_class = 8


                #image_features.append([f,lip_width, eye1_width, eye2_width, nose_length, nose_width, eyebrow1_length, eyebrow2_length, age_class])
                image_features.append([filename, format(ratio1, '.2f'), format(ratio2, '.2f'), format(ratio3, '.2f'), format(ratio4, '.2f'), format(ratio5, '.2f'), format(ratio6, '.2f'), age_class])
                print filename, ratio1, ratio2, ratio3, ratio4, ratio5, ratio6, age_class
         except:
            continue

writeToCSVFile(image_features, "feture_set3_test.csv")


                

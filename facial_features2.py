# USAGE
# python facial_features2.py --shape-predictor shape_predictor_68_face_landmarks.dat

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

def calculate_mid(coord_1, coord_2):
    c1 = np.ndarray.tolist(coord_1)
    c2 = np.ndarray.tolist(coord_2)
    x1 = float(c1[0])
    y1 = float(c1[1])
    x2 = float(c2[0])
    y2 = float(c2[1])
    x = (x1 + x2)/2
    y = (y1+y2)/2
    #return np.ndarray([x,y])
    return np.array([x,y])

def writeToCSVFile(features,fileName):
        with open(fileName, 'a') as csv_file:
           writer = csv.writer(csv_file, dialect='excel')
           csv.field_size_limit(100000) 
           #writer = csv.writer(csv_file)
           fileEmpty = os.stat(os.path.realpath(fileName)).st_size == 0
           if fileEmpty:
            writer.writerow(["Image", "facial_index", "mandibular_index", "intercanthal_index", "orbital_width_index", "eye_fissure_index", "nasal_index", "vermilion_height_index", "mouth_face_width_index", "age_class"])
            for feature in features:
                writer.writerow(feature)
           else:
            for feature in features:
                writer.writerow(feature)

features_list_train = []
source_folder_path = ""
class_type = 2
# folder_path = raw_input("Kindly enter the folder name from where the training images set has to be read --> ")
folder_path = "/Users/jishavarun/Documents/Python/CS256/wiki"
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

                # Feature set 3

                facial_index = (calculate_distance(calculate_mid(shape_cropped[21],shape_cropped[22]),shape_cropped[8]))/calculate_distance(shape_cropped[1], shape_cropped[15])

                mandibular_index = (calculate_distance(calculate_mid(shape_cropped[62], shape_cropped[66]), shape_cropped[8]))/calculate_distance(shape_cropped[5], shape_cropped[11])

                intercanthal_index = (calculate_distance(shape_cropped[39], shape_cropped[42]))/calculate_distance(shape_cropped[36], shape_cropped[45])

                temp = (calculate_distance(shape_cropped[36], shape_cropped[39]) + calculate_distance(shape_cropped[42], shape_cropped[45]))/2
                orbital_width_index = temp/(calculate_distance(shape_cropped[39], shape_cropped[42]))

                temp = (calculate_distance(calculate_mid(shape_cropped[37], shape_cropped[38]), calculate_mid(shape_cropped[41], shape_cropped[40]))+ \
                 calculate_distance(calculate_mid(shape_cropped[43], shape_cropped[44]), calculate_mid(shape_cropped[47], shape_cropped[46])))/2

                eye_fissure_index = (temp/(calculate_distance(shape_cropped[36], shape_cropped[39]) + calculate_distance(shape_cropped[42], shape_cropped[45])))/2

                nasal_index = (calculate_distance(shape_cropped[31], shape_cropped[35]))/(calculate_distance(calculate_mid(shape_cropped[21], shape_cropped[22]), shape_cropped[33]))

                vermilion_height_index = (calculate_distance(shape_cropped[51], calculate_mid(shape_cropped[62], shape_cropped[66])))/(calculate_distance(calculate_mid(shape_cropped[62], shape_cropped[66]),shape_cropped[57]))

                mouth_face_width_index = (calculate_distance(shape_cropped[48], shape_cropped[54]))/(calculate_distance(shape_cropped[1], shape_cropped[15]))



               



                """features_list_input.append(
                    {'image_path': img, 'lip_width': lip_width, 'eye1_width': eye1_width, 'eye2_width': eye2_width,
                     'nose_length': nose_length, 'nose_width': nose_width, 'eyebrow1_length': eyebrow1_length,
                     'eyebrow2_length': eyebrow2_length, 'class_type': class_type})"""
                # print "features_list_input", features_list_input

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


                image_features.append([f,format(facial_index, '.2f'), format(mandibular_index, '.2f'), format(intercanthal_index, '.2f'), format(orbital_width_index, '.2f'), format(eye_fissure_index, '.2f'), format(nasal_index, '.2f'), format(vermilion_height_index, '.2f'), format(mouth_face_width_index, '.2f'), age_class])
                print f,format(facial_index, '.2f'), format(mandibular_index, '.2f'), format(intercanthal_index, '.2f'), format(orbital_width_index, '.2f'), format(eye_fissure_index, '.2f'), format(nasal_index, '.2f'), format(vermilion_height_index, '.2f'), format(mouth_face_width_index, '.2f'), age_class
         except:
            print "Skipping the image"

writeToCSVFile(image_features, "feture_set2.csv")
print image_features

                

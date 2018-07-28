import numpy as np
import cv2
import os
import csv
import os.path

EPS = 0.00000000000000001

def build_filters():
    filters = []
    ksize = 31
    for theta in np.arange(0, np.pi, np.pi/4):
        for lamb in np.arange(np.pi/4, np.pi, np.pi/4):
            kern = cv2.getGaborKernel((ksize, ksize), 4.0, theta, lamb, 0.5, 0, ktype=cv2.CV_32F)
            kern /= 1.5*kern.sum()
            filters.append(kern)

    return filters
 
def process(img, filters):
    img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    img = cv2.resize(img, (100,100))

    responses = []
    for kern in filters:
        fimg = cv2.filter2D(img, cv2.CV_8UC3, kern)
        responses.append(fimg)

    return responses

def get_local_energy (matrix):
    local_energy = 0.0
    for row in range (len(matrix)):
        for col in range(len(matrix[0])):
            val = int(matrix[row][col]) * int(matrix[row][col])
            local_energy = local_energy + val

    local_energy = local_energy / 650250000
    return EPS if local_energy == 0 else local_energy

def get_mean_amplitude (matrix):
    mean_amp = 0.0

    for row in range (len(matrix)):
        for col in range(len(matrix[0])):
            val = abs(int(matrix[row][col]))
            mean_amp = mean_amp + val

    mean_amp = mean_amp / 2550000
    return EPS if mean_amp == 0 else mean_amp

def load_images_from_folder (folder):
    images = []
    fileNames = []
    for path, subdirs, files in os.walk(folder):
        for filename in files:
            age = get_age(filename)
            img = cv2.imread(os.path.join(path,filename))
            if img is not None:
                images.append((img, age))
                fileNames.append(filename)
    return images, fileNames

def get_image_feature_vector(image, filename, filters):
    response_matrices = process(image[0], filters)

    local_energy_results = []
    mean_amplitude_results = []

    for matrix in response_matrices:
        local_energy = format(get_local_energy(matrix), '.2f')
        mean_amplitude = format(get_mean_amplitude(matrix), '.2f')
        local_energy_results.append (local_energy)
        mean_amplitude_results.append(mean_amplitude)
        

    feature_set = [filename] + local_energy_results + mean_amplitude_results + [image[1]]
    return feature_set

def get_all_image_feature_vectors(images, fileNames):
    filters = build_filters()
    feature_sets = []
    for image, f in zip(images, fileNames):
        feature_set = get_image_feature_vector(image, f, filters)
        feature_sets.append (feature_set)

    return feature_sets

def writeToCSVFile(fileName, dir):
        # Load images
        images, fileNames = load_images_from_folder (dir)

        # Get feature vectors of images
        feature_vectors = get_all_image_feature_vectors (images, fileNames)

        with open(fileName, 'a') as csv_file:
           writer = csv.writer(csv_file, dialect='excel')
           csv.field_size_limit(100000) 
           #writer = csv.writer(csv_file)
           fileEmpty = os.stat(os.path.realpath(fileName)).st_size == 0
           if fileEmpty:
            writer.writerow(["Image", "feature1", "feature2", "feature3", "feature4", "feature5", "feature6", "feature7", "feature8",\
                "feature9", "feature10", "feature11","feature12", "feature13", "feature14", "feature15", "feature16","feature17", \
                "feature18", "feature19", "feature20", "feature21", "feature22", "feature23", "feature24", "age_class"])
            for feature in feature_vectors:
                writer.writerow(feature)
           else:
            for feature in feature_vectors:
                writer.writerow(feature)


def get_age(image):
    if not image.startswith('.'):
        num_values = image.split('_')
        year_of_birth = int(num_values[1][0:4])
        year_of_image = int(num_values[2][0:4])
        age = int(year_of_image - year_of_birth)
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
        return age_class


if __name__ == '__main__':
    writeToCSVFile("gabor_feature_set_train.csv", "wiki_train")
    writeToCSVFile("gabor_feature_set_test.csv", "wiki_test")
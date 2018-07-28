# USAGE
#python feature_extract_for_image_new.py --shape-predictor shape_predictor_68_face_landmarks.dat --image /Users/jishavarun/Documents/Python/CS256/project/Age_Classification/facial-landmarks/wiki_images/311200_1943-09-21_2013.jpg
#python feature_extract_for_image_new.py --shape-predictor shape_predictor_68_face_landmarks.dat --image /Users/jishavarun/Documents/Python/headshot-3-tst2/headshots-39.jpeg

# import the necessary packages
from imutils import face_utils
from skimage.feature import local_binary_pattern
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn import neighbors, tree
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, precision_score, recall_score
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
import sys


PRED_nb = []
PRED_svc = []
PRED_dt = []
PRED_rf = []
PRED_knn_uniform = []
PRED_knn_distance = []
PRED_feature1 = []
PRED_feature2 = []
PRED_feature3 = []
PRED_LBP = []
PRED_gabor = []

import cgi
class UI():
	def print_html(self, img_name, PRED_svc, PRED_dt, PRED_rf, PRED_nb, PRED_knn_distance, PRED_knn_uniform):
	    try:
	        
	        #html_output = "<img src=homepage.jpg height=300 width=1500>"
	        
	        #PRED_svc = [5,2,1,6,1]
	        #PRED_dt = [1,2,5,8,4]
	        #PRED_rf = [7,8,1,2,8]
	        #PRED_nb = [9,1,0,2,9]
	        #PRED_knn_distance = [1,7,3,5,0]
	        #PRED_knn_uniform = [8,3,2,1,10]
	        #img_name = "savedimg.jpg"

	        html_output = """<br><h2 align=center style=font-family:courier;color:#3498DB;font-size:250%;>AGE CLASSIFICATION OF FACIAL IMAGES</h2><hr>"""
	        
	        html_output_img = """<h4 style=font-family:courier;color:#D35400;font-size:100%;> The input image is: </h4>"""
	        
	        img_input = "<img src=" +img_name+" width=200 height=200 align=middle>" 
	        
	        html_text = """<br><br><h4 style=font-family:courier;color:#BA4A00;font-size:100%;> The predicted age group for the given image is as follows:"""
	        html_output1 = """<body>
	                <br><br>
	                <table align=center border=1px>
	                <tr>
	                <th>Feature Sets</th>
	                <th>NB Classifier</th>
	                <th>Random Forest</th>
	                <th>KNN (Uniform classifier) (k=9)</th>
	                <th>KNN (Distance classifier) (k=9)</th>
	                <th>Decision Tree Classifier</th>
	                <th>SVM</th>
	             </tr>
	             <tr align=center>
	               <td> Local Binary Pattern </td>
	               <td>""" +str(PRED_nb[0])+""" </td>
	               <td>""" +str(PRED_rf[0])+""" </td>
	               <td>""" +str(PRED_knn_uniform[0])+""" </td>
	               <td>""" +str(PRED_knn_distance[0])+""" </td>
	               <td>""" +str(PRED_dt[0])+""" </td>
	               <td>""" +str(PRED_svc[0])+""" </td>
	             </tr>
	  
	             <tr align=center>
	               <td> Feature Set 3</td>
	               <td>""" +str(PRED_nb[1])+""" </td>
	               <td>""" +str(PRED_rf[1])+""" </td>
	               <td>""" +str(PRED_knn_uniform[1])+""" </td>
	               <td>""" +str(PRED_knn_distance[1])+""" </td>
	               <td>""" +str(PRED_dt[1])+""" </td>
	               <td>""" +str(PRED_svc[1])+""" </td>
	             </tr>
	  
	             <tr align=center>
	               <td> Feature Set 1</td>
	               <td>""" +str(PRED_nb[2])+""" </td>
	               <td>""" +str(PRED_rf[2])+""" </td>
	               <td>""" +str(PRED_knn_uniform[2])+""" </td>
	               <td>""" +str(PRED_knn_distance[2])+""" </td>
	               <td>""" +str(PRED_dt[2])+""" </td>
	               <td>""" +str(PRED_svc[2])+""" </td>
	             </tr>
	  
	             <tr align=center>
	               <td> Feature Set 2</td>
	               <td>""" +str(PRED_nb[3])+""" </td>
	               <td>""" +str(PRED_rf[3])+""" </td>
	               <td>""" +str(PRED_knn_uniform[3])+""" </td>
	               <td>""" +str(PRED_knn_distance[3])+""" </td>
	               <td>""" +str(PRED_dt[3])+""" </td> 
	               <td>""" +str(PRED_svc[3])+""" </td>
	             </tr>
	  
	             <tr align=center>
	               <td> Gabbor filters</td>
	               <td>""" +str(PRED_nb[4])+""" </td>
	               <td>""" +str(PRED_rf[4])+""" </td>
	               <td>""" +str(PRED_knn_uniform[4])+""" </td>
	               <td>""" +str(PRED_knn_distance[4])+""" </td>
	               <td>""" +str(PRED_dt[4])+""" </td>
	               <td>""" +str(PRED_svc[4])+""" </td>
	             </tr>
	             </table>
	        </body>"""
	        
	       
	        file_html = open("FinalResults.html","w")
	        file_html.write(str(html_output+html_output_img+img_input+html_text+html_output1))
	        file_html.close()        
	        
	        
	    except NameError, e:
	        #print("Name error")
	        print e
	    
	    except AttributeError:
	            print("Attribute error occured!!")
	    except:
	            print("Error occured!!") 


class featureSets():


	def calculate_distance(self, coord_1, coord_2):
		c1 = np.ndarray.tolist(coord_1)
		c2 = np.ndarray.tolist(coord_2)
		x1 = float(c1[0])
		y1 = float(c1[1])
		x2 = float(c2[0])
		y2 = float(c2[1])
		distance = math.sqrt(math.pow((x1 - x2), 2) + math.pow((y1 - y2), 2))
		return distance

	def calculate_mid(self, coord_1, coord_2):
		c1 = np.ndarray.tolist(coord_1)
		c2 = np.ndarray.tolist(coord_2)
		x1 = float(c1[0])
		y1 = float(c1[1])
		x2 = float(c2[0])
		y2 = float(c2[1])
		x = (x1 + x2)/2
		y = (y1+y2)/2
		return np.array([x,y])

	def feature_set1(self, f, shape_cropped):
		lip_width = self.calculate_distance(shape_cropped[48], shape_cropped[54])
		eye1_width = self.calculate_distance(shape_cropped[36], shape_cropped[39])
		eye2_width = self.calculate_distance(shape_cropped[42], shape_cropped[45])
		nose_length = self.calculate_distance(shape_cropped[27], shape_cropped[30])
		nose_width = self.calculate_distance(shape_cropped[31], shape_cropped[35])
		eyebrow1_length = self.calculate_distance(shape_cropped[17], shape_cropped[18]) + self.calculate_distance(
			shape_cropped[18], shape_cropped[19]) + \
		self.calculate_distance(shape_cropped[19], shape_cropped[20]) + self.calculate_distance(
			shape_cropped[20], shape_cropped[21])
		eyebrow2_length = self.calculate_distance(shape_cropped[22], shape_cropped[23]) + self.calculate_distance(
			shape_cropped[23], shape_cropped[24]) + \
		self.calculate_distance(shape_cropped[24], shape_cropped[25]) + self.calculate_distance(
			shape_cropped[25], shape_cropped[26])

		jawline_width = self.calculate_distance(shape_cropped[1], shape_cropped[17])

		feature["feture_set1_train.csv"] = [f,format(lip_width, '.2f'), format(eye1_width, '.2f'), format(eye2_width, '.2f'), format(nose_length, '.2f'), format(nose_width, '.2f'), format(eyebrow1_length, '.2f'), format(eyebrow2_length, '.2f'), format(jawline_width, '.2f')]

	def feature_set2(self, f, shape_cropped):
		facial_index = (self.calculate_distance(self.calculate_mid(shape_cropped[21],shape_cropped[22]),shape_cropped[8]))/self.calculate_distance(shape_cropped[1], shape_cropped[15])

		mandibular_index = (self.calculate_distance(self.calculate_mid(shape_cropped[62], shape_cropped[66]), shape_cropped[8]))/self.calculate_distance(shape_cropped[5], shape_cropped[11])

		intercanthal_index = (self.calculate_distance(shape_cropped[39], shape_cropped[42]))/self.calculate_distance(shape_cropped[36], shape_cropped[45])

		temp = (self.calculate_distance(shape_cropped[36], shape_cropped[39]) + self.calculate_distance(shape_cropped[42], shape_cropped[45]))/2
		orbital_width_index = temp/(self.calculate_distance(shape_cropped[39], shape_cropped[42]))

		temp = (self.calculate_distance(self.calculate_mid(shape_cropped[37], shape_cropped[38]), self.calculate_mid(shape_cropped[41], shape_cropped[40]))+ \
		 self.calculate_distance(self.calculate_mid(shape_cropped[43], shape_cropped[44]), self.calculate_mid(shape_cropped[47], shape_cropped[46])))/2

		eye_fissure_index = (temp/(self.calculate_distance(shape_cropped[36], shape_cropped[39]) + self.calculate_distance(shape_cropped[42], shape_cropped[45])))/2

		nasal_index = (self.calculate_distance(shape_cropped[31], shape_cropped[35]))/(self.calculate_distance(self.calculate_mid(shape_cropped[21], shape_cropped[22]), shape_cropped[33]))

		vermilion_height_index = (self.calculate_distance(shape_cropped[51], self.calculate_mid(shape_cropped[62], shape_cropped[66])))/(self.calculate_distance(self.calculate_mid(shape_cropped[62], shape_cropped[66]),shape_cropped[57]))

		mouth_face_width_index = (self.calculate_distance(shape_cropped[48], shape_cropped[54]))/(self.calculate_distance(shape_cropped[1], shape_cropped[15]))

		feature["feture_set2_train.csv"] = [f,format(facial_index, '.2f'), format(mandibular_index, '.2f'), format(intercanthal_index, '.2f'), format(orbital_width_index, '.2f'), format(eye_fissure_index, '.2f'), format(nasal_index, '.2f'), format(vermilion_height_index, '.2f'), format(mouth_face_width_index, '.2f')]

	def feature_set3(self, f, shape_cropped):

		left_eye_right_eye = (self.calculate_distance(shape_cropped[37], shape_cropped[44]) + self.calculate_distance(shape_cropped[38], shape_cropped[43]))/2

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

		feature["feture_set3_train.csv"] = [f, format(ratio1, '.2f'), format(ratio2, '.2f'), format(ratio3, '.2f'), format(ratio4, '.2f'), format(ratio5, '.2f'), format(ratio6, '.2f')]



	def extract_facial_landmarks(self, f):
				ap = argparse.ArgumentParser()
				ap.add_argument("-p", "--shape-predictor", required=True,
				                help="path to facial landmark predictor")
				ap.add_argument("-i", "--image", required=True,
				help="path to input image")
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
				    	shape_cropped = predictor(gray_resized, rect_cropped)
				    	shape_cropped = face_utils.shape_to_np(shape_cropped)
				    self.feature_set1(f,shape_cropped)
				    self.feature_set2(f,shape_cropped)
				    self.feature_set3(f,shape_cropped)

class gaborFilters():

	def build_filters(self):
		filters = []
		ksize = 31
		for theta in np.arange(0, np.pi, np.pi/4):
			for lamb in np.arange(np.pi/4, np.pi, np.pi/4):
				kern = cv2.getGaborKernel((ksize, ksize), 4.0, theta, lamb, 0.5, 0, ktype=cv2.CV_32F)
				kern /= 1.5*kern.sum()
				filters.append(kern)

		return filters
	 
	def process(self, img, filters):
		img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
		img = cv2.resize(img, (100,100))

		responses = []
		for kern in filters:
			fimg = cv2.filter2D(img, cv2.CV_8UC3, kern)
			responses.append(fimg)
		return responses

	def get_local_energy (self, matrix):
		local_energy = 0.0
		for row in range (len(matrix)):
			for col in range(len(matrix[0])):
				val = int(matrix[row][col]) * int(matrix[row][col])
				local_energy = local_energy + val

		local_energy = local_energy / 650250000
		return EPS if local_energy == 0 else local_energy

	def get_mean_amplitude (self, matrix):
		mean_amp = 0.0

		for row in range (len(matrix)):
			for col in range(len(matrix[0])):
				val = abs(int(matrix[row][col]))
				mean_amp = mean_amp + val

		mean_amp = mean_amp / 2550000
		return EPS if mean_amp == 0 else mean_amp

	def get_image_feature_vector(self, path):
			image = cv2.imread(sys.argv[-1])
			filters = self.build_filters()
			response_matrices = self.process(image, filters)
			local_energy_results = []
			mean_amplitude_results = []
			for matrix in response_matrices:
				local_energy = format(self.get_local_energy(matrix), '.2f')
				mean_amplitude = format(self.get_mean_amplitude(matrix), '.2f')
				local_energy_results.append (local_energy)
				mean_amplitude_results.append(mean_amplitude)


			feature_set = [path] + local_energy_results + mean_amplitude_results
			feature["gabor_feature_set_train.csv"]=feature_set
			return feature_set


class LBP():
		def __init__(self):
			self.num_of_points = 24
			self.radius = 8

		def create_histogram(self, f, eps = 1e-7):
			image = cv2.imread(f)
			gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
			extracted_features = local_binary_pattern(gray, self.num_of_points, self.radius, method = "uniform" )
			print "Features extracted"
			hist, unused_arg = np.histogram(extracted_features.ravel(), bins = np.arange(0, self.num_of_points + 3), range = (0, self.num_of_points + 2))
			hist = hist.astype("float")
			hist = hist/(hist.sum() + eps)
			lbp_list = []
			lbp_list.append(f)
			lbp_list = lbp_list + hist.tolist()
			feature["LBP_features_train.csv"] = lbp_list
			return hist

class NBClassifier:
	def __init__(self, train_data, train_labels, test_data):
		self.train_data = train_data
		self.train_labels = train_labels
		self.test_data = test_data

	def implement_classifier(self):
		# Initialize our classifier
		gnb = GaussianNB()

		# Train our classifier
		model = gnb.fit(self.train_data, self.train_labels)
		# Make predictions
		preds = gnb.predict(self.test_data)
		PRED_nb.append(preds[0])
		print "Naive Bayes: "
		print preds

class RFClassifier:
	def __init__(self, train_data, train_labels, test_data):
		self.train_data = train_data
		self.train_labels = train_labels
		self.test_data = test_data

	def classify(self):
		# Initialize our classifier
		clf = RandomForestClassifier(max_depth=2, random_state=0)

		# Train our classifier
		clf.fit(self.train_data, self.train_labels)

		# Make predictions
		preds = clf.predict(self.test_data)
		PRED_rf.append(preds[0])
		print "Random Forest Classifier: "
		print preds

class kNearestNeighbors:
	def __init__(self, train_data, train_labels, test_data):
		self.train_data = train_data
		self.train_labels = train_labels
		self.test_data = test_data

	def classify(self):
		print "knn Classifier: "
		for n_neighbors in range(1, 10, 2):
			print "No. of neighbors = " + str(n_neighbors) + ": "
			for weights in ['uniform', 'distance']:
				clf = neighbors.KNeighborsClassifier(n_neighbors, weights=weights)
				clf.fit(self.train_data, self.train_labels)
				preds = clf.predict(self.test_data)
				if n_neighbors == 9:
					if weights == 'uniform':
						PRED_knn_uniform.append(preds[0])
					elif weights == 'distance':
						PRED_knn_distance.append(preds[0])
				print "Weight type = " + str(weights) + ": "
				print(preds)


class DecisionTreeClassifier:
	def __init__(self, train_data, train_labels, test_data):
		self.train_data = train_data
		self.train_labels = train_labels
		self.test_data = test_data

	def classify(self):
		# Initialize our classifier
		clf = tree.DecisionTreeClassifier()

		# Train our classifier
		clf.fit(self.train_data, self.train_labels)

		# Make predictions
		preds = clf.predict(self.test_data)

		print "Decision Tree Classifier: "
		print preds
		PRED_dt.append(preds[0])

class SVMClassifier:
	def __init__(self, train_data, train_labels, test_data):
		self.train_data = train_data
		self.train_labels = train_labels
		self.test_data = test_data

	def classify(self):
		train_size = int(len(self.train_data))
		test_size = int(len(self.test_data))

		trainvec = self.train_data[:train_size, 0::]
		train_targets = self.train_labels[:train_size]
		testvec = self.test_data[:test_size, 0::]

		new_trainvec = trainvec
		new_train_targets = train_targets

		classifier = LinearSVC(C=0.01, penalty='l2', dual=True)
		classifier.fit(new_trainvec, new_train_targets)

		print "Support Vector Classifier"
		predicted = classifier.predict(testvec)
		PRED_svc.append(predicted[0])


if __name__ == '__main__':
	feature = {}

	# Extract image features for feature set 1, 2 & 3
	feature_set = featureSets()
	feature_set.extract_facial_landmarks(sys.argv[-1])

	# Extract gabor features
	gabor_filter = gaborFilters()
	EPS = 0.00000000000000001
	# Get feature vectors of images
	feature_vectors = gabor_filter.get_image_feature_vector(sys.argv[-1])

	# Extract LBP features
	lbp_features = LBP()
	lbp_feature_vector = lbp_features.create_histogram(sys.argv[-1])
	print feature

	train_file_names = ['feature_set1_train.csv', 'feature_set2_train.csv', 'feature_set3_train.csv',
						'gabor_feature_set_train.csv', 'lbp_features.csv']
	for key, value in feature.items():
		print "\n"
		print key + "\n\n"
		train_data = []
		train_labels = []

		

		with open(key) as f:
			csvreader = csv.reader(f)
			csvreader.next()
			for row in csvreader:
				if int(row[-1]) <= 4:
					train_data.append(row[1:len(row) - 1])
					train_labels.append('young')

				else:
					train_data.append(row[1:len(row) - 1])
					train_labels.append('old')
		test_data = feature[key]
		train_data = np.array(train_data).astype(np.float)
		test_data = value[1:]
		for ind in range(0, len(test_data)):
			test_data[ind] = float(test_data[ind])
		#print test_data
		test_data = np.array([test_data]).astype(np.float)

		# Implement the NB classifer
		naive_bayes_object = NBClassifier(train_data, train_labels, test_data)
		#print "Naive Bayes Classifier: "
		accuracy_nb = naive_bayes_object.implement_classifier()

		# Implement the Random Forest Classifier
		random_forest_object = RFClassifier(train_data, train_labels, test_data)
		#print "Random Forest Classifier: "
		accuracy_rf = random_forest_object.classify()

		# Implement the kNN classifier
		knn_object = kNearestNeighbors(train_data, train_labels, test_data)
		#print "k Nearest Neighbors Classifier: "
		accuracy_knn = knn_object.classify()

		# Implement the Decision Tree Classifier
		decision_tree_object = DecisionTreeClassifier(train_data, train_labels, test_data)
		#print "Decision Tree Classifier: "
		accuracy_dt = decision_tree_object.classify()

		# Implement the SVM Classifier
		svm_object = SVMClassifier(train_data, train_labels, test_data)
		#print "SVM Classifier: "
		accuracy_svm = svm_object.classify()
	print "\n"
	print "Results are written to FinalResults.html"
	print "\n"
	ui = UI()
	ui.print_html(sys.argv[-1], PRED_svc, PRED_dt, PRED_rf, PRED_nb, PRED_knn_distance, PRED_knn_uniform)






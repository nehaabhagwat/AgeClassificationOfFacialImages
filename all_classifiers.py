from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn import neighbors, tree
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score
import numpy as np
import csv

class NBClassifier:
    def __init__(self, train_data, train_labels, test_data, test_labels):
        self.train_data = train_data
        self.train_labels = train_labels
        self.test_data = test_data
        self.test_labels = test_labels

    def implement_classifier(self):
        # Initialize our classifier
        gnb = GaussianNB()

        # Train our classifier
        model = gnb.fit(self.train_data, self.train_labels)

        # Make predictions
        preds = gnb.predict(self.test_data)

        results_list = []
        correct_count = 0
        for label_index in range(0, len(test_labels)):
            if self.test_labels[label_index] == preds[label_index]:
                correct_count += 1
            results_list.append([filenames[label_index], self.test_labels[label_index], preds[label_index]])

        """for result in results_list:
            print "Filename: ", result[0]
            print "Actual Label: ", result[1]
            print "Predicted Label: ", result[1]
            print "*************************************"""""

        # print "Correct count: ", correct_count
        # print "Total test data: ", len(preds)
        # Evaluate accuracy
        accuracy_nb = (accuracy_score(self.test_labels, preds))
        print "Accuracy: ", accuracy_nb
        return accuracy_nb

class RFClassifier:
    def __init__(self, train_data, train_labels, test_data, test_labels):
        self.train_data = train_data
        self.train_labels = train_labels
        self.test_data = test_data
        self.test_labels = test_labels

    def classify(self):
        # Initialize our classifier
        clf = RandomForestClassifier(max_depth=2, random_state=0)

        # Train our classifier
        clf.fit(train_data, train_labels)

        # Make predictions
        preds = clf.predict(self.test_data)

        results_list = []
        correct_count = 0
        for label_index in range(0, len(test_labels)):
            if self.test_labels[label_index] == preds[label_index]:
                correct_count += 1
            results_list.append([filenames[label_index], self.test_labels[label_index], preds[label_index]])

        """for result in results_list:
            print "Filename: ", result[0]
            print "Actual Label: ", result[1]
            print "Predicted Label: ", result[1]
            print "*************************************"""""

        # print "Correct count: ", correct_count
        # print "Total test data: ", len(preds)
        # Evaluate accuracy
        accuracy_rf = (accuracy_score(self.test_labels, preds))
        print "Accuracy: ", accuracy_rf
        return accuracy_rf

class kNearestNeighbors:
    def __init__(self, train_data, train_labels, test_data, test_labels):
        self.train_data = train_data
        self.train_labels = train_labels
        self.test_data = test_data
        self.test_labels = test_labels

    def classify(self):
        for n_neighbors in range(1, 10, 2):
            for weights in ['uniform', 'distance']:
                clf = neighbors.KNeighborsClassifier(n_neighbors, weights=weights)
                clf.fit(self.train_data, self.train_labels)

                # Make predictions
                preds = clf.predict(self.test_data)
                # print(preds)

                results_list = []
                correct_count = 0
                for label_index in range(0, len(self.test_labels)):
                    if self.test_labels[label_index] == preds[label_index]:
                        correct_count += 1
                    results_list.append([filenames[label_index], self.test_labels[label_index], preds[label_index]])

                """for result in results_list:
                    print "Filename: ", result[0]
                    print "Actual Label: ", result[1]
                    print "Predicted Label: ", result[1]
                    print "*************************************"
                """
                """print "Correct count: ", correct_count
                print "Total test data: ", len(preds)
                """
                print "Number of neighbours: ", n_neighbors
                print "Weight type: ", weights
                # Evaluate accuracy
                print "Accuracy: ", accuracy_score(self.test_labels, preds)
                print "*************************************"

class DecisionTreeClassifier:
    def __init__(self, train_data, train_labels, test_data, test_labels):
        self.train_data = train_data
        self.train_labels = train_labels
        self.test_data = test_data
        self.test_labels = test_labels

    def classify(self):
        # Initialize our classifier
        clf = tree.DecisionTreeClassifier()

        # Train our classifier
        clf.fit(train_data, train_labels)

        # Make predictions
        preds = clf.predict(test_data)

        results_list = []
        correct_count = 0
        for label_index in range(0, len(test_labels)):
            if self.test_labels[label_index] == preds[label_index]:
                correct_count += 1
            results_list.append([filenames[label_index], self.test_labels[label_index], preds[label_index]])

        """for result in results_list:
            print "Filename: ", result[0]
            print "Actual Label: ", result[1]
            print "Predicted Label: ", result[1]
            print "*************************************"""""

        # print "Correct count: ", correct_count
        # print "Total test data: ", len(preds)
        # Evaluate accuracy
        accuracy_dt = (accuracy_score(self.test_labels, preds))
        print "Accuracy: ", accuracy_dt
        return accuracy_dt

class SVMClassifier:
    def __init__(self, train_data, train_labels, test_data, test_labels):
        self.train_data = train_data
        self.train_labels = train_labels
        self.test_data = test_data
        self.test_labels = test_labels

    def classify(self):
        train_size = int(len(train_data))
        test_size = int(len(test_data))

        trainvec = train_data[:train_size, 0::]
        train_targets = train_labels[:train_size]
        testvec = test_data[:test_size, 0::]
        test_targets = test_labels[:test_size]

        new_trainvec = trainvec
        new_train_targets = train_targets

        print testvec

        classifier = LinearSVC(C=0.01, penalty='l2', dual=True)
        classifier.fit(new_trainvec, new_train_targets)

        print "Accuracy of SVM Classifier for feature set-1"
        predicted = classifier.predict(testvec)

        results_list = []
        correct_count = 0
        for label_index in range(0, len(test_labels)):
            if test_labels[label_index] == predicted[label_index]:
                correct_count += 1
            results_list.append([filenames[label_index], test_labels[label_index], predicted[label_index]])

        """for result in results_list:
                    print "Filename: ", result[0]
                    print "Actual Label: ", result[1]
                    print "Predicted Label: ", result[1]
                    print "*************************************"""""

        # print "Correct count: ", correct_count
        # print "Total test data: ", len(preds)
        # Evaluate accuracy
        accuracy_svc = (accuracy_score(self.test_labels, predicted))
        print "Accuracy: ", accuracy_svc
        return accuracy_svc


if __name__ == "__main__":
    train_file_names = ['feature_set1_train.csv', 'feature_set2_train.csv', 'gabor_feature_set_train.csv', 'lbp_features.csv']
    test_file_names = ['feature_set1_test.csv', 'feature_set2_test.csv', 'gabor_feature_set_test.csv', 'lbp_test.csv']

    for file_count in range(0, len(train_file_names)):
        train_data = []
        train_labels = []
        test_data = []
        test_labels = []

        print "Prediction for file " + test_file_names[file_count] + ": "

        with open(train_file_names[file_count]) as f:
            csvreader = csv.reader(f)
            csvreader.next()
            for row in csvreader:
                if int(row[-1]) <= 4:
                    train_data.append(row[1:len(row)-1])
                    train_labels.append('young')

                else:
                    train_data.append(row[1:len(row)-1])
                    train_labels.append('old')

        filenames = []
        with open(test_file_names[file_count]) as f:
            csvreader = csv.reader(f)
            csvreader.next()

            for row in csvreader:
                filenames.append(row[0])
                if int(row[-1]) <= 4:
                    test_data.append(row[1:len(row)-1])
                    test_labels.append('young')

                else:
                    test_data.append(row[1:len(row)-1])
                    test_labels.append('old')


        train_data = np.array(train_data).astype(np.float)
        test_data = np.array(test_data).astype(np.float)

        # Implement the NB classifer
        naive_bayes_object = NBClassifier(train_data, train_labels, test_data, test_labels)
        print "Naive Bayes Classifier: "
        accuracy_nb = naive_bayes_object.implement_classifier()

        # Implement the Random Forest Classifier
        random_forest_object = RFClassifier(train_data, train_labels, test_data, test_labels)
        print "Random Forest Classifier: "
        accuracy_rf = random_forest_object.classify()

        # Implement the kNN classifier
        knn_object = kNearestNeighbors(train_data, train_labels, test_data, test_labels)
        print "k Nearest Neighbors Classifier: "
        accuracy_knn = knn_object.classify()

        # Implement the Decision Tree Classifier
        decision_tree_object = DecisionTreeClassifier(train_data, train_labels, test_data, test_labels)
        print "Decision Tree Classifier: "
        accuracy_dt = decision_tree_object.classify()

        # Implement the SVM Classifier
        svm_object = SVMClassifier(train_data, train_labels, test_data, test_labels)
        print "SVM Classifier: "
        accuracy_svm = svm_object.classify()


        print "_____________________________________________________"

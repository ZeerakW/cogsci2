import os
import random
import numpy as np
from collections import defaultdict
from PIL import Image
from glob import glob
from scipy import ndimage
from sklearn.svm import SVC
from skimage.feature import canny
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.cross_validation import StratifiedKFold
from sklearn.grid_search import GridSearchCV

"""
DONE
0. Resize all images to same size
1. Extract canny features from training set (either human OR drawn) 
2. Add labels to data
3. Add cross validation
4. Shuffle data
5. Feed features to classifier(s) - Classifiers: KNearest, SVM, LogisticRegression, k_means
6. Predict on the test set
7. Implement GridSearchCV in get_prediction

TODO
8. Predict drawings on human trained classifier and vice versa
"""

def get_features(img, sigma):
    """
    :param img: img
    :returns: feature list
    """

    img = ndimage.gaussian_filter(img, 4)
    can = canny(img, sigma)
    return can 

def get_prediction(clf, params, trainX, trainY, testX, testY):
    trainX = trainX.reshape(len(trainX), -1)
    testX = testX.reshape(len(testX), -1)
    # accuracies = {}
    # fold_acc = []
    # for train, test in StratifiedKFold(trainY, n_folds = 4):
    #     clf.fit(trainX[train], trainY[train])
    #     pred = clf.predict(trainX[test])
    #     fold_acc.append(accuracy_score(trainY[test], pred))
    
    mod_clf = GridSearchCV(clf, params)
    mod_clf.fit(trainX, trainY)
    pred = mod_clf.predict(testX)
    acc = accuracy_score(testY, pred)

    # accuracies['folds'] = fold_acc
    # accuracies['testSet'] = accuracy_score(testY, pred)

    return acc, mod_clf

def read_Data():
    """
    Reads in data and adds labels.
    :return: Shuffled data
    """
    # Get filenames
    thumbs_h = glob(os.getcwd() + '/data/thumbs_up/human/*')
    thumbs_d = glob(os.getcwd() + '/data/thumbs_up/Drawn/*')
    peace_h  = glob(os.getcwd() + '/data/peace/human/*')
    peace_d  = glob(os.getcwd() + '/data/peace/Drawn/*')

    """
    Data structure:
    print humans accesses array of all images
    print humans[i] accesses array of single image and label
    print humans[i][0] accesses image
    print humans[i][1] accesses label
    """

    # Read files in and save them in np.array
    humans_t = np.array([[np.array(Image.open(thumbs).convert('L'), 'f'), 1] for thumbs in thumbs_h])
    humans_p = np.array([[np.array(Image.open(peace).convert('L'), 'f'), 0] for peace in peace_h])
    drawn_t  = np.array([[np.array(Image.open(thumbs).convert('L'), 'f'), 1] for thumbs in thumbs_d])
    drawn_p  = np.array([[np.array(Image.open(peace).convert('L'), 'f'), 0] for peace in peace_d])
    
    # Stack all human and drawn images
    humans = np.vstack((humans_t, humans_p))
    drawings = np.vstack((drawn_t, drawn_p))
    
    # Shuffle data
    np.random.shuffle(humans)
    np.random.shuffle(drawings)

    return humans, drawings

def fit_pred(clf, trainX, testX, testY):
    pred = clf.predict(testX)
    return accuracy_score(testY, pred)

def main():
    # Read in data
    humans, drawings = read_Data()

    # Get features and labels 
    h_feats = []
    h_labels = []
    d_feats = []
    d_labels = []

    for i in range(len(humans)):
        h_feats.append(get_features(humans[i][0], sigma=3))
        h_labels.append(humans[i][1])
        d_feats.append(get_features(drawings[i][0], sigma=3))
        d_labels.append(drawings[i][1])

    # Split data
    dataSplit = len(humans)/5
    h_train_x = np.array(h_feats[dataSplit:])
    h_train_y = np.array(h_labels[dataSplit:])
    h_test_x  = np.array(h_feats[:dataSplit])
    h_test_y  = np.array(h_labels[:dataSplit])
    d_train_x = np.array(d_feats[dataSplit:])
    d_train_y = np.array(d_labels[dataSplit:])
    d_test_x  = np.array(d_feats[:dataSplit])
    d_test_y  = np.array(d_labels[:dataSplit])

    classifiers = {KNeighborsClassifier(): {'n_neighbors': [1,3,5,7,9]}, 
            SVC(): {'C': [0.001, 0.01, 0.1, 1, 10], 'kernel': ['rbf', 'linear']}, 
            LogisticRegression(): {'C': [0.001, 0.01, 0.1, 1, 10]}
            }
    
    pred           = defaultdict(lambda: defaultdict(list))
    human_clfs     = []
    drawings_clfs  = []
    human_draw_acc = defaultdict(list)
    draw_human_acc = defaultdict(list)
    # print d_test_x.shape
    # print h_test_x.shape

    for clf in classifiers.keys():
        # Gridsearch and get prediction on human-human test and human trained clf on test set
        pred['humans'][clf], h_clf = get_prediction(clf, classifiers[clf], 
                h_train_x, h_train_y, h_test_x, h_test_y)
        human_clfs.append(h_clf)
        human_draw_acc[clf] = fit_pred(h_clf, h_train_x, d_train_x, d_test_y)

        pred['drawings'][clf], d_clf = get_prediction(clf, 
                classifiers[clf], d_train_x, d_train_y, d_test_x, d_test_y)
        drawings_clfs.append(d_clf)
        # predict = d_clf.predict(h_test_x)
        # draw_human_acc[clf] = accuracy_score(h_test_y, predict)
    
    # print "Humans"
    # for clf in pred['humans'].keys():
    #     print "Classifier:\n%s\nScore on test set:\n%s\n" % (str(clf), str(pred['humans'][clf]))
    #     print "\nClassification on drawings\n%s", str(human_draw_acc[clf])
    #
    # print "Drawings"
    # for clf in classifiers.keys():
    #     print "Classifier:\n%s\nScore on test set:\n%s\n" % (str(clf), str(pred['drawings'][clf]))
    #     print "\nClassification on humans\n%s" % str(draw_human_acc[clf])
    


main()

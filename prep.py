import os
import random
import numpy as np
from PIL import Image
from glob import glob
from scipy import ndimage
from sklearn.svm import SVC
from skimage.feature import canny
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.cross_validation import StratifiedKFold

"""
DONE
0. Resize all images to same size
1. Extract canny features from training set (either human OR drawn) 
2. Add labels to data
3. Add cross validation


TODO
1. Shuffle data
2. Feed features to classifier(s) - Classifiers: KNearest, SVM, LogisticRegression, k_means
3. Predict on the test set
"""

def get_features(img, sigma):
    """
    :param img: img
    :returns: feature list
    """

    img = ndimage.gaussian_filter(img, 4)

    return canny(img, sigma)

def get_prediction(clf, trainX, trainY, testX, testY):
    
    accuracies = {}
    fold_acc = []
    for train, test in StratifiedKFold(labels, n_folds = 4):
        clf.fit(trainX[train])
        pred = clf.predict(trainX[test])
        fold_acc.append(accuracy_score(trainY[test], pred))
    
    pred = clf.predict(testX)

    accuracies['folds'] = fold_acc
    accuracies['testSet'] = accuracy_score(testY, pred)

    return accuracies

def main():
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

    classifiers = [KNeighborsClassifier(), SVC(), LogisticRegression()] 
    
    pred = {}
    for clf in classifiers:
        pred[clf] = get_prediction(clf, h_feats, h_labels)


main()

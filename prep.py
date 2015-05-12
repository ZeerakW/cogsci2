from skimage.feature import canny
from scipy import ndimage
from glob import glob
import numpy as np
import os
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.cluster import k_means
from PIL import Image

"""
DONE
1. Extract canny features from training set (either human OR drawn) 
2. Add labels to data

TODO
0. Resize all images to same size
1. 
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

def get_prediction(clf, train, test):
    # TODO Crossvalidate here
    clf.fit(train)
    return clf.predict(test)

def main():
    thumbs_h = glob(os.getcwd() + '/data/thumbs_up/human/*.jpg')
    thumbs_d = glob(os.getcwd() + '/data/thumbs_up/Drawn/*.jpg')
    peace_h  = glob(os.getcwd() + '/data/peace/human/*.jpg')
    peace_d  = glob(os.getcwd() + '/data/peace/Drawn/*.jpg')

    """
    Data structure:
    print humans accesses array of all images
    print humans[i] accesses array of single image and label
    print humans[i][0] accesses image
    print humans[i][1] accesses label
    """
    humans_t = np.array([[np.array(Image.open(thumbs).convert('L'), 'f'), 1] for thumbs in thumbs_h])
    humans_p = np.array([[np.array(Image.open(peace).convert('L'), 'f'), 1] for peace in peace_h])
    drawn_t  = np.array([[np.array(Image.open(thumbs).convert('L'), 'f'), 0] for thumbs in thumbs_d])
    drawn_p  = np.array([[np.array(Image.open(peace).convert('L'), 'f'), 0] for peace in peace_d])
    
    # Stack all human and drawn images
    humans = np.vstack((humans_t, humans_p))
    drawings = np.vstack((drawn_t, drawn_p))

    # Get features and labels 
    h_feats = []
    h_labels = []

    d_feats = []
    d_labels = []

    for i in range(40):
        h_feats.append(get_features(humans[i][0]), sigma=3)
        h_labels.append(humans[i][1])
        d_feats.append(get_features(drawings[i][0]), sigma=3)
        d_labels.append(drawings[i][1])

    # classifiers = [KNeighborsClassifier(), SVC(), LogisticRegression()] 
    
    # predictions = []
    # for clf in classifiers:
    #     get_prediction(clf, thumbs_h_feats, )

main()

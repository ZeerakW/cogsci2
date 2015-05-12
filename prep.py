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

TODO
0. Resize all images to same size
1. Add labels to data
2. Feed features to classifier(s) - Classifiers: KNearest, SVM, LogisticRegression, k_means
3. Predict on the test set
"""

def get_features(img, sigma):
    """
    :param img: img
    :returns: feature list
    """

    img = ndimage.gaussian_filter(img, 4)
    feats = canny(img, sigma)
    # TODO Add label

    return 

def get_prediction(clf, train, test):
    # TODO Crossvalidate here
    clf.fit(train)
    return clf.predict(test)

def main():
    thumbs_h = glob(os.getcwd() + '/data/thumbs_up/human/*.jpg')
    thumbs_d = glob(os.getcwd() + '/data/thumbs_up/Drawn/*.jpg')
    peace_h  = glob(os.getcwd() + '/data/peace/human/*.jpg')
    peace_d  = glob(os.getcwd() + '/data/peace/Drawn/*.jpg')

    # TODO Read images
    humans_t = [[np.array(Image.open(thumbs).convert('L'), 'f'), 1] for thumbs in thumbs_h]
    humans_p = [[np.array(Image.open(peace).convert('L'), 'f'), 1] for peace in peace_h]
    drawn_t  = [[np.array(Image.open(thumbs).convert('L'), 'f'), 0] for thumbs in thumbs_d]
    drawn_p  = [[np.array(Image.open(peace).convert('L'), 'f'), 0] for peace in peace_d]

    humans = []   
    print humans_t.extend(humans_p)
    drawn = np.array(drawn_t.extend(drawn_p))

    print humans_t
    print np.array(humans).shape

    # thumbs_h_feats = []
    # thumbs_d_feats = []
    # peace_h_feats  = []
    # peace_d_feats  = []
    # for i in range(len(thumbs_h)):
    #     thumbs_h_feats.append(get_features(thumbs_h[i], 3))
    #     peace_h_feats.append(get_features(peace_h[i], 3))
    #     thumbs_d_feats.append(get_features(thumbs_d[i], 3))
    #     peace_d_feats.append(get_features(peace_d[i], 3))

    # drawn_feats = thumbs_d_feats.extend(peace_d_feats)
    # human_feats = thumbs_h_feats.extend(peace_h_feats)
        
    # classifiers = [KNeighborsClassifier(), SVC(), LogisticRegression()] 
    
    # predictions = []
    # for clf in classifiers:
    #     get_prediction(clf, thumbs_h_feats, )

main()

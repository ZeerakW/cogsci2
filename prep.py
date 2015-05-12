from skimage.features import canny
from scipy import ndimage

"""
DONE
1. Extract canny features from training set (either human OR drawn) 

TODO
0. Resize all images to same size
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
    clf.fit(train)
    clf.predict(test)

def main():
    # TODO Get canny features for test set
    # TODO Get canny features for training set

    

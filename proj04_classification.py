import numpy as np
from skimage.feature import hog
from sklearn.neighbors import NearestCentroid
import cv2
import os
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier


def readFiles(type):
    X = []
    if type == "palm":
        path = '/Users/eyangpc/PycharmProjects/StarterCode/palm/'

    elif type == "notpalm":
        path = '/Users/eyangpc/PycharmProjects/StarterCode/notpalm/'

    else:
        print('No such type!')

    for filename in os.listdir(path):
        X.append(cv2.imread(path + filename))

    return X


def hog_transform(mat, which):
    mat_hog = []
    if which == 'fd':
        for img in mat:
            fd, hog_image = hog(img, orientations=8, pixels_per_cell=(16, 16),
                                cells_per_block=(1, 1), visualize=True, multichannel=True)
            # img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            # img = img.resize((width, height), Image.ANTIALIAS)
            # img = cv2.resize(img, (width, height), interpolation=cv2.INTER_AREA)
            # Xlbp.append(local_binary_pattern(img, n_points, radius, METHOD))
            mat_hog.append(fd)

    elif which == 'hog':
        for img in mat:
            fd, hog_image = hog(img, orientations=8, pixels_per_cell=(16, 16),
                                cells_per_block=(1, 1), visualize=True, multichannel=True)
            mat_hog.append(hog_image)

    mat_hog = np.array(mat_hog)

    return mat_hog


# read images of palms and not palms
Xpalm = readFiles("palm")
Xnotpalm = readFiles("notpalm")

X = Xpalm + Xnotpalm
X = np.array(X)

y = []
for i in range(0, 111):
    y.append('Palm')

for i in range(0, 190):
    y.append('NotPalm')

# Xg = []
# for img in X:
#     img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
#     Xg.append(img)
#

X_train_orig, X_test_orig, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=2019)

# #############################################################################

X_train = hog_transform(X_train_orig, 'fd')
X_test = hog_transform(X_test_orig, 'fd')

# ########### Near Neighbors #################################################

print('--------- Near Neighbors ---------')

clfNear = KNeighborsClassifier(n_neighbors=8, weights='distance',
                               algorithm='auto', leaf_size=30, p=2)
clfNear.fit(X_train, y_train)
y_fit_near = clfNear.predict(X_test)

print(classification_report(y_test, y_fit_near))
print(confusion_matrix(y_test, y_fit_near))

for i in range(0, len(y_test)):
    if y_test[i] != y_fit_near[i]:
        print(i, 'miss matched')

# ########### Random Forest ##################################################

print('--------- Random Forest ---------')

clfRF = RandomForestClassifier(n_estimators=10, max_depth=None,
                               min_samples_split=2, random_state=210)
clfRF.fit(X_train, y_train)
y_fit_rf = clfRF.predict(X_test)

print(classification_report(y_test, y_fit_rf))
print(confusion_matrix(y_test, y_fit_rf))

# ########### Neural Network ##################################################

print('--------- Neural Network ---------')

clfNet = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5, 2), random_state=2100)
clfNet.fit(X_train, y_train)
y_fit_net = clfNet.predict(X_test)

print(classification_report(y_test, y_fit_net))
print(confusion_matrix(y_test, y_fit_net))

# ########## Support Vector Machine ###########################################

print('--------- linear SVM ---------')

clf = SVC(kernel='linear')
clf = clf.fit(X_train, y_train)
y_fit = clf.predict(X_test)

print(classification_report(y_test, y_fit))
print(confusion_matrix(y_test, y_fit))

print('--------- RBF SVM ---------')

clf = SVC(kernel='rbf', gamma='scale')
clf = clf.fit(X_train, y_train)
y_fit = clf.predict(X_test)

print(classification_report(y_test, y_fit))
print(confusion_matrix(y_test, y_fit))

for i in range(0, len(y_test)):
    if y_test[i] != y_fit[i]:
        print(i, 'miss matched')


# cv2.imwrite('/Users/eyangpc/PycharmProjects/StarterCode/mismatch/18.jpg', X_test_orig[18])
# cv2.imwrite('/Users/eyangpc/PycharmProjects/StarterCode/mismatch/32.jpg', X_test_orig[32])
# cv2.imwrite('/Users/eyangpc/PycharmProjects/StarterCode/mismatch/59.jpg', X_test_orig[59])
# cv2.imwrite('/Users/eyangpc/PycharmProjects/StarterCode/mismatch/61.jpg', X_test_orig[61])
# cv2.imwrite('/Users/eyangpc/PycharmProjects/StarterCode/mismatch/73.jpg', X_test_orig[73])

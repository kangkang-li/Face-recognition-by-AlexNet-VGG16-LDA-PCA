from skmultilearn.problem_transform import BinaryRelevance
# from __future__ import print_function
from time import time
import logging
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cross_validation import train_test_split
from sklearn.datasets import fetch_lfw_people
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.decomposition import RandomizedPCA
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier as kNN

from souceCode import dataRead_1


# Display progress logs on stdout
# logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')


###############################################################################
def accuCalculator(a,b):
  count=0
  for i in range(len(a)):
    if a[i]==b[i]:
      count+=1
  # return '%0.2f%%' % (count/len(a)*100)
  return count/len(a)*100


def faceRec(option,sizeTrainGroup,y_train,X_train, y_test,X_test):#option: PCA/LDA, classifier using knn & svm
  # Download the data, if not already on disk and load it as numpy arrays

  # lfw_people = fetch_lfw_people(min_faces_per_person=70, resize=0.4)

  # introspect the images arrays to find the shapes (for plotting)
  # n_samples, h, w = lfw_people.images.shape


  # for machine learning we use the 2 data directly (as relative pixel
  # positions info is ignored by this model)
  n_features = X_train.shape[1]

  # the label to predict is the id of the person
  # y = lfw_people.target
  # y=labelLib.reshape(len(labelLib),)
  # print(y.shape)
  # n_samples=len(y)
  h=w=32
  # target_names = lfw_people.target_names
  target_names=np.array([str(i) for i in range(0,39)])
  n_classes = 38

  # print("Total dataset size:")
  # print("n_samples: %d" % n_samples)
  # print("n_features: %d" % n_features)
  # print("n_classes: %d" % n_classes)


  ###############################################################################
  # Split into a training set and a test set using a stratified k fold

  # split into a training and testing set
  # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=n_samples-10*n_classes)
  # y_train=y_train.reshape(len(y_train),)
  # y_test=y_test.reshape(len(y_test),)
  # print(X_train)
  # print(X_test)
  # print(y_train)
  # print(y_test)
  # print(y_train.shape)
  # y_train=y_train.T
  # y_train=y_train.reshape(len(y_train),)
  # y_test=y_test.reshape(len(y_test),)
  ###############################################################################
  # Compute a PCA (eigenfaces) on the face dataset (treated as unlabeled
  # dataset): unsupervised feature extraction / dimensionality reduction
  n_components = 100

  # print("Extracting the top %d eigenfaces from %d faces" % (n_components, X_train.shape[0]))
  t0 = time()
  pca = PCA(n_components=n_components, whiten=False).fit(X_train)
  # print("done in %0.3fs" % (time() - t0))

  eigenfaces = pca.components_.reshape((n_components, h, w))
  # print(pca.components_.shape)
  # print(pca.explained_variance_ratio_)
  # print(pca.explained_variance_.shape)
  
  pca.explained_variance_=np.delete(pca.explained_variance_,[0,1,2,3,4],0)
  pca.components_=np.delete(pca.components_,[0,1,2,3,4],0)

  # print(pca.components_.shape)
  # print("Projecting the input data on the eigenfaces orthonormal basis")
  t0 = time()
  X_train_pca = pca.transform(X_train)
  X_test_pca = pca.transform(X_test)
  # print("done in %0.3fs" % (time() - t0))
  if (option==0):
    X_train_target=X_train_pca
    X_test_target=X_test_pca
  else:
    lda = LinearDiscriminantAnalysis()
    lda.fit(X_train_pca, y_train)
    # y_pred
    X_train_lda=lda.transform(X_train_pca)
    X_test_lda=lda.transform(X_test_pca)
    X_train_target=X_train_lda
    X_test_target=X_test_lda

  #----------------------SRC---------
  # y_train.reshape(380,1)
  # print(y_train.shape)
  yt=[[n] for n in y_train]
  SRC = BinaryRelevance(classifier = SVC(), require_dense = [False, True])
  SRC.fit(X_train_target, yt)
  resultMatrix= SRC.predict(X_test_target)
  y_pred=[int(str(item).split('\t')[-1]) for item in resultMatrix]
  # print(y_pred)
  # print(str(y_pred[0]).split('\t'))
  accu3=accuCalculator(y_pred,y_test)


# -----------------------knn------------
  knnClf = kNN(n_neighbors=1)
  knnClf.fit(X_train_target, y_train)
  y_pred = knnClf.predict(X_test_target)
  accu1=accuCalculator(y_pred,y_test)
  # print(accu)
  # print(classification_report(y_test, y_pred, target_names=target_names))

  ###############################################################################
  # Train a SVM classification model

  # print("Fitting the classifier to the training set")
  t0 = time()
  param_grid = {'C': [1e3, 5e3, 1e4, 5e4, 1e5],
                'gamma': [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.1], }
  clf = GridSearchCV(SVC(kernel='rbf', class_weight=None), param_grid)
  # clf=SVC()
  # print("!!!!!",X_train_pca.shape,X_test_pca.shape)
  clf = clf.fit(X_train_target, y_train)
  # print("done in %0.3fs" % (time() - t0))
  # print("Best estimator found by grid search:")
  # print(clf.best_estimator_)


  ###############################################################################
  # Quantitative evaluation of the model quality on the test set

  # print("Predicting people's names on the test set")
  t0 = time()
  y_pred = clf.predict(X_test_target)



  # y_test=[str(i) for i in y_test]
  # y_pred=[str(i) for i in y_pred]
  # print(y_test)
  # print("done in %0.3fs" % (time() - t0))
  # print([str(i) for i in y_test.T])
  # for item in y_pred:
    # print(item)
  # print(y_test.T.shape,y_pred.T.shape)
  accu2=accuCalculator(y_pred,y_test)
    # print(accu)
    # print(classification_report(y_test, y_pred, target_names=target_names))
    # print(confusion_matrix(y_test, y_pred, labels=range(n_classes)))

  # prediction_titles = [title(y_pred, y_test, target_names, i) for i in range(y_pred.shape[0])]

  # plot_gallery(X_test, prediction_titles, h, w)

  # # # plot the gallery of the most significative eigenfaces

  # eigenface_titles = ["eigenface %d" % i for i in range(eigenfaces.shape[0])]
  # plot_gallery(eigenfaces, eigenface_titles, h, w)

  # plt.show()
  # print(accu)



  return accu1,accu2,accu3
###############################################################################
# Qualitative evaluation of the predictions using matplotlib

def plot_gallery(images, titles, h, w, n_row=3, n_col=4):
    """Helper function to plot a gallery of portraits"""
    plt.figure(figsize=(1.8 * n_col, 2.4 * n_row))
    plt.subplots_adjust(bottom=0, left=.01, right=.99, top=.90, hspace=.35)
    for i in range(n_row * n_col):
        plt.subplot(n_row, n_col, i + 1)
        plt.imshow(images[i].reshape((h, w)), cmap=plt.cm.gray)
        plt.title(titles[i], size=12)
        plt.xticks(())
        plt.yticks(())


# plot the result of the prediction on a portion of the test set

def title(y_pred, y_test, target_names, i):
    pred_name = target_names[y_pred[i]].rsplit(' ', 1)[-1]
    true_name = target_names[y_test[i]].rsplit(' ', 1)[-1]
    return 'predicted: %s\ntrue:      %s' % (pred_name, true_name)


labelLib, labelGroupSizeS,faceLib=dataRead_1.readFile()

result=np.ones((2,3,5))
sizeArray=[10,20,30,40,50][:]
for i in range(len(sizeArray)) :
  sizeTrainGroup=sizeArray[i]
  y_train,X_train, y_test,X_test = dataRead_1.trainTestSpliter(labelLib,labelGroupSizeS,faceLib,sizeTrainGroup)
  for option in [0,1]:#PCA or LDA
      result[option,:,i]=faceRec(option,sizeTrainGroup,y_train,X_train, y_test,X_test)
print('PCA-----kNN')
print('   -----SVM')
print('   -----SRC')
print('LDA-----kNN')
print('   -----SVM')
print('   -----SRC')
print(np.array2string(result))
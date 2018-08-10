'''
Created on Feb 9, 2018

@author: KangKang
'''
import scipy.io as scio  
import numpy as np
import cv2
import random

def readFile():
	labelGroupSizeS=[]
	dataLib = scio.loadmat('YaleB_32x32') 
	faceLib=dataLib['fea']
	labelLib=dataLib['gnd']
	# print(type(faceLib))
	# print(faceLib)
	# faceLibGrouped=np.ndarray((38*10,32*32))
	id=1
	count=0
	for item in labelLib:
		if item[0]==id:
			# faceLibGrouped[id-1,]
			count+=1
		else:
			# print((id,count))
			labelGroupSizeS.append(count)
			id=item[0]
			count=1

		# print((id,count))
	# print(labelLib)
	labelGroupSizeS.append(count)
	# print(labelLib)
	# print(labelGroupSizeS)
	return labelLib, labelGroupSizeS,faceLib

def randIntGenerator(base,scale,amount):
	counterS=[0]*scale
	result=[]
	counter=0
	i=0
	while counter<amount:
		i=random.randint(0,scale-1)
		if not counterS[i]:
			counterS[i]=1
			result.append(base+i)
			counter+=1
	result.sort()


	# result=list(range(base,base+amount))
	return result

def trainTestSpliter(labelLib,labelGroupSizeS,faceLib,sizeTrainGroup):
	trainIndexes=[]
	testIndexes=[]
	enumerator=0
	lenTest=len(faceLib)-38*sizeTrainGroup
	trainFaceS=np.ndarray((38*sizeTrainGroup,32*32))
	# print(len(labelLib))
	testFaceS=np.ndarray((lenTest,32*32))

	for sizei in labelGroupSizeS:
		trainIndexes+=randIntGenerator(enumerator,sizei,sizeTrainGroup)
		enumerator+=sizei
	trainIndex=0
	testIndex=0
	lenTrainLabel=len(trainIndexes)
	for i in range(enumerator):
		if i==trainIndexes[trainIndex]:
			trainFaceS[trainIndex]=faceLib[i]
			trainIndex+=1
			if trainIndex==lenTrainLabel:#reach last train label, append remaining labels to testIndexes
				testIndexes+=list(range(i+1,enumerator))
				testFaceS[testIndex:]=faceLib[i+1:]
				break
		else:
			testIndexes.append(i)
			testFaceS[testIndex]=faceLib[i]
			testIndex+=1
	# print(testFaceS)
	# print(trainFaceS)
	# print((trainIndexes[:100],testIndexes[:100]))
	# print(len(trainIndexes),len(testIndexes),sum(labelLib))
	
	# for item in trainFaceS:
	# 	displayArray(item)

	# print(trainIndexes)
	trainLabelS=[int(labelLib[i]) for i in trainIndexes ]
	# print(trainLabelS)
	testLabelS=[int(labelLib[i]) for i in testIndexes ]

	return trainLabelS,np.asmatrix(trainFaceS),testLabelS,np.asmatrix(testFaceS)

def displayArray(array):
	array2=np.array(array,copy=True)
	array2=array2.reshape((32,32))
	array2/=255
	cv2.imshow('image',array2)
	cv2.waitKey(0);

def fisherFace(trainLabelS,trainFaceS,testLabelS,testFaceS):
	return

def eigenFace(trainLabelS,trainFaceS,testLabelS,testFaceS,selecthr):
	lenTest=len(testFaceS)
	# trainFaceS=trainFaceS[:3]
	avgFace = np.mean(trainFaceS,0)
	# diffTrain=trainFaceS-avgFace.T
	trainFaceS-=avgFace
	testFaceS-=avgFace
	# print(testFaceS)
	# print(np.shape(np.dot(trainFaceS,trainFaceS.T)))
	# eigVals,eigVects = np.linalg.eig(np.dot(trainFaceS,trainFaceS.T))
	eigVals,eigVects = np.linalg.eig(np.dot(trainFaceS.T,trainFaceS))
	# print(eigVals)
	# print(np.dot(trainFaceS.T,trainFaceS).shape)
	# print(np.dot(trainFaceS,trainFaceS.T).shape)

	# print(eigVals.shape)
	# print(eigVects.shape)
	eigSortIndexes = np.argsort(-eigVals)
	# print(124,eigVals)

	# print(eigSortIndexes)
	sumEigVal=eigVals.sum()
	n=0
	for i in range(len(trainLabelS)):
		n+=eigVals[eigSortIndexes[i]]/sumEigVal
		if n >= selecthr:
			eigSortIndexes = eigSortIndexes[:i+1]
			break
	# print(eigVects.shape)
	eigVects=eigVects[:,eigSortIndexes]
	# print(eigVects.shape)
	# covVects = np.dot(trainFaceS.T,eigVects) # covVects is the eigenvector of covariance matrix
	covVects = eigVects[:,2:]
	# print(covVects.shape)
	# print(covVects)
	# return
	# for vec in covVects:
	# 	displayArray(vec)
	
	testCounter=[0]*lenTest
	# weiAveFaces=
	# print(sum(sum(trainFaceS)))
	# for item in trainFaceS:
	# 	print((trainFaceS))
	# return
	for i in range(lenTest):
		# print(testFaceS[i])
		# displayArray(testFaceS[i])
		# return
		# print(i)
		# print(sum(sum(trainFaceS)))
		testCounter[i]=trainLabelS[eigenFaceTest(testFaceS[i],trainFaceS,covVects)]
	return testCounter
	# print(testCounter)
	# print(testLabelS[:10])

def eigenFaceTest(testFace,trainFaceS,covVects):
	sampleVec=np.dot(covVects.T,testFace.T)
	lenTrain=len(trainFaceS)
	minDiff=-1
	labelMatch=-1
	for i in range(lenTrain):
		trainVec=np.dot(covVects.T,trainFaceS[i].T)
		diffI=(np.array(sampleVec-trainVec)**2).sum()
		# print(diffI)
		# return
		if minDiff==-1:
			minDiff=diffI
			labelMatch=i
		elif diffI<minDiff:
				minDiff=diffI
				labelMatch=i
	# print(minDiff)
	return labelMatch
def svmClassifier(X_train, y_train, X_test,y_test):


	###############################################################################
	# Train a SVM classification model

	print("Fitting the classifier to the training set")
	t0 = time()
	param_grid = {'C': [1e3, 5e3, 1e4, 5e4, 1e5],
	              'gamma': [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.1], }
	clf = GridSearchCV(SVC(kernel='rbf', class_weight=None), param_grid)
	# clf=SVC()
	clf = clf.fit(X_train_pca, y_train)
	print("done in %0.3fs" % (time() - t0))
	print("Best estimator found by grid search:")
	print(clf.best_estimator_)


	###############################################################################
	# Quantitative evaluation of the model quality on the test set

	print("Predicting people's names on the test set")
	t0 = time()
	y_pred = clf.predict(X_test_pca)
	# y_test=[str(i) for i in y_test]
	# y_pred=[str(i) for i in y_pred]
	# print(y_test)
	print("done in %0.3fs" % (time() - t0))
	# print([str(i) for i in y_test.T])
	# for item in y_pred:
	  # print(item)
	# print(y_test.T.shape,y_pred.T.shape)
	print(classification_report(y_test, y_pred, target_names=target_names))
	print(confusion_matrix(y_test, y_pred, labels=range(n_classes)))




def ReconginitionVector(FaceMat,selecthr = 0.8):
    # step1: load the face image data ,get the matrix consists of all image
    # FaceMat = loadImageSet('D:\python/face recongnition\YALE\YALE\unpadded/').T
    # step2: average the FaceMat
    avgImg = np.mean(FaceMat,1)
    # step3: calculate the difference of avgImg and all image data(FaceMat)
    diffTrain = FaceMat-avgImg
    # print(np.shape(diffTrain.T*diffTrain))
    #step4: calculate eigenvector of covariance matrix (because covariance matrix will cause memory error)
    eigvals,eigVects = np.linalg.eig(np.mat(diffTrain.T*diffTrain))
    # print (eigvals)
    eigSortIndex = np.argsort(-eigvals)
    for i in range(np.shape(FaceMat)[1]):
        if (eigvals[eigSortIndex[:i]]/eigvals.sum()).sum() >= selecthr:
            eigSortIndex = eigSortIndex[:i]
            break
    covVects = diffTrain * eigVects[:,eigSortIndex] # covVects is the eigenvector of covariance matrix
    # avgImg 是均值图像，covVects是协方差矩阵的特征向量，diffTrain是偏差矩阵
    return avgImg,covVects,diffTrain
 
def judgeFace(judgeImg,FaceVector,avgImg,diffTrain):
    # print(judgeImg)
    # print(avgImg)
    diff = judgeImg- avgImg.T
    # print(diff)
    # print(np.shape(diff))
    # return
    # print(np.shape(FaceVector))
    weiVec = FaceVector.T* diff.T
    # print(np.shape(weiVec))
    # return
    res = 0
    resVal = np.inf
    for i in range(np.shape(diffTrain)[1]):
        TrainVec = FaceVector.T*diffTrain[:,i]
        # print(np.shape(TrainVec))
        # return
        if  (np.array(weiVec-TrainVec)**2).sum() < resVal:
            res =  i
            resVal = (np.array(weiVec-TrainVec)**2).sum()
    return res+1

def checkAccuracy(testArray,truthArray):
	count=0
	for i in range(len(testArray)):
		if testArray[i]==truthArray[i]:
			count+=1
	return (count/len(testArray))




peopleSize=38

labelLib, labelGroupSizeS,faceLib=readFile()

for sizeTrainGroup in [10,20,30,40,50][:]:
	trainIndexes,trainFaceS,testIndexes,testFaceS=trainTestSpliter(labelLib,labelGroupSizeS,faceLib,sizeTrainGroup)
	# print(trainFaceS.shape)
	eigenResult=eigenFace(trainIndexes,trainFaceS,testIndexes,testFaceS,0.9)
	
	print(sizeTrainGroup,checkAccuracy(testIndexes, eigenResult))


# trainIndexes,trainFaceS,testIndexes,testFaceS=trainTestSpliter(labelLib,labelGroupSizeS,faceLib,10)
# avgImg,FaceVector,diffTrain = ReconginitionVector(np.mat(trainFaceS.T),0.9)
# print(FaceVector)
# print(trainIndexes)
# result=[]
# for i in range(200):
# 	result.append(trainIndexes[ judgeFace(testFaceS[i],FaceVector,avgImg,diffTrain)]==testIndexes[i])
# print(sum(result)/len(result))
# print(len(trainFaceS))


# for face in trainFaceS[0]:
# 	print(result)
# from dataRead import *
import scipy.io as scio  
import numpy as np
import cv2
from sklearn.metrics.pairwise import euclidean_distances
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve


def read2():
	labelGroupSizeS=[]
	dataLib = scio.loadmat('YaleB_32x32') 
	return (dataLib['fea'][:64],dataLib['fea'][64:128],dataLib['gnd'][:128])
	# labelLib=dataLib['gnd']


def resizeBy0(imageS,currH,currL,targetH,targetL):
	if len(imageS[0])!=currL*currH:
		print('wrong input')
		return []
	meanS=np.mean(imageS,0)
	# print(len(meanS))
	result=np.zeros([len(imageS),targetH,targetL,3],dtype='float32')
	counter=0
	for i,image in enumerate(imageS):
		for row in range(currH):
			for col in range(currL):
				result[i][row][col][0]=result[i][row][col][1]=result[i][row][col][2]=(image[counter]-meanS[counter])/256
				counter+=1
		# print(result[i])
		# cv2.imshow('image',result[i])
		# cv2.waitKey(0)
		counter=0
	return result
def feaVeri(feaArray1,feaArray2):
	len1=len(feaArray1)
	len2=len(feaArray2)
	lenSameClass=len1*(len1-1)+len2*(len2-1)
	testSameClass=np.zeros(lenSameClass,dtype='int16')
	testDiffClass=np.ones(len1*len2,dtype='int16')

	scoreSameClass=np.zeros(lenSameClass,dtype=float)
	counter=0
	# for i in range(len1):
	# 	for j in range(len1):
	# 		if i==j:
	# 			continue
	# 		scoreSameClass[counter]=num
	# 		counter+=1
	# 		scoreSameClass[counter]=num


	for num in euclidean_distances(feaArray1,feaArray1).reshape(len1*len1):
		if num:
			scoreSameClass[counter]=num

		# print(scoreSameClass,num)
			counter+=1
	for num in euclidean_distances(feaArray2,feaArray2).reshape(len2*len2):
		if num:
			scoreSameClass[counter]=num
			counter+=1
		# print(scoreSameClass,num)

	scorediffClass=np.array(euclidean_distances(feaArray1,feaArray2)).reshape(len1*len2)
	# sameClass1=
	# print(scoreSameClass)
	# print(scorediffClass)
	return (np.append(testSameClass,testDiffClass),np.append(scoreSameClass,scorediffClass))

def ROCfigure(f1,t1,f2,t2,title):
	plt.figure()
	lw = 1
	plt.plot(f1, t1, color='darkorange',
	         lw=lw, label='original')
	plt.plot(f2, t2, color='blue',
	         lw=lw, label='after fine tune')
	plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
	plt.xlim([0.0, 1.0])
	plt.ylim([0.0, 1.05])
	plt.xlabel('False Positive Rate')
	plt.ylabel('True Positive Rate')
	plt.title(title)
	plt.legend(loc="lower right")
	plt.show()

if __name__ == '__main__':
	a1=[[1,2,1],[3,4,1]]
	a2=[[5,5,1],[5,6,7]]
	y_test,y_score=feaVeri(a1,a2)
	print(y_test,y_score)
	f,t,_=roc_curve(y_test, y_score)
	ROCfigure(f,t)
	# face1S_32x32,face2S_32x32,lableS=read2()
	# face1S=resizeBy0(face1S_32x32,32,32,227,227)


# def writeJson()
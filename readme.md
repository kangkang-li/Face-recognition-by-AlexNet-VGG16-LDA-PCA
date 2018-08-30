# Face Recognition by AlexNet, VGG16, LDA, & PCA

## Summary
Recognition of human face is a technology growing explodingly in recent
years. This technology relies on algorithms to process and classify
digital signals from images or videos. These project helps understanding
the ideas and architecture of fundamental algorithms. Eigenface,
Fisherface, k-nearest neighbors, support vector machine, and sparse
representation classification were implemented on YaleB 32x32 dataset
with the optimization of vector selection, whitening, and the size of
training data. Dropping top 5 eigenvectors significantly improved the
accuracy of eigenface with k-nearest neighbors from 42% to 74%, but not
for others. Fisherface shows overall higher accuracies than eigenface,
and fisherface with sparse representation classification is the optimum
combination with an accuracy at 97%.

# Idea
Face recognition starts from the most intuitive way based on the
geometric features of a face. This approach is limited by the complicate
registration of the marker points, even with state of the art
algorithms. Comparatively, another family of algorithms treats images
and faces as vectors to classify in a n-domentional space, which is
described and introduced in the following paragraphs.

# File Description:
* featureVerification-AlexNet.py:    fine-tune of AlexNet
* featureVerification-VGG:           fine-tune of VGG-16*
* LDA_PCA.py:                        classification by LDA and PCA
* sourceCode includes packages of AlexNet, Vgg-16, PCA. and LDA
* report.pdf reports details of this project
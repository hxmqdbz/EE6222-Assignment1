It is a python code about PCA&LDA using mahalanobis classifier for EE6222.
Dataset:
cifar10
1、preprocess of dataset:
Devide the whole dataset of 60000 pictures into training(50000) and test(10000) data.
normalization.
change rgb to grayscale pictures.
2、what we do:
Combine PCA and LDA method with Mahalanobis classifier,and train it on training dataset.Then apply it on the test dataset to test the accuracy of the classifier.
After that we collect both the accuracy of PCA and LDA on the test dataset with different remain devisions, and then draw them into a graphic to conclude the change of
accuracy with dimension.
3、what is include:
The code include PCA method,LDA method and a mahalanobis classifier.
PCA method:

# It is a python code about PCA&LDA using mahalanobis classifier for EE6222.  
### Dataset:  
cifar10  
### 1、preprocess of dataset:  
Devide the whole dataset of 60000 pictures into training(50000) and test(10000) data.  
Normalization.  
Change rgb to grayscale pictures.  
### 2、what we do:  
Combine PCA and LDA method with Mahalanobis classifier,and train it on training dataset.Then apply it on the test dataset to  test the accuracy of the classifier.  
After that we collect both the accuracy of PCA and LDA on the test dataset with different remain devisions, and then draw them  into a graphic to conclude the change of accuracy with dimension.  
### 3、what is include:  
The code include PCA method,LDA method and a mahalanobis classifier.  
**PCA method:** PCA.py  
**LDA methid:** LDA.py  
**combine of PCA-Mahalanobis:** PCA-Mahalanobis Classifier.py  
**combine of LDA-Mahalanobis:** LDA-Mahalanobis Classifier.py    
__IDE__  
pycharm  
python 3.12  
__Environment__  
Markdown	3.7	3.7  
MarkupSafe	3.0.2	3.0.2  
Pygments	2.19.1	2.19.1  
Werkzeug	3.1.3	3.1.3  
absl-py	2.2.0	2.2.0  
astunparse	1.6.3	1.6.3  
attrs	25.3.0	25.3.0  
certifi	2025.1.31	2025.1.31  
charset-normalizer	3.4.1	3.4.1  
colorama	0.4.6	0.4.6  
contourpy	1.3.1	1.3.1  
cycler	0.12.1	0.12.1  
dm-tree	0.1.9	0.1.9  
docstring_parser	0.16	0.16  
einops	0.8.1	0.8.1  
etils	1.12.2	1.12.2  
flatbuffers	25.2.10	25.2.10  
fonttools	4.56.0	4.56.0  
fsspec	2025.3.0	2025.3.0  
gast	0.6.0	0.6.0  
google-pasta	0.2.0	0.2.0  
googleapis-common-protos	1.69.2	1.69.2  
grpcio	1.71.0	1.71.0  
h5py	3.13.0	3.13.0  
idna	3.10	3.10  
immutabledict	4.2.1	4.2.1  
importlib_resources	6.5.2	6.5.2  
keras	3.9.0	3.9.0  
kiwisolver	1.4.8	1.4.8  
libclang	18.1.1	18.1.1  
markdown-it-py	3.0.0	3.0.0  
matplotlib	3.10.1	3.10.1  
mdurl	0.1.2	0.1.2  
ml_dtypes	0.5.1	0.5.1  
namex	0.0.8	0.0.8  
numpy	2.1.3	2.2.4  
opt_einsum	3.4.0	3.4.0  
optree	0.14.1	0.14.1  
packaging	24.2	24.2  
pillow	11.1.0	11.1.0  
pip	23.2.1	25.0.1  
promise	2.3	2.3  
protobuf	5.29.4	6.30.1  
psutil	7.0.0	7.0.0  
pyarrow	19.0.1	19.0.1  
pyparsing	3.2.1	3.2.1  
python-dateutil	2.9.0.post0	2.9.0.post0  
requests	2.32.3	2.32.3  
rich	13.9.4	13.9.4  
scipy	1.15.2	1.15.2  
setuptools	77.0.3	77.0.3  
simple-parsing	0.1.7	0.1.7  
six	1.17.0	1.17.0  
tensorboard	2.19.0	2.19.0  
tensorboard-data-server	0.7.2	0.7.2  
tensorflow	2.19.0	2.19.0  
tensorflow-datasets	4.9.8	4.9.8  
tensorflow-metadata	1.16.1	1.16.1  
termcolor	2.5.0	2.5.0  
toml	0.10.2	0.10.2  
tqdm	4.67.1	4.67.1  
typing_extensions	4.12.2	4.12.2  
urllib3	2.3.0	2.3.0   
wheel	0.45.1	0.45.1  
wrapt	1.17.2	1.17.2  
zipp	3.21.0	3.21.0  

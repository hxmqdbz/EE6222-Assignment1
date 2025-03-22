import numpy as np
from scipy.spatial.distance import mahalanobis
import LDA
import tensorflow as tf
import tensorflow_datasets as tfds
import matplotlib.pyplot as plt
#from tensorflow.python.keras.api.datasets import load_data

#(X_train, y_train), (X_test, y_test) = keras.Sequential.cifar10.load_data()
(ds_train, ds_test), ds_info = tfds.load('cifar10',
                                        split=['train', 'test'],
                                        as_supervised=True,
                                        with_info=True)

# 预处理：归一化到 [0,1]
def preprocess(image, label):
    image = tf.image.convert_image_dtype(image, tf.float32)  # 归一化
    image = tf.image.rgb_to_grayscale(image)  # 转为灰度图，降维
    return tf.reshape(image, (-1,)), label  # 展平


# 应用预处理
ds_train = ds_train.map(preprocess).batch(50000)  # 一次性加载所有数据
ds_test = ds_test.map(preprocess).batch(10000)

# 提取 NumPy 数据
for images, labels in ds_train:
    x_train, y_train = images.numpy(), labels.numpy()
for images, labels in ds_test:
    x_test, y_test = images.numpy(), labels.numpy()

class LDA_MahalanobisClassifier:
    def __init__(self, n_components):
        self.lda = LDA.LDA(n_components)
        self.means = {}
        self.inv_cov_matrix = None

    def fit(self, X, y):
        X_reduced = self.lda.fit_transform(X, y)
        classes = np.unique(y)
        self.means = {cls: X_reduced[y == cls].mean(axis=0) for cls in classes}
        cov_matrix = np.cov(X_reduced, rowvar=False)
        self.inv_cov_matrix = np.linalg.pinv(cov_matrix)

    def predict(self, X):
        X_reduced = self.lda.transform(X)
        predictions = []
        for x in X_reduced:
            distances = {cls: mahalanobis(x, mean, self.inv_cov_matrix) for cls, mean in self.means.items()}
            predictions.append(min(distances, key=distances.get))
        return np.array(predictions)



if __name__ == "__main__":
    #np.random.seed(42)
    #X_class1 = np.random.multivariate_normal(mean=[2, 2, 2], cov=np.eye(3), size=50)
    #X_class2 = np.random.multivariate_normal(mean=[6, 6, 6], cov=np.eye(3), size=50)
    #X = np.vstack((X_class1, X_class2))
    #y = np.array([0] * 50 + [1] * 50)
    accuracy = []
    division = []
    for i in range(10, 900):
        classifier = LDA_MahalanobisClassifier(n_components=i)
        classifier.fit(x_train, y_train)

        #X_test = np.array([[3, 3, 3], [5, 5, 5], [7, 7, 7]])
        predictions = classifier.predict(x_test)
        #print("Predictions:", predictions)

        accuracy.append(np.mean(predictions == y_test))
        division.append(i)
        print("i:", i, "accuracy:", np.mean(predictions == y_test))
        # print(f"马哈拉诺比斯分类器在 CIFAR-10 上的准确率: {accuracy:.4f}")
    plt.plot(division, accuracy)
    plt.xlabel("Dimension")
    plt.ylabel("Accuracy")
    plt.show()


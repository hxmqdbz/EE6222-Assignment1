import numpy as np
from scipy.spatial.distance import mahalanobis


class MahalanobisClassifier:
    def __init__(self):
        self.means = {}
        self.inv_cov_matrix = None

    def fit(self, X, y):
        """
        训练分类器，计算每个类别的均值和协方差矩阵的逆。
        :param X: 训练数据，形状为 (n_samples, n_features)
        :param y: 标签，形状为 (n_samples,)
        """
        classes = np.unique(y)
        self.means = {cls: X[y == cls].mean(axis=0) for cls in classes}

        # 计算协方差矩阵并取逆
        cov_matrix = np.cov(X, rowvar=False)
        self.inv_cov_matrix = np.linalg.inv(cov_matrix)

    def predict(self, X):
        """
        对测试数据进行分类。
        :param X: 测试数据，形状为 (n_samples, n_features)
        :return: 预测标签，形状为 (n_samples,)
        """
        predictions = []
        for x in X:
            distances = {cls: mahalanobis(x, mean, self.inv_cov_matrix) for cls, mean in self.means.items()}
            predictions.append(min(distances, key=distances.get))
        return np.array(predictions)


# 示例用法
if __name__ == "__main__":
    # 生成示例数据
    np.random.seed(42)
    X_class1 = np.random.multivariate_normal(mean=[2, 2], cov=[[1, 0.5], [0.5, 1]], size=50)
    X_class2 = np.random.multivariate_normal(mean=[6, 6], cov=[[1, -0.3], [-0.3, 1]], size=50)
    X = np.vstack((X_class1, X_class2))
    y = np.array([0] * 50 + [1] * 50)

    # 训练分类器
    classifier = MahalanobisClassifier()
    classifier.fit(X, y)

    # 预测新数据
    X_test = np.array([[3, 3], [5, 5], [7, 7]])
    predictions = classifier.predict(X_test)
    print("Predictions:", predictions)

import numpy as np


class PCA:
    def __init__(self, n_components):
        """
        主成分分析 (PCA) 算法
        :param n_components: 要保留的主成分数
        """
        self.n_components = n_components
        self.mean = None
        self.components = None

    def fit(self, X):
        """
        计算数据的主成分
        :param X: 形状为 (n_samples, n_features) 的数据矩阵
        """
        # 计算均值并中心化数据
        self.mean = np.mean(X, axis=0)
        X_centered = X - self.mean

        # 计算协方差矩阵
        cov_matrix = np.cov(X_centered, rowvar=False)

        # 计算特征值和特征向量
        eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)

        # 按特征值降序排序
        sorted_indices = np.argsort(eigenvalues)[::-1]
        self.components = eigenvectors[:, sorted_indices[:self.n_components]]

    def transform(self, X):
        """
        将数据投影到主成分空间
        :param X: 形状为 (n_samples, n_features) 的数据矩阵
        :return: 形状为 (n_samples, n_components) 的降维数据
        """
        X_centered = X - self.mean
        return np.dot(X_centered, self.components)

    def fit_transform(self, X):
        """
        先拟合数据，然后进行降维
        :param X: 形状为 (n_samples, n_features) 的数据矩阵
        :return: 形状为 (n_samples, n_components) 的降维数据
        """
        self.fit(X)
        return self.transform(X)


# 示例用法
if __name__ == "__main__":
    # 生成示例数据
    np.random.seed(42)
    X = np.random.rand(100, 5)  # 100个样本，5个特征

    # 进行PCA降维到2维
    pca = PCA(n_components=2)
    X_reduced = pca.fit_transform(X)
    print("降维后的数据:", X_reduced[:5])

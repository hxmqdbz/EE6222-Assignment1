import numpy as np


class LDA:
    def __init__(self, n_components):
        """
        线性判别分析 (LDA)
        :param n_components: 目标维度数
        """
        self.n_components = n_components
        self.means_ = None
        self.scalings_ = None

    def fit(self, X, y):
        """
        训练LDA模型
        :param X: 形状为 (n_samples, n_features) 的数据矩阵
        :param y: 形状为 (n_samples,) 的类别标签
        """
        classes = np.unique(y)
        n_features = X.shape[1]
        mean_overall = np.mean(X, axis=0)

        # 计算类内散度矩阵 Sw 和类间散度矩阵 Sb
        Sw = np.zeros((n_features, n_features))
        Sb = np.zeros((n_features, n_features))

        self.means_ = {}
        for cls in classes:
            X_cls = X[y == cls]
            mean_cls = np.mean(X_cls, axis=0)
            self.means_[cls] = mean_cls
            Sw += np.dot((X_cls - mean_cls).T, (X_cls - mean_cls))
            mean_diff = (mean_cls - mean_overall).reshape(n_features, 1)
            Sb += X_cls.shape[0] * np.dot(mean_diff, mean_diff.T)

        # 计算 Sw^(-1) * Sb
        Sw_inv = np.linalg.pinv(Sw)  # 计算伪逆
        eigvals, eigvecs = np.linalg.eigh(np.dot(Sw_inv, Sb))

        # 按特征值降序排序
        sorted_indices = np.argsort(eigvals)[::-1]
        self.scalings_ = eigvecs[:, sorted_indices[:self.n_components]]

    def transform(self, X):
        """
        将数据投影到LDA空间
        :param X: 形状为 (n_samples, n_features) 的数据矩阵
        :return: 形状为 (n_samples, n_components) 的降维数据
        """
        return np.dot(X, self.scalings_)

    def fit_transform(self, X, y):
        """
        先拟合数据，然后进行降维
        :param X: 形状为 (n_samples, n_features) 的数据矩阵
        :param y: 形状为 (n_samples,) 的类别标签
        :return: 形状为 (n_samples, n_components) 的降维数据
        """
        self.fit(X, y)
        return self.transform(X)


#测试数据
if __name__ == "__main__":
    np.random.seed(42)
    X_class1 = np.random.multivariate_normal(mean=[2, 2], cov=np.eye(2), size=50)
    X_class2 = np.random.multivariate_normal(mean=[6, 6], cov=np.eye(2), size=50)
    X = np.vstack((X_class1, X_class2))
    y = np.array([0] * 50 + [1] * 50)

    lda = LDA(n_components=1)
    X_reduced = lda.fit_transform(X, y)
    print("降维后的数据:", X_reduced[:5])

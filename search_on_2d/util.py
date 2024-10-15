from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import numpy as np
 
def detect_outliers(X, clusters, per_cent_other_centile=True, percent=0.9, percentile=95):
    """
    使用z分数（Z-score）检测离群点
    X: 数据点
    clusters: KMeans模型
    n_std: 多少个标准差定义为离群点
    """
    cluster_labels = clusters.labels_
    centers = [clusters.cluster_centers_[i] for i in range(clusters.n_clusters)]
 
    # 计算每个点到其簇心的距离
    #distances = [np.linalg.norm(X - center) for center in centers]
    distances = [np.sqrt(np.sum((X - center) ** 2, axis=1)) for center in centers]
 
    # 计算每个点的z分数
    z_scores = [(distance - np.mean(distances)) / np.std(distances) for distance in distances]
 
    # 根据z分数判定离群点
    abs_z_scores = [abs(scores) for scores in z_scores]
    if per_cent_other_centile:
        thresholds = [max(score)* percent for score in abs_z_scores]
    else:
        thresholds = [np.percentile(scores, percentile) for scores in abs_z_scores]
    outliers_masks = [z_scores > thresh for z_scores, thresh in zip(abs_z_scores, thresholds)]
 
    # 返回离群点的标签和掩码
    return [[i for i, x in enumerate(outliers_mask) if x] for outliers_mask in outliers_masks], outliers_masks

if __name__ == '__main__':
    # 示例使用
    X = np.random.rand(100, 2)
    kmeans = KMeans(n_clusters=2)
    kmeans.fit(X)
    #labels, mask = detect_outliers(X, kmeans)
    labels, mask = detect_outliers(X, kmeans, False)
    print("离群点的索引:", labels)
    # 使用scatter函数绘制点
    plt.scatter(X[:, 0], X[:, 1])
    # 显示图形
    plt.show()

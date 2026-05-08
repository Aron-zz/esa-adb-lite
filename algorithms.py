"""6 种经典异常检测算法，纯函数，无 Docker 依赖。"""
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import NearestNeighbors
from sklearn.decomposition import PCA


def _per_channel_scores(X, score_fn):
    """逐通道计算异常分数，返回 (n_samples, n_channels) 形状。"""
    scores = np.empty_like(X, dtype=np.float64)
    for i in range(X.shape[1]):
        col = X[:, i].reshape(-1, 1)
        scores[:, i] = score_fn(col).ravel()
    return scores


def pcc(X_train, X_test, random_state=42):
    """PCA 重构误差"""
    clf = PCA(n_components=2, random_state=random_state).fit(X_train)
    X_recon = clf.inverse_transform(clf.transform(X_test))
    return np.mean((X_test - X_recon) ** 2, axis=1, keepdims=True)


def hbos(X_train, X_test, n_bins=50, random_state=42):
    """Histogram-Based Outlier Score，逐通道。"""
    def _hbos(col):
        n = len(col)
        counts, edges = np.histogram(col, bins=n_bins)
        freqs = np.clip(counts / n, 1e-10, 1.0)
        indices = np.clip(np.digitize(col, edges[1:-1]), 0, n_bins - 1)
        return -np.log(freqs[indices])
    return _per_channel_scores(X_test, _hbos)


def std(X_train, X_test, tol=3.0, random_state=42):
    """统计阈值：偏离 mean ± tol*std 判定为异常。"""
    mean = np.mean(X_train, axis=0)
    stdv = np.std(X_train, axis=0)
    stdv = np.where(stdv == 0, 1.0, stdv)
    return ((X_test > mean + tol * stdv) | (X_test < mean - tol * stdv)).astype(np.float64)


def iforest(X_train, X_test, n_estimators=100, random_state=42):
    """Isolation Forest，逐通道。"""
    def _iforest(col):
        clf = IsolationForest(n_estimators=n_estimators, contamination="auto",
                              random_state=random_state, n_jobs=1)
        clf.fit(col)
        return -clf.decision_function(col)
    return _per_channel_scores(X_test, _iforest)


def knn(X_train, X_test, n_neighbors=5, random_state=42):
    """kNN 距离异常分数。"""
    nn = NearestNeighbors(n_neighbors=n_neighbors, n_jobs=1).fit(X_train)
    dist, _ = nn.kneighbors(X_test)
    return np.mean(dist, axis=1, keepdims=True)


def subsequence_if(X_train, X_test, window_size=17, n_estimators=200, random_state=42):
    """滑动窗口 + IForest，逐通道。"""
    def _sub_if(col):
        n = len(col)
        col_2d = col.reshape(-1, 1)
        # 对训练集所有窗口做 iForest
        windows = np.lib.stride_tricks.sliding_window_view(col_2d.ravel(), window_size)
        clf = IsolationForest(n_estimators=n_estimators, contamination="auto",
                              random_state=random_state, n_jobs=1)
        clf.fit(windows)
        # 测试集窗口评分
        test_win = np.lib.stride_tricks.sliding_window_view(col_2d.ravel(), window_size)
        scores_win = -clf.decision_function(test_win)
        # 将窗口分数映射回点
        scores = np.zeros(n, dtype=np.float64)
        scores[:window_size-1] = scores_win[0]
        scores[window_size-1:] = np.maximum.accumulate(scores_win[:, None])[len(test_win)-n+window_size-1:]
        return scores.reshape(-1, 1)
    return _per_channel_scores(X_test, _sub_if)


ALGORITHMS = {
    "PCC": pcc,
    "HBOS": hbos,
    "STD": std,
    "iForest": iforest,
    "KNN": knn,
    "Subsequence_IF": subsequence_if,
}

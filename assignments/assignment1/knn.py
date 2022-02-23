import numpy as np


class KNN:
    """
    K-neariest-neighbor classifier using L1 loss
    """
    def __init__(self, k=1):
        self.k = k

    def fit(self, X, y):
        self.train_X = X
        self.train_y = y

    def predict(self, X, num_loops=0):
        if num_loops == 0:
            dists = self.compute_distances_no_loops(X)
        elif num_loops == 1:
            dists = self.compute_distances_one_loop(X)
        else:
            dists = self.compute_distances_two_loops(X)

        if self.train_y.dtype == np.bool:
            return self.predict_labels_binary(dists)
        else:
            return self.predict_labels_multiclass(dists)

    def compute_distances_two_loops(self, X):
        num_train = self.train_X.shape[0] # equal len(self.train_X)
        num_test = X.shape[0] # equal len(X)
        dists = np.zeros((num_test, num_train), np.float32)
        for i_test in range(num_test):
            for i_train in range(num_train):
                dists[i_test][i_train] = np.sum(np.abs(X[i_test] - self.train_X[i_train])) 
        return dists

    def compute_distances_one_loop(self, X):
        num_train = self.train_X.shape[0]
        num_test = X.shape[0]
        dists = np.zeros((num_test, num_train), np.float32)
        for i_test in range(num_test):
            dists[i_test] = np.sum(np.abs(X[i_test] - self.train_X), axis = 1) # np.sum(row)
        return dists 

    def compute_distances_no_loops(self, X):
        num_train = self.train_X.shape[0]
        num_test = X.shape[0]
        dists = np.zeros((num_test, num_train), np.float32)
        dists = np.sum(np.abs(X[:, None] - self.train_X[None, : ]), axis = 2) # broadcast, np.sum(3 axis, but it is a row)
        return dists

    def predict_labels_binary(self, dists):
        num_test = dists.shape[0]
        pred = np.zeros(num_test, np.bool)
        for i in range(num_test):
            min_dists = np.argpartition(dists[i], 1) # return array of indexes, k-th element for sorting
            k_nearest_neighbours = self.train_y[min_dists[:self.k]]
            unique_neighbours, counts = np.unique(k_nearest_neighbours, return_counts=True)
            pred[i] = unique_neighbours[np.argmax(counts)]
        return pred

    def predict_labels_multiclass(self, dists):
        num_test = dists.shape[0]
        pred = np.zeros(num_test, np.int)
        for i in range(num_test):
            min_dists = np.argpartition(dists[i], 1) # return array of indexes, k-th element for sorting
            k_nearest_neighbours = self.train_y[min_dists[:self.k]]
            unique_neighbours, counts = np.unique(k_nearest_neighbours, return_counts=True)
            pred[i] = unique_neighbours[np.argmax(counts)]
        return pred

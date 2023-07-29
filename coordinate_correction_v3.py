import cv2
import numpy as np
import matplotlib.pyplot as plt

from numpy import ndarray
from tqdm import tqdm
from pycpd import RigidRegistration
from scipy.spatial.distance import cdist


class RANSAC:
    def __init__(self, iters: int = 1000, n_samples: int = 8):
        self._iters = iters
        self._n_samples = n_samples

    def _closest_match(self, pts0: ndarray, pts1: ndarray) -> ndarray:
        best_match_idx = np.argmin(cdist(pts0, pts1), axis=1)
        return pts1[best_match_idx]
    
    def _add_bias(self, X: ndarray) -> ndarray:
        bias = np.ones(X.shape[0], dtype=X.dtype)
        return np.c_[X, bias]
    
    def _lsq(self, X: ndarray, Y: ndarray) -> ndarray:
        X = self._add_bias(X)
        return np.linalg.pinv(X).dot(Y)

    def __call__(self, X: ndarray, Y: ndarray) -> ndarray:
        best_loss = np.inf 
        X_best = X

        with tqdm(total=self._iters) as pbar:
            for _ in range(self._iters):

                # Select subsample of points in `pts0`
                indices = np.random.permutation(X.shape[0])
                X_subset = X[indices[:self._n_samples]]

                # Identify closest points in `pts1`
                Y_subset = self._closest_match(X_subset, Y)

                # Compute least-squares transformation
                weights = self._lsq(X_subset, Y_subset)
                
                # Transform all other points acc. to fitted transform
                Y_pred = self._add_bias(X).dot(weights)

                # Compute loss -> the average distance to closest point
                loss = np.mean(np.linalg.norm(Y_pred - self._closest_match(Y_pred, Y), axis=1))

                if loss < best_loss:
                    X_best = Y_pred
                    best_loss = loss

                pbar.set_postfix_str('Loss = %.2f - Best = %.2f' % (loss, best_loss))
                pbar.update(1)

        return X_best



class PointAlignment:
    def __init__(self, iters: int = 5000, n_samples: int = 12, verbose: bool = True):
        self._iters = iters
        self._n_samples = n_samples
        self._verbose = verbose
        self._ransac = RANSAC(iters=iters, n_samples=n_samples)

    def __plot(self, Y, Z1, Z2):
        plt.figure()
        plt.subplot(1, 2, 1)
        plt.plot(Y[:, 0], Y[:, 1], '.', c='C0', ms=5)
        plt.plot(Z1[:, 0], Z1[:, 1], '.', c='C1', ms=3)
        plt.title('RigidRegistration')

        plt.subplot(1, 2, 2)
        plt.plot(Y[:, 0], Y[:, 1], '.', c='C0', ms=5)
        plt.plot(Z2[:, 0], Z2[:, 1], '.', c='C1', ms=3)
        plt.title('RigidRegistration + RanSaC')
        plt.show()

    def rigid_ransac_registration(self, X: ndarray, Y: ndarray) -> ndarray:
        """Aligns point cloud in spherical coordinates (`X`) with point cloud in screen coordinates (`Y`) 
        using a scale-invariant rigid registration algorithm followed by Random Sample Consensus (RanSaC). 

        Args:
        ndarray X: Source points in spherical coordinates (longitude, latitude).
        ndarray Y: Target points in screen space coordinates (pixels).

        Returns: Points of `X` adjusted to align with point cloud `Y`.
        """
        Z1, _ = RigidRegistration(X=Y, Y=X).register()
        Z2 = self._ransac(X=Z1, Y=Y)

        if self._verbose:
            self.__plot(Y, Z1, Z2)

        weights = self._ransac._lsq(X, Z2)
        return Z2, weights



if __name__ == '__main__':

    N = 60
    pts0 = np.zeros((N, 2), dtype=np.float32)
    pts0[:, 0] = np.random.uniform(-80, 30, N)
    pts0[:, 1] = np.random.uniform(-5, 95, N)

    # Randomly distort data
    t = np.mean(pts0, axis=0, keepdims=True)
    angle = .5 * np.random.uniform(-np.pi, np.pi)
    rot = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    scale = np.random.uniform(1, 4, size=(1, 2))
    trans = np.random.uniform(-100, 100, size=(1, 2))

    pts1 = ((pts0 - t).dot(rot) * scale) + t + trans

    # Plot original data
    plt.plot(pts0[:, 0], pts0[:, 1], '.', c='C0', ms=5)
    plt.plot(pts1[:, 0], pts1[:, 1], '.', c='C1', ms=3)
    plt.title('Original')
    plt.show()


    sp_coords, weights = PointAlignment().rigid_ransac_registration(pts0, pts1)
    print(weights)

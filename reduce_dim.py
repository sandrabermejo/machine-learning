# -*- coding: utf-8 -*-
"""Module for dimensionality reduction.

This module implements two ways to reduce the dimensionality
of data by using Linear Discriminant Analysis (LDA) and
Principal Component Analysis (PCA) techniques.
"""

from __future__ import division
import numpy as np
from scipy.linalg import eigh


class LDA:
    """Implementation of Linear Discriminant Analysis.

    The goal of LDA is to project a multiclass dataset onto
    a lower-dimensional space with good class-separability.

    First, the optimal linear projection is computed by calling
    the method `fit`. Then, the method `transform` projects data
    to maximize class separation.

    See "Pattern Recognition and Machine Learning" by Bishop, Chapter 4.1.6.

    Attributes
    ----------
    mean : numpy.ndarray
           Mean of the training points.
    W : numpy.ndarray
        Projection matrix.

    """

    def fit(self, X_train, y_train, reduced_dim):
        """Compute the projection matrix W.

        The weight vectors of `W` are determined by those eigenvectors of
        `(SW^-1) * SB` that correspond to the `reduced_dim` largest
        eigenvalues, where `SW` is the within-class covariance matrix and
        `SB` is the between-class covariance matrix.

        Parameters
        ----------
        X_train : numpy.ndarray
            List of training points of dimension D,
            [[x1,...,xD],...,[z1,...,zD]].
        y_train : numpy.ndarray
            List of integers between 0 and `n_classes` - 1 representing
            the class where each point of `X_train` belongs to.
        reduced_dim : int
            The dimension (<= `n_classes` - 1) wanted for the projected data.

        Examples
        --------
        >>> lda = LDA()
        >>> X_train = np.array([[1, 2, 3], [5, 6, 7], [2, -2, 2], [1, -3, -4],
        [2, 6, 9], [0, 2, 8]])
        >>> y_train = np.array([0, 1, 1, 2, 0, 2])
        >>> lda.fit(X_train, y_train, 2)

        Check the weight vectors of `W`

        >>> lda.W
        array([[-2.46417826,  0.02349338],
               [ 1.445442  ,  0.19252487],
               [-0.76617386, -0.05807627]])

        """

        n_classes = max(y_train) + 1
        self.mean = np.add.reduce(X_train, 0) / X_train.shape[0]

        # Sort X_train and y_train for faster class separation
        sort_args = y_train.argsort()
        X_train, y_train = X_train[sort_args], y_train[sort_args]

        # Get starting index of each class
        class_limit = y_train.searchsorted(range(n_classes), 'right')

        # Compute matrices SW and SB
        right_limit = class_limit[0]
        X0 = X_train[:right_limit]
        size_x0 = X0.shape[0]
        mean_x0 = np.add.reduce(X0, 0) / size_x0

        SW = (lambda x: x.T.dot(x))(X0 - mean_x0)
        SB = size_x0 * (lambda x: x[:, None].dot(x[None]))(self.mean - mean_x0)

        for k in range(1, n_classes):
            left_limit, right_limit = right_limit, class_limit[k]
            Xk = X_train[left_limit:right_limit]
            size_xk = Xk.shape[0]
            mean_xk = np.add.reduce(Xk, 0) / size_xk
            SW = SW + (lambda x: x.T.dot(x))(Xk - mean_xk)
            SB = SB + size_xk * \
                (lambda x: x[:, None].dot(x[None]))(self.mean - mean_xk)

        # Find eigenvectors of matrix (SW^-1)*SB
        _, eigenvectors = eigh(SB, SW, check_finite=False)

        # Take the eigenvectors corresponding to the 'reduced_dim'
        # largest eigenvalues, in descending order
        self.W = (eigenvectors[:, -reduced_dim:])[:, ::-1]

    def transform(self, X):
        """Projects the input data `X` onto `reduced_dim` dimensions.

        Parameters
        ----------
        X : numpy.ndarray
            List of points of dimension D, [[x1,...,xD],...,[z1,...,zD]].

        Returns
        -------
        numpy.ndarray
            The projected points, which have dimension `reduced_dim`.

        Examples
        --------
        >>> lda = LDA()
        >>> X_train = np.array([[1, 2, 3], [5, 6, 7], [2, -2, 2], [1, -3, -4],
        [2, 6, 9], [0, 2, 8]])
        >>> y_train = np.array([0, 1, 1, 2, 0, 2])
        >>> lda.fit(X_train, y_train, 2)

        >>> X = np.array([[2, 0, 5], [1, 5, 2]])
        >>> lda.transform(X)
        array([[-3.6991516 , -0.39744358],
               [ 8.29075825,  0.71591617]])

        """

        # Center data before projection
        return (X - self.mean).dot(self.W)


class PCA:
    """Implementation of Principal Component Analysis.

    PCA seeks a space of lower dimensionality, such that the
    orthogonal projection of the data points onto this subspace
    minimizes the average projection cost, defined as the mean
    squared distance between the data point and their projections.
    This is equivalent to maximize the variance of the projected points.

    First, the optimal linear projection is computed by calling
    the method `fit`. Then, the method `transform` applies the
    dimensionality reduction on the input data.

    See "Pattern Recognition and Machine Learning" by Bishop, Chapter 12.

    Attributes
    ----------
    mean : numpy.ndarray
           Mean of the training points.
    W : numpy.ndarray
        Projection matrix.

    """

    def fit(self, X_train, reduced_dim):
        """Computes the projection matrix W.

        The weight vectors of `W` are determined by those eigenvectors
        of `S` that correspond to the `reduced_dim` largest eigenvalues,
        where `S` is the data covariance matrix.

        Parameters
        ----------
        X_train : numpy.ndarray
            List of training points of dimension D,
            [[x1,...,xD],...,[z1,...,zD]].
        reduced_dim : int
            The dimension (< D) wanted for the projected data.

        Examples
        --------
        >>> pca = PCA()
        >>> X_train = np.array([[1, 2, 3], [4, 5, 6], [7, 4, 0]])
        >>> pca.fit(X_train, 2)

        Check the weight vectors of `W`

        >>> pca.W
        array([[-0.72843861,  0.51425592],
               [-0.10370276,  0.57036068],
               [ 0.67721704,  0.64049163]])

        """

        self.mean = np.add.reduce(X_train, 0) / X_train.shape[0]

        # Compute matrix S
        S = (lambda x: x.T.dot(x))(X_train - self.mean)

        # Find eigenvectors of matrix S
        _, eigenvectors = eigh(S, check_finite=False, overwrite_a=True)

        # Take the eigenvectors corresponding to the 'reduced_dim'
        # largest eigenvalues, in descending order
        self.W = (eigenvectors[:, -reduced_dim:])[:, ::-1]

    def transform(self, X):
        """Projects the input data `X` onto `reduced_dim` dimensions.

        Parameters
        ----------
        X : numpy.ndarray
            List of points of dimension D, [[x1,...,xD],...,[z1,...,zD]].

        Returns
        -------
        numpy.ndarray
            The projected points, which have dimension `reduced_dim`.

        Examples
        --------
        >>> pca = PCA()
        >>> X_train = np.array([[1, 2, 3], [4, 5, 6], [7, 4, 0]])
        >>> pca.fit(X_train, 2)

        >>> X = np.array([[0, 3, 0], [-1, 1, 2]])
        >>> pca.transform(X)
        array([[ 0.95123849, -4.35873906],
               [ 3.24151672, -4.73273308]])

        """

        # Center data before projection
        return (X - self.mean).dot(self.W)

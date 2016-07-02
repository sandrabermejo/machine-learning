# -*- coding: utf-8 -*-

from __future__ import division
import numpy as np


class Fisher:
    """Implementation of Fisher's algorithm for classification.

    The goal is to classify two classes of points, by choosing a threshold
    such that minimizes the mistake probability (a mistake occurs when an
    input vector belonging to class C0 is assigned to class C1 or vice versa).

    Attributes
    ----------
    w : numpy.ndarray
        Weight vector. It is used to project an input vector onto one
        dimension. We choose 'w' so that gives a large separation
        between the projected class means while also giving a small variance
        within each class, thereby minimizing the class overlap.
    c : numpy.float64
        Optimal threshold that minimizes the mistake probability. We model the
        class-conditional densities using Normal distributions.

    """

    def compute_w(self, X0, X1, mean_x0, mean_x1):
        """Computes the weight vector 'w'.

        Parameters
        ----------
        param1 : X0
            List of training points [[x0,...,xD],...,[z0,...,zD]].
        param2 : X1
            List of training points [[x0,...,xD],...,[z0,...,zD]].
        param3 : mean_x0
            Mean of list X0 [x0,...,xD].
        param4 : mean_x1
            Mean of list X1 [x0,...,xD].

        """

        sw_matrix = np.cov(X0, rowvar=0) + np.cov(X1, rowvar=0)
        self.w = np.linalg.solve(sw_matrix, mean_x1 - mean_x0)
        self.w = self.w / np.linalg.norm(self.w)


    def get_coeffs(self, X0, X1, mean_x0, mean_x1):
        """Computes the coefficients of the quadratic equation used to
        get the threshold 'c'.

        Parameters
        ----------
        param1 : X0
            List of training points [[x0,...,xD],...,[z0,...,zD]].
        param2 : X1
            List of training points [[x0,...,xD],...,[z0,...,zD]].
        param3 : mean_x0
            Mean of list X0 [x0,...,xD].
        param4 : mean_x1
            Mean of list X1 [x0,...,xD].

        Returns
        -------
        (float, float, float)
            Coefficients of the quadratic equation used to get the threshold.
        """

        shape_x0 = np.shape(X0)[0]
        shape_x1 = np.shape(X1)[0]
        mu_x0 = self.w.dot(mean_x0[:, np.newaxis])
        mu_x1 = self.w.dot(mean_x1[:, np.newaxis])
        prob_x0 = shape_x0 / (shape_x0 + shape_x1)
        prob_x1 = shape_x1 / (shape_x0 + shape_x1)
        sigma_x0 = np.std(self.w.dot(X0.T))
        sigma_x1 = np.std(self.w.dot(X1.T))

        coeff_c = -1 / 2 * mu_x0**2 / (sigma_x0**2) \
            + 1 / 2 * mu_x1**2 / (sigma_x1**2) \
            + np.log(prob_x0 / sigma_x0) \
            - np.log(prob_x1 / sigma_x1)
        coeff_b = mu_x0 / (sigma_x0**2) - mu_x1 / (sigma_x1**2)
        coeff_a = -1 / (2 * sigma_x0**2) + 1 / (2 * sigma_x1**2)

        return float(coeff_a), float(coeff_b), float(coeff_c)


    def compute_c(self, X0, X1, mean_x0, mean_x1):
        """Computes the threshold 'c'.

        Parameters
        ----------
        param1 : X0
            List of training points [[x0,...,xD],...,[z0,...,zD]].
        param2 : X1
            List of training points [[x0,...,xD],...,[z0,...,zD]].
        param3 : mean_x0
            Mean of list X0 [x0,...,xD].
        param4 : mean_x1
            Mean of list X1 [x0,...,xD].

        """

        coeff_a, coeff_b, coeff_c = self.get_coeffs(X0, X1, mean_x0, mean_x1)
        c_solutions = np.roots([coeff_a, coeff_b, coeff_c])
        c1 = np.amin(c_solutions)
        c2 = np.amax(c_solutions)
        evalua = np.polyval([coeff_a, coeff_b, coeff_c], np.mean(c_solutions))
        self.c = c1
        if (evalua > 0):
            self.c = c2


    def train_fisher(self, X0, X1):
        """Computes the class attributes: the weight vector 'w' and
        the threshold 'c'.

        Parameters
        ----------
        param1 : X0
            List of training points [[x0,...,xD],...,[z0,...,zD]].
        param2 : X1
            List of training points [[x0,...,xD],...,[z0,...,zD]].

        """

        X0 = np.asarray(X0)
        X1 = np.asarray(X1)
        mean_x0 = np.mean(X0, axis=0)
        mean_x1 = np.mean(X1, axis=0)
        self.compute_w(X0, X1, mean_x0, mean_x1)
        self.compute_c(X0, X1, mean_x0, mean_x1)


    def classify_fisher(self, X):
        """Classifies a list of points.

        We classify a point x as belonging to C0 if y(x) >= c and
        we classify it as belonging to C1 otherwise.

        Parameters
        ----------
        param1 : X
            List of points for classification [[x0,...,xD],...,[z0,...,zD]].

        Returns
        -------
        list
            List of classes [c0,c1,...,cM] where 'ci' is 0 if the point belongs
            to class C0 or 1 if the point belongs to class C1.

        """

        X = np.asarray(X)
        y = self.w.dot(X.T)
        classes = (y >= self.c).astype(int)
        return classes.tolist()

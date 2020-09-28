# -----------------------------------------------------------------------------
# Copyright 2019 (C) Jeremy Fix
# Released under a BSD two-clauses license
#
# Reference: B. Scholkopf, A. Smola, K-R. Müller (1999)
#            Kernel principal component analysis.
#            Advances in Kernel methods - Support Vector Machines, 327-352.
# Dataset : Database of faces from AT&T Laboratories Cambridge
#           https://www.cl.cam.ac.uk/research/dtg/attarchive/facedatabase.html
# -----------------------------------------------------------------------------
import numpy as np

import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import matplotlib.cm as cmx
import matplotlib.colors as colors


class KernelPCA:
    ''' Kernel PCA class '''

    def __init__(self, n_components, kernel):
        '''
        Arguments
        ---------
        n_components(int) : the nomber of components to keep
        kernel(function) : k(x, y) = <phi(x), phi(y)>
        '''
        self.n_components = n_components
        self.kernel = kernel
        self.X = None

    def fit(self, X):
        '''
        Compute the PCA by diagonalizing the double centered Gram matrix
        The data are supposed to be laid out in the rows of X
        '''

        # We need to keep a copy of the datapoints for later computing
        # the dot products in the feature space
        self.X = X.copy()

        # Compute the Gram matrix, i.e. the matric of the kernels k(x_i, x_j)
        N = X.shape[0]
        G = self.compute_gram(self.X, self.X)

        # Apply the double centering transform
        Gdc = (np.eye(N) - 1/N * np.ones((N, N))) @ G @ (np.eye(N) - 1/N * np.ones((N, N)))

        # Compute the eigenvalue/eigenvector decomposition of Gdc
        self.eigvals, self.eigvecs = np.linalg.eigh(Gdc)

        # It might be that some eigvals < 0
        # which theoretically cannot happen
        self.eigvals[self.eigvals < 0] = 0
        self.normalization = np.sqrt(self.eigvals)
        self.normalization[self.normalization == 0] = 1.
        self.eigvecs = self.eigvecs / self.normalization

    def compute_gram(self, X, Y):
        '''
        Computes the matrix of the dot products between the rows of X
        and the rows of Y
        '''
        Nx, Ny = X.shape[0], Y.shape[0]
        G = np.zeros((Nx, Ny))
        for i in range(Nx):
            for j in range(Ny):
                G[i, j] = self.kernel(X[i, :], Y[j, :])
        return G

    def transform(self, Z):
        '''
        Uses a fitted PCA to transform the row vectors of Z
        Remember the eigen vectors are ordred by ascending order
        Denoting Z_trans = transform(Z),
        The first  component is Z_trans[:, -1]
        The second component is Z_trans[:, -2]
        The coordinates in the projected space of the i-th sample
        is Z_trans[i, :]
        '''
        # Build the matrix of the kernels Kxz_{i,j} = kernel(X[i, :], Z[j, :])
        G = self.compute_gram(self.X, Z)
        return G.T @ self.eigvecs[:, -self.n_components:]


if __name__ == '__main__':

    # Example 1 : 400 x (112, 92) Olivetti face dataset
    # We have much more dimensions (10304) than samples (400), a situation
    # where computing the PCA from the Gramm matrix is advantageous
    # -------------------------------------------------------------------------
    print("AT&T faces dataset example")
    print('-' * 70)

    samples = np.load('att_faces.npy')
    X, y = samples['input'], samples['label']

    # Kernels
    linear_kernel = lambda x, y: np.dot(x, y)

    # Extract the 15 first principal component vectors
    n_components = 10
    kernel_pca = KernelPCA(n_components=n_components, kernel=linear_kernel)
    kernel_pca.fit(X)

    # Project the original data
    X_trans = kernel_pca.transform(X)

    print("{:.2f}% of the variance is kept with {} components".format(
          100 * kernel_pca.eigvals[-n_components:].sum()/kernel_pca.eigvals.sum(),
          n_components))

    # Plot
    fig = plt.figure(figsize=(10, 4), tight_layout=True)
    gs = GridSpec(3, n_components)

    # Coordinates in the projected space
    # the color shows how the digits get separated by the principal vectors
    ax = fig.add_subplot(gs[0:2, :])
    ax.scatter(X_trans[:, -1], X_trans[:, -2], alpha=.25)
    plt.savefig('eigenface.png')

    # Show some samples ordered by 1st principal component
    idx_sorted_by_1st_pc = np.argsort(X_trans[:, -1])

    # Select a subset of equally spaced images ordered by increasing
    # first PC
    selected_idx = idx_sorted_by_1st_pc[::40]

    # Plot
    fig, axes = plt.subplots(1, len(selected_idx),
                             figsize=(10, 2),
                             tight_layout=True)
    for i, idx in enumerate(selected_idx):
        ax = axes[i]
        ax.imshow(X[idx, :].reshape((112, 92)), cmap='gray')
        ax.set_title('{:.0f}'.format(X_trans[idx, -1]))
        ax.set_xticks([])
        ax.set_yticks([])
    plt.suptitle('Few samples ordered by 1st PC')

    plt.savefig('eigenface_samples_1st.png')
    plt.show()

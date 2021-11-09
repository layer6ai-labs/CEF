from pathlib import Path
from typing import Any, Callable, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as f
import PIL
from scipy import stats
from torch.utils.data import Dataset


class CelebA(Dataset):
    '''
    CelebA PyTorch dataset

    The built-in PyTorch dataset for CelebA is outdated.
    '''
    base_folder = 'celeba'

    def __init__(self, root: str, split: str = "train", transform: Optional[Callable] = None,):
        self.root = Path(root)
        self.split = split
        self.transform = transform

        celeb_path = lambda x: self.root / self.base_folder / x

        split_map = {
            'train': 0,
            'valid': 1,
            'test': 2,
            'all': None,
        }
        splits_df = pd.read_csv(celeb_path('list_eval_partition.csv'))
        self.filename = splits_df[splits_df['partition'] == split_map[split]]['image_id'].tolist()

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        img_path = (self.root / self.base_folder / 'img_align_celeba' /
                    'img_align_celeba' / self.filename[index])
        X = PIL.Image.open(img_path)

        target: Any = []
        if self.transform is not None:
            X = self.transform(X)

        return X, 0

    def __len__(self) -> int:
        return len(self.filename)


class Sphere(Dataset):
    '''
    Sample from a Gaussian distribution projected to a sphere
    '''

    def __init__(self, manifold_dim=1, ambient_dim=2, size=1000, mu=None, sigma=None):
        self.manifold_dim = manifold_dim
        self.ambient_dim = ambient_dim

        if mu is None:
            mu = np.zeros(manifold_dim + 1)
        if sigma is None:
            sigma = np.diag(np.ones(manifold_dim + 1))
        self._generate_points(mu, sigma, size)

    def _generate_points(self, mu, sigma, size):
        gaussian_points = np.random.multivariate_normal(mu, sigma, size)
        sphere_points = gaussian_points / np.linalg.norm(gaussian_points, axis=1)[:,None]
        self.points = f.pad(torch.Tensor(sphere_points),
                            pad=(0, self.ambient_dim - self.manifold_dim - 1, 0, 0))

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        return self.points[index]

    def __len__(self) -> int:
        return len(self.points)


class AffineSubspace(Dataset):
    '''
    Sample from a Gaussian mixture on an m-dimensional affine subspace in n dimensions.
    For m=2, n=3, this is a Gaussian mixture on a plane.
    '''

    def __init__(self, manifold_dim=2, ambient_dim=3, size=10000,
                 mus=None, sigmas=None, weights=None, transf=None, bias=None,
                 extra_axes=0):
        self.manifold_dim = manifold_dim
        self.ambient_dim = ambient_dim
        self.extra_axes = extra_axes

        # mus and sigmas should be lists of means
        if mus is None:
            mus = [np.zeros(manifold_dim)]
        if sigmas is None:
            sigmas = [np.eye(manifold_dim) for _ in mus]
        if weights is None:
            weights = np.ones(len(mus)) / len(mus)

        # transf should be an nxm matrix embedding the subspace into ambient space
        if transf is None:
            transf = np.concatenate(
                (np.diag(np.ones(manifold_dim)),
                 np.zeros((self.ambient_dim - self.manifold_dim, self.manifold_dim))))
        # bias should be an n-vector providing the affine component (0 is mapped to here)
        if bias is None:
            bias = np.zeros(ambient_dim)

        self.mus, self.sigmas, self.weights, self.transf, self.bias = (
            mus, sigmas, weights, transf, bias)
        self._generate_points(mus, sigmas, weights, transf, bias, size)

    def _generate_points(self, mus, sigmas, weights, transf, bias, size):
        rng = np.random.default_rng()
        components = np.argmax(rng.multinomial(1, weights, size), axis=1)
        samples = [np.random.multivariate_normal(mus[component], sigmas[component])
                   for component in components]

        # project the samples into the ambient space
        self.points = torch.Tensor([
            transf @ sample + bias for sample in samples])

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        out_shape = (self.ambient_dim,) + (1,)*self.extra_axes
        return self.points[index].reshape(out_shape)

    def __len__(self) -> int:
        return len(self.points)

    def density_grid(self, x, y, z):
        points_transp = np.stack((x, y, z)).reshape(3, -1)
        plane_points = (np.linalg.pinv(self.transf) @ points_transp).T

        component_densities = [stats.multivariate_normal.pdf(plane_points, mean=mu, cov=sigma)
                               for mu, sigma in zip(self.mus, self.sigmas)]
        p_latent = np.sum(self.weights * np.stack(component_densities, axis=1), axis=1)
        det_j = np.linalg.det(self.transf.T @ self.transf)
        densities = p_latent / np.sqrt(det_j)

        return densities.reshape(x.shape)


class PaperAffineSubspace(AffineSubspace):
    '''The exact AffineSubspace configuration used in the paper'''

    def __init__(self, size=64000):
        super().__init__(
            manifold_dim=2,
            ambient_dim=3,
            size=size,
            mus=[[1, 0], [-1/2, np.sqrt(3)/2], [-1/2, -np.sqrt(3)/2]],
            sigmas=[0.05*np.eye(2) for _ in range(3)],
            transf=np.array([[np.sqrt(1 - 0.2**2), 0],
                             [0, 1],
                             [-0.2, 0]]),
            extra_axes=2,
        )

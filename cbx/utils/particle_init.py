import numpy as np


def init_particles(shape=(1, 1, 1), tensor=None, rank=None, x_min=-1.0, x_max=1.0, delta=1.0, method="uniform"):
    r"""Initialize particles

    Parameters
    ----------
    shape : tuple of ints, optional
        Shape of the particles array. The default is (1, 1, 1).
    tensor : numpy.ndarray, optional
        Tensor for initialization. Default is None.
    rank : int, optional
        Rank for factorization. Default is None.
    x_min : float, optional
        Lower bound for the uniform distribution. The default is -1.0.
    x_max : float, optional
        Upper bound for the uniform distribution. The default is 1.0.
    delta : float, optional
        Standard deviation for the normal distribution. The default is 1.0.
    method : str, optional
        Method for initializing the particles. The default is "uniform".
        Possible values: "uniform", "normal".

    Returns
    -------
    list of dict
        A list where each entry is a dictionary containing NumPy arrays A, B, and C.
    """

    if tensor is not None and rank is not None:
        I, J, K = tensor.shape
        M, N, d = shape
        particles = []

        for i in range(N):
            if method == "uniform":
                A = np.random.uniform(x_min, x_max, (I, rank))
                B = np.random.uniform(x_min, x_max, (J, rank))
                C = np.random.uniform(x_min, x_max, (K, rank))
            elif method == "normal":
                A = np.random.normal(0, delta, (I, rank))
                B = np.random.normal(0, delta, (J, rank))
                C = np.random.normal(0, delta, (K, rank))
            else:
                raise ValueError('Error: Unknown method for init_particles specified!')

            # Ensure A, B, and C are numpy arrays (they already are, but you can check)
            A, B, C = np.array(A), np.array(B), np.array(C)

            # Create a dictionary for each particle
            particle = {'A': A, 'B': B, 'C': C}
            particles.append(particle)

        return particles

    elif tensor is None and rank is not None:
        raise ValueError('Input error: Define the tensor!')
    elif tensor is not None and rank is None:
        raise ValueError('Input error: Define the rank!')
    else:
        if method == "uniform":
            x = np.random.uniform(x_min, x_max, shape)
        elif method == "normal":
            if len(shape) == 3:
                M, N, d = shape
            elif len(shape) == 2:
                N, d = shape
                M = 1
            else:
                raise RuntimeError('Normal initialization only supported for 2D or 3D shapes!')

            # Generate multivariate normal particles
            x = np.random.multivariate_normal(np.zeros((d,)), delta * np.eye(d), (M, N))
        else:
            raise RuntimeError('Unknown method for init_particles specified!')

        return x

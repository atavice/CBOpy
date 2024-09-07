import numpy as np

def init_particles(shape=(1,1,1), x_min=-1.0, x_max = 1.0, delta=1.0, method="uniform"):
    r"""Initialize particles
    
    Parameters
    ----------
    N : int, optional
        Number of particles. The default is 100.
    d : int, optional
        Dimension of the particles. The default is 2.
    x_min : float, optional
        Lower bound for the uniform distribution. The default is 0.0.
    x_max : float, optional
        Upper bound for the uniform distribution. The default is 1.0.
    delta : float, optional
        Standard deviation for the normal distribution. The default is 1.0.
    method : str, optional
        Method for initializing the particles. The default is "uniform".
        Possible values: "uniform", "normal"
    
    Returns
    -------
    x : numpy.ndarray
        Array of particles of shape (N, d)
    """


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
        
        x = np.random.multivariate_normal(np.zeros((d,)), delta * np.eye(d), (M, N))
    else:
        raise RuntimeError('Unknown method for init_particles specified!')
        
    return x





def init_particles_as_multiple_matrices(N=100, M=50, I=10, J=10, K=10, rank=5, x_min=-1.0, x_max=1.0, delta=1.0,
                                        method="uniform"):
    r"""Initialize particles where each particle consists of 3 matrices (A, B, C) for tensor decomposition.

    Parameters
    ----------
    num_particles : int, optional
        Number of particles (default is 100).
    I : int
        First dimension of the tensor to be decomposed.
    J : int
        Second dimension of the tensor to be decomposed.
    K : int
        Third dimension of the tensor to be decomposed.
    rank : int
        Rank of the decomposition.
    x_min : float, optional
        Lower bound for the uniform distribution (default is -1.0).
    x_max : float, optional
        Upper bound for the uniform distribution (default is 1.0).
    delta : float, optional
        Standard deviation for the normal distribution (default is 1.0).
    method : str, optional
        Method for initializing the particles. Possible values are "uniform" or "normal".

    Returns
    -------
    particles : list of tuples
        A list where each particle consists of three matrices (A, B, C).
        A is of shape (I, rank), B is of shape (J, rank), C is of shape (K, rank).
    """

    # Initialize an empty array to hold the particles, where each particle has 3 matrices
    particles = np.zeros((N, 3, max(I, J, K), rank))

    for idx in range(N):
        if method == "uniform":
            A = np.random.uniform(x_min, x_max, (I, rank))
            B = np.random.uniform(x_min, x_max, (J, rank))
            C = np.random.uniform(x_min, x_max, (K, rank))
        elif method == "normal":
            A = np.random.normal(0, delta, (I, rank))
            B = np.random.normal(0, delta, (J, rank))
            C = np.random.normal(0, delta, (K, rank))
        else:
            raise RuntimeError('Unknown method for init_particles_as_multiple_matrices specified!')

        # Stack A, B, and C along the first axis (dim=0) to fit into a single particle
        # Pad the matrices A, B, C to the size (max(I, J, K), rank)
        particles[idx, 0, :I, :] = A  # Assign matrix A
        particles[idx, 1, :J, :] = B  # Assign matrix B
        particles[idx, 2, :K, :] = C  # Assign matrix C

    return particles

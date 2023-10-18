"""
Various helper functions.

Note:
    The functions are in alphabetical order.
"""
import numpy as np


def batch_kron(a: np.ndarray, b: np.ndarray):
    """
    Batch Kronecker product: np.kron(a[i, :, :], b[i, :, :]) for all i.
    """
    return np.einsum('bik,bjl->bijkl', a, b).reshape(a.shape[0], a.shape[1] * b.shape[1], a.shape[2] * b.shape[2])


def cov2cov(matrices: np.ndarray):
    """
    Convert the real representations of complex covariance matrices back to
    complex representations.
    """
    if matrices.ndim == 2:
        # the case of diagonal matrices
        n_mats, n_diag = matrices.shape
        mats = np.zeros([n_mats, n_diag, n_diag])
        for i in range(n_mats):
            mats[i, :, :] = np.diag(matrices[i, :])
    else:
        mats = matrices

    n_mats, rows, columns = mats.shape
    row_half = rows // 2
    column_half = columns // 2
    covs = np.zeros((n_mats, row_half, column_half), dtype=complex)
    for c in range(n_mats):
        upper_left_block = mats[c, :row_half, :column_half]
        upper_right_block = mats[c, :row_half, column_half:]
        lower_left_block = mats[c, row_half:, :column_half]
        lower_right_block = mats[c, row_half:, column_half:]
        covs[c, :, :] = upper_left_block + lower_right_block + 1j * (lower_left_block - upper_right_block)
    return covs


def cplx2real(vec: np.ndarray, axis=0):
    """
    Concatenate real and imaginary parts of vec along axis=axis.
    """
    return np.concatenate([vec.real, vec.imag], axis=axis)


def crandn(*arg, rng=np.random.default_rng()):
    return np.sqrt(0.5) * (rng.standard_normal(arg) + 1j * rng.standard_normal(arg))


def kron_approx_sep_ls(
    mats_A: np.ndarray,
    init_C: np.ndarray,
    rows_B: int,
    cols_B: int,
    iterations: int = 10
):
    """
    Approximate a matrix in terms of a Kronecker product of two matrices. The
    array init_C is an initialization for the matrix C and will be overwritten.
    If it is structured, for example, if it is positive definite, then the
    returned matrices B and C will have the same structure. Section 5 of the
    source explains what kind of structure can be used.

    Note:
        This corresponds to Framework 2 in Section 4 in the source.

    Source:
        "Approximation with Kronecker Products"
        by Van Loan, Pitsianis
        https://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.42.1924&rep=rep1&type=pdf
    """
    if mats_A.ndim == 2:
        mats_A = np.expand_dims(mats_A, 0)
        init_C = np.expand_dims(init_C, 0)

    mats_B = np.zeros((mats_A.shape[0], rows_B, cols_B), dtype=mats_A.dtype)
    mats_C = init_C
    rows_C, cols_C = mats_C.shape[-2:]

    # Extract the blocks A_ij: Split the 3d array of shape (n, rows_B * rows_C, cols_B * cols_C) into a 5d array of
    # shape (n, rows_B, cols_B, rows_C, cols_C) where [:, i, j, :, :] corresponds to the blocks A_ij in equation (2).
    blocks_A = np.zeros((mats_A.shape[0], rows_B, cols_B, rows_C, cols_C), dtype=mats_A.dtype)
    for i, block_row in enumerate(np.split(mats_A, rows_B, axis=-2)):
        for j, block in enumerate(np.split(block_row, cols_B, axis=-1)):
            blocks_A[:, i, j, :, :] = block

    # Extract the blocks Ahat_ij: Split the 3d array of shape (n, rows_B * rows_C, cols_B * cols_C) into a 5d array of
    # shape (n, rows_C, cols_C, rows_B, cols_B) where [:, i, j, :, :] corresponds to the blocks Ahat_ij in equation (4).
    blocks_Ahat = np.zeros((mats_A.shape[0], rows_C, cols_C, rows_B, cols_B), dtype=mats_A.dtype)
    for i in range(rows_C):
        for j in range(cols_C):
            blocks_Ahat[:, i, j, :, :] = \
                mats_A[:, i: i + (rows_B - 1) * rows_C + 1: rows_C, j: j + (cols_B - 1) * cols_C + 1: cols_C]

    beta_or_gamma = np.zeros((mats_A.shape[0], 1, 1), dtype=mats_A.dtype)

    def project(blocks_ij, b_or_c, c_or_b_out):
        """
        For every block A_ij (or Ahat_ij), compute equation (8) (or (9)) in Theorem 4.1.
        """
        np.einsum('ijklm,ilm->ijk', blocks_ij, b_or_c, out=c_or_b_out)
        np.einsum('ijk,ijk->i', b_or_c, b_or_c, out=beta_or_gamma[:, 0, 0])
        c_or_b_out /= beta_or_gamma

    for _ in range(iterations):
        project(blocks_A, mats_C, mats_B)
        project(blocks_Ahat, mats_B, mats_C)

    return mats_B, mats_C


def kron_approx_svd(mats_A: np.ndarray, rows_B: int, cols_B: int, rows_C: int, cols_C: int):
    r"""
    Approximate a matrix in terms of a Kronecker product of two matrices.

    Note:
        Given a matrix A, find matrices B and C of shapes (rows_B, cols_B) and
        (rows_C, cols_C) such that \| A - B \otimes C \|_F is minimized.
        If A is structured, e.g., symmetric or positive definite, the function
        kron_approx_sep_ls can be used.

    Source:
        "Approximation with Kronecker Products"
        by Van Loan, Pitsianis
        https://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.42.1924&rep=rep1&type=pdf
    """
    if mats_A.ndim == 2:
        mats_A = np.expand_dims(mats_A, 0)
    n, rows_A, cols_A = mats_A.shape
    if rows_B * rows_C != rows_A:
        raise ValueError(f'rows_B*rows_C = {rows_A} is required, but rows_B*rows_C = {rows_B*rows_C}')
    if cols_B * cols_C != cols_A:
        raise ValueError(f'cols_B*cols_C = {cols_A} is required, but cols_B*cols_C = {cols_B*cols_C}')

    def block(A, i, j):
        """
        Extract block A_ij from the matrix A. Every block A_ij has shape
        (rows_C, cols_C) and there are rows_B * cols_B such blocks.
        """
        return A[i * rows_C: (i + 1) * rows_C, j * cols_C: (j + 1) * cols_C]

    out_B = np.zeros((n, rows_B, cols_B), dtype=mats_A.dtype)
    out_C = np.zeros((n, rows_C, cols_C), dtype=mats_A.dtype)
    for ni in range(n):
        # the following implements equation (5)
        rearranged_A = np.zeros((rows_B * cols_B, rows_C * cols_C), dtype=mats_A.dtype)
        for j in range(cols_B):
            # extract the matrix A_j
            rearranged_A[j * rows_B: (j + 1) * rows_B, :] = np.concatenate(
                [block(mats_A[ni, :, :], i, j).flatten('F')[np.newaxis, :] for i in range(rows_B)],
                axis=0
            )
        u, s, vh = np.linalg.svd(rearranged_A)
        out_B[ni, :, :] = u[:, 0].reshape(rows_B, cols_B, order='F') * s[0]
        out_C[ni, :, :] = vh[0, :].reshape(rows_C, cols_C, order='F')
    return out_B, out_C


def kron_real(mats1: np.ndarray, mats2: np.ndarray):
    """
    Assuming mats1 and mats2 are real representations of complex covariance
    matrices, compute the real representation of the Kronecker product of the
    complex covariance matrices.
    """
    if mats1.ndim != mats2.ndim:
        raise ValueError(
            'The two arrays need to have the same number of dimensions, '
            f'but we have mats1.ndim = {mats1.ndim} and mats2.ndim = {mats2.ndim}.'
        )
    if mats1.ndim == 2:
        mats1 = np.expand_dims(mats1, 0)
        mats2 = np.expand_dims(mats2, 0)

    n = mats1.shape[0]
    rows1, cols1 = mats1.shape[-2:]
    rows2, cols2 = mats2.shape[-2:]
    row_half1 = rows1 // 2
    column_half1 = cols1 // 2
    row_half2 = rows2 // 2
    column_half2 = cols2 // 2
    rows3 = 2 * row_half1 * row_half2
    cols3 = 2 * column_half1 * column_half2

    out_kron_prod = np.zeros((n, rows3, cols3))
    for i in range(n):
        A1 = mats1[i, :row_half1, :column_half1]
        B1 = mats1[i, :row_half1, column_half1:]
        C1 = mats1[i, row_half1:, :column_half1]
        D1 = mats1[i, row_half1:, column_half1:]

        A2 = mats2[i, :row_half2, :column_half2]
        B2 = mats2[i, :row_half2, column_half2:]
        C2 = mats2[i, row_half2:, :column_half2]
        D2 = mats2[i, row_half2:, column_half2:]

        A = np.kron(A1 + D1, A2 + D2)
        D = -np.kron(C1 - B1, C2 - B2)
        B = -np.kron(A1 + D1, C2 - B2)
        C = np.kron(C1 - B1, A2 + D2)

        A = 0.5 * (A + D)
        D = A
        B = 0.5 * (B - C)
        C = -B

        out_kron_prod[i, :, :] = np.concatenate(
            (np.concatenate((A, B), axis=1), np.concatenate((C, D), axis=1)),
            axis=0
        )
    return np.squeeze(out_kron_prod)


def mat2bsc(mat: np.ndarray):
    """
    Arrange the real and imaginary parts of a complex matrix mat in block-
    skew-circulant form.

    Source:
        See https://ieeexplore.ieee.org/document/7018089.
    """
    upper_half = np.concatenate((mat.real, -mat.imag), axis=-1)
    lower_half = np.concatenate((mat.imag, mat.real), axis=-1)
    return np.concatenate((upper_half, lower_half), axis=-2)


def real2real(mats):
    re = np.real(mats)
    im = np.imag(mats)
    rows = mats.shape[1]
    cols = mats.shape[2]
    out = np.zeros([mats.shape[0], 2*rows, 2*cols])
    out[:, :rows, :cols] = 0.5*re
    out[:, rows:, cols:] = 0.5*re
    return out


def imag2imag(mats):
    im = np.real(mats)
    rows = mats.shape[1]
    cols = mats.shape[2]
    out = np.zeros([mats.shape[0], 2*rows, 2*cols])
    #out[:, :rows, :cols] = 0.5*re
    for i in range(mats.shape[0]):
        out[i, :rows, cols:] = 0.5*im[i,:,:].T
        out[i, rows:, :cols] = 0.5*im[i,:,:]
    #out[:, rows:, cols:] = 0.5*re
    return out


def nmse(actual: np.ndarray, desired: np.ndarray):
    """
    Mean squared error between actual and desired divided by the total number
    of elements.
    """
    mse = 0
    for i in range(actual.shape[0]):
        mse += np.linalg.norm(actual - desired) ** 2 / np.linalg.norm(desired) ** 2
    return mse / actual.shape[0]
    #return np.sum(np.abs(actual - desired) ** 2) / desired.size


def real2cplx(vec: np.ndarray, axis=0):
    """
    Assume vec consists of concatenated real and imaginary parts. Return the
    corresponding complex vector. Split along axis=axis.
    """
    re, im = np.split(vec, 2, axis=axis)
    return re + 1j * im


def sec2hours(seconds: float):
    """"
    Convert a number of seconds to a string h:mm:ss.
    """
    # hours
    h = seconds // 3600
    # remaining seconds
    r = seconds % 3600
    return '{:.0f}:{:02.0f}:{:02.0f}'.format(h, r // 60, r % 60)

def check_random_state(seed):
    import numbers
    """Turn seed into a np.random.RandomState instance

    Parameters
    ----------
    seed : None, int or instance of RandomState
        If seed is None, return the RandomState singleton used by np.random.
        If seed is an int, return a new RandomState instance seeded with seed.
        If seed is already a RandomState instance, return it.
        Otherwise raise ValueError.
    """
    if seed is None or seed is np.random:
        return np.random.mtrand._rand
    if isinstance(seed, numbers.Integral):
        return np.random.RandomState(seed)
    if isinstance(seed, np.random.RandomState):
        return seed
    raise ValueError('%r cannot be used to seed a numpy.random.RandomState'
                     ' instance' % seed)

def print_dict(dict: dict, entries_per_row: int=1):
    """Print the keys and values of dictionary dict."""
    if entries_per_row < 1:
        raise ValueError(f'The number of entries per row needs to be >= 1 but is {entries_per_row}')
    for c, (key, value) in enumerate(dict.items()):
        if c % entries_per_row == 0 and c > 0:
            print()
        else:
            c > 0 and print(' | ', end='')
        print('{}: {}'.format(key, value), end='')
    print()


def dft_matrix(n_antennas, n_grid):
    grid = np.linspace(-1, 1, n_grid + 1)[:n_grid]

    d = 1 / np.sqrt(n_antennas) * np.exp(1j * np.pi * np.outer(np.arange(n_antennas), grid.conj().T))
    return d

#def dist_fac(n_bits: int):
#    """"Compute distortion factor for an arbitrary number of bits."""
#    if n_bits == 1:
#        raise ValueError('The distortion factor for 1-bit is known in closed form.')
#    else:
#        return n_bits * 2**(-2*n_bits)
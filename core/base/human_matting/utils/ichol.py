
import numpy as np
import scipy.sparse
from numba import njit


@njit("i8(i8, f8[:], i8[:], i8[:], f8[:], i8[:], i8[:], f8, f8, i8)", cache=True, nogil=True)
def _ichol(n, Av, Ar, Ap, Lv, Lr, Lp, discard_threshold, shift, max_nnz):
    nnz = 0
    c_n = 0
    s = np.zeros(n, np.int64)  # Next non-zero row index i in column j of L
    t = np.zeros(n, np.int64)  # First subdiagonal index i in column j of A
    l = np.zeros(n, np.int64) - 1  # Linked list of non-zero columns in row k of L
    a = np.zeros(n, np.float64)  # Values of column j
    b = np.zeros(
        n, np.bool8
    )  # b[i] indicates if the i-th element of column j is non-zero
    c = np.empty(n, np.int64)  # Row indices of non-zero elements in column j
    d = np.full(n, shift, np.float64)  # Diagonal elements of A
    for j in range(n):
        for idx in range(Ap[j], Ap[j + 1]):
            i = Ar[idx]
            if i == j:
                d[j] += Av[idx]
                t[j] = idx + 1
    for j in range(n):  # For each column j
        for idx in range(t[j], Ap[j + 1]):  # For each L_ij
            i = Ar[idx]
            L_ij = Av[idx]
            if L_ij != 0.0 and i > j:
                a[i] += L_ij  # Assign non-zero value to L_ij in sparse column
                if not b[i]:
                    b[i] = True  # Mark it as non-zero
                    c[c_n] = i  # Remember index for later deletion
                    c_n += 1
        k = l[j]  # Find index k of column with non-zero element in row j
        while k != -1:  # For each column of that type
            k0 = s[k]  # Start index of non-zero elements in column k
            k1 = Lp[k + 1]  # End index
            k2 = l[k]  # Remember next column index before it is overwritten
            L_jk = Lv[k0]  # Value of non-zero element at start of column
            k0 += 1  # Advance to next non-zero element in column
            if k0 < k1:  # If there is a next non-zero element
                s[k] = k0  # Advance start index in column k to next non-zero element
                i = Lr[k0]  # Row index of next non-zero element in column k
                l[k] = l[i]  # Remember old list i index in list k
                l[i] = k  # Insert index of non-zero element into list i
                for idx in range(k0, k1):  # For each non-zero L_ik in column k
                    i = Lr[idx]
                    L_ik = Lv[idx]
                    a[i] -= L_ik * L_jk  # Update element L_ij in sparse column
                    if not b[i]:  # Check if sparse column element was zero
                        b[i] = True  # Mark as non-zero in sparse column
                        c[c_n] = i  # Remember index for later deletion
                        c_n += 1
            k = k2  # Advance to next column k
        if d[j] <= 0.0:
            return -1
        if nnz + 1 + c_n > max_nnz:
            return -2
        d[j] = np.sqrt(d[j])  # Update diagonal element L_ii
        Lv[nnz] = d[j]  # Add diagonal element L_ii to L
        Lr[nnz] = j  # Add row index of L_ii to L
        nnz += 1
        s[j] = nnz  # Set first non-zero index of column j
        for i in np.sort(
            c[:c_n]
        ):  # Sort row indices of column j for correct insertion order into L
            L_ij = a[i] / d[j]  # Get non-zero element from sparse column j
            d[i] -= L_ij * L_ij  # Update diagonal element L_ii
            if abs(L_ij) > discard_threshold:  # If element is sufficiently non-zero
                Lv[nnz] = L_ij  # Add element L_ij to L
                Lr[nnz] = i  # Add row index of L_ij
                nnz += 1
            a[i] = 0.0  # Set element i in column j to zero
            b[i] = False  # Mark element as zero
        c_n = 0  # Discard row indices of non-zero elements in column j.
        Lp[j + 1] = nnz  # Update count of non-zero elements up to column j
        if Lp[j] + 1 < Lp[j + 1]:  # If column j has a non-zero element below diagonal
            i = Lr[Lp[j] + 1]  # Row index of first off-diagonal non-zero element
            l[j] = l[i]  # Remember old list i index in list j
            l[i] = j  # Insert index of non-zero element into list i
    return nnz


@njit("void(f8[:], i8[:], i8[:], f8[:], i8)", cache=True, nogil=True)
def _backsub_L_csc_inplace(data, indices, indptr, x, n):
    for j in range(n):
        k = indptr[j]
        L_jj = data[k]
        temp = x[j] / L_jj

        x[j] = temp

        for k in range(indptr[j] + 1, indptr[j + 1]):
            i = indices[k]
            L_ij = data[k]

            x[i] -= L_ij * temp


@njit("void(f8[:], i8[:], i8[:], f8[:], i8)", cache=True, nogil=True)
def _backsub_LT_csc_inplace(data, indices, indptr, x, n):
    for i in range(n - 1, -1, -1):
        s = x[i]

        for k in range(indptr[i] + 1, indptr[i + 1]):
            j = indices[k]
            L_ji = data[k]
            s -= L_ji * x[j]

        k = indptr[i]
        L_ii = data[k]

        x[i] = s / L_ii


class CholeskyDecomposition(object):


    def __init__(self, Ltuple):
        self.Ltuple = Ltuple

    @property
    def L(self):

        Lv, Lr, Lp = self.Ltuple
        n = len(Lp) - 1
        return scipy.sparse.csc_matrix(self.Ltuple, (n, n))

    def __call__(self, b):
        Lv, Lr, Lp = self.Ltuple
        n = len(b)
        x = b.copy()
        _backsub_L_csc_inplace(Lv, Lr, Lp, x, n)
        _backsub_LT_csc_inplace(Lv, Lr, Lp, x, n)
        return x


def ichol(
    A,
    discard_threshold=1e-4,
    shifts=[0.0, 1e-4, 1e-3, 1e-2, 0.1, 0.5, 1.0, 10.0, 100, 1e3, 1e4, 1e5],
    max_nnz=int(4e9 / 16),
):


    if isinstance(A, scipy.sparse.csr_matrix):
        A = A.T

    if not isinstance(A, scipy.sparse.csc_matrix):
        raise ValueError("Matrix A must be a scipy.sparse.csc_matrix")

    if not A.has_canonical_format:
        A.sum_duplicates()

    m, n = A.shape

    assert m == n

    Lv = np.empty(max_nnz, dtype=np.float64)  # Values of non-zero elements of L
    Lr = np.empty(max_nnz, dtype=np.int64)  # Row indices of non-zero elements of L
    Lp = np.zeros(
        n + 1, dtype=np.int64
    )  # Start(Lp[i]) and end(Lp[i+1]) index of L[:, i] in Lv

    for shift in shifts:
        nnz = _ichol(
            n,
            A.data,
            A.indices.astype(np.int64),
            A.indptr.astype(np.int64),
            Lv,
            Lr,
            Lp,
            discard_threshold,
            shift,
            max_nnz,
        )

        if nnz >= 0:
            break

        if nnz == -1:
            print("PERFORMANCE WARNING:")
            print(
                "Thresholded incomplete Cholesky decomposition failed due to insufficient positive-definiteness of matrix A with parameters:"
            )
            print("    discard_threshold = %e" % discard_threshold)
            print("    shift = %e" % shift)
            print("Try decreasing discard_threshold or start with a larger shift")
            print("")

        if nnz == -2:
            raise ValueError(
                "Thresholded incomplete Cholesky decomposition failed because more than max_nnz non-zero elements were created. Try increasing max_nnz or discard_threshold."
            )

    if nnz < 0:
        raise ValueError(
            "Thresholded incomplete Cholesky decomposition failed due to insufficient positive-definiteness of matrix A and diagonal shifts did not help."
        )

    Lv = Lv[:nnz]
    Lr = Lr[:nnz]

    return CholeskyDecomposition((Lv, Lr, Lp))

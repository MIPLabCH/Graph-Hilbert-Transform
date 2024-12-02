"""
Copyright © 2024 Chun Hei Michael Chan, MIPLab EPFL
"""

from src.utils import *
from src.metrics import *


wrap_pi = lambda x: (x + np.pi) % (2 * np.pi) - np.pi

def normalize_adjacency(A: np.ndarray, tnorm="left"):
    """
    Normalize the adjacency matrix by in-degrees / out-degrees / symmetric

    Parameters
    ----------
    A : np.ndarray
        The adjacency matrix to normalize 
    tnorm : str
        The normalization method. Can be "right", "left", or "symmetric".

    Returns
    -------
    normA : np.ndarray
        The normalized adjacency matrix
    """
    if tnorm == "right":
        outdegrees = np.sum(A, axis=0)
        factors_in = np.diag(np.divide(1, outdegrees, where=np.abs(outdegrees) > 1e-10))
        normA = A @ factors_in

    if tnorm == "left":
        indegrees = np.sum(A, axis=1)
        factors_out = np.diag(
            np.divide(1, indegrees, where=np.abs(indegrees) > 1e-10)
        )
        normA = factors_out @ A

    if tnorm == "symmetric":
        indegrees = np.sum(A, axis=1)
        outdegrees = np.sum(A, axis=0)

        if (np.sum(indegrees < 0) + np.sum(outdegrees < 0)) > 0:
            print("Negative Degrees")
            return

        indegrees = np.sqrt(indegrees)
        outdegrees = np.sqrt(outdegrees)

        factors_in = np.diag(np.divide(1, indegrees, where=np.abs(indegrees) > 1e-10))
        factors_out = np.diag(
            np.divide(1, outdegrees, where=np.abs(outdegrees) > 1e-10)
        )
        normA = factors_out @ A @ factors_in

    return normA    

def compute_directed_laplacian(A:np.ndarray, in_degree:bool=True):
    """
    Compute the directed Laplacian matrix for a given adjacency matrix A. 

    The directed Laplacian is defined as L = D - A, where D is a diagonal matrix containing the in-degree of each node, and A is the adjacency matrix.

    Parameters
    ----------
    A : ndarray
        Adjacency matrix
    in_degree : bool
        Flag to compute in-degree or out-degree

    Returns
    -------
    L : ndarray
        Directed Laplacian matrix
    """

    if in_degree:
        deg = A.sum(axis=1).astype(float)
    else:
        deg = A.sum(axis=0).astype(float)
    ret = np.diag(deg) - A.astype(float)

    return ret


def compute_basis(L:np.ndarray, method:str="eig", tol:float=1e-13, verbose:bool=True, gso:str='laplacian'):
    """
    Computes basis for transform matrix supporting different methods:

    - eig: Eigendecomposition
    - Removed all other methods currently

    Parameters
    ----------
    L : numpy array
        Input matrix
    method : str
        Method to use for decomposition ('eig', 'jord', 'ortho')
    tol : float
        Tolerance for treating small values as zero 
    verbose : bool
        Whether to print chosen method  

    Returns
    -------
    U : numpy array
        Left transform matrix
    V : numpy array 
        Diagonal transform matrix
    """

    if verbose:
        print(f"Method chosen is: {method}")
    # Prior to all be careful to remove all the numerical potential rounding error from float
    # i.e set 1e-12 -> 0
    L = L * (np.abs(L) > tol)
    if method == "eig":
        eigval, eigvect = np.linalg.eig(L)
        # Case of perfect cycle, in this case we don't want to do reordering
        if np.all(np.abs(eigval - 1) < 1e-10):
            V = eigval
            U = eigvect

        else:
            if gso == 'laplacian':
                frequencies = np.abs(eigval)
            elif gso == 'adj':
                lbd_max = np.abs(np.linalg.eigvals(L)).max()
                frequencies = np.array([TV2(eigvect[:,k], L, norm='L1', lbd_max=lbd_max) for k in range(len(L))])

            V = eigval[np.argsort(frequencies)]
            U = eigvect[:, np.argsort(frequencies)]

    else:
        print("Method not supported ...")

    return U, V

def polar_decomposition(A):
    """
    Compute the polar decomposition of a directed graph A
    
    Parameters
    ----------
    A : np.ndarray
        The directed graph adjacency matrix
    
    Returns
    -------
    Q : np.ndarray
        The orthogonal matrix
    F : np.ndarray
        The positive semi-definite matrix
    P : np.ndarray
        The positive semi-definite matrix
    """

    U, S, vh = np.linalg.svd(A)
    F = vh.T @ np.diag(S) @ vh
    P = U @ np.diag(S) @ U.T
    Q = U @ vh
    return Q, F, P


def hermitian(A:np.ndarray):
    """
    Compute the Hermitian (conjugate transpose) of a matrix.
    
    Parameters
    ----------
    A : numpy array
        Input matrix
    
    Returns
    -------
    ret : numpy array
        Hermitian of A
    """
    ret = np.conjugate(A).T
    return ret

def conjugate_frequency(idx:int, V:np.ndarray):
    """
    Return conjugate frequency of the idx-th harmonic
    """
    cf = np.where(np.abs(hermitian(V[idx]) - V) < 1e-8)[0]
    return cf[0]


def eigvalues_pairs(V:np.ndarray):
    """
    Compute a list of groups (pairs or singletons) of complex conjugate eigenvalues.

    This function takes a numpy array `V` representing the eigenvectors of a graph Laplacian,
    and returns a list of groups of indices where the corresponding eigenvalues are either
    complex conjugate pairs or singletons (real eigenvalues).

    Parameters
    ----------
    V : numpy.ndarray
        The eigenvectors of the graph Laplacian.

    Returns
    -------
    tasks: numpy.ndarray
        A list of groups of indices where the corresponding eigenvalues are either
        complex conjugate pairs or singletons.
    """

    indexes = np.arange(V.shape[0])
    assigned = []
    tasks = []
    for idx in indexes:
        if idx in assigned:
            continue
        gp = np.where(np.abs(V[idx].real - V.real) < 1e-8)[0]
        tasks.append(gp)
        assigned += list(gp)
    return tasks


#######################
#### Graph Fourier ####
#######################
def GFT(signal:np.ndarray, U:np.ndarray, herm:bool=False, Uinv:Optional[np.ndarray]=None):
    """
    Compute the graph Fourier transform of a signal.

    Parameters
    ----------
    signal : numpy array
        Input signal defined on graph vertices 

    U : numpy array
        Graph Fourier basis (eigenvectors of graph Laplacian)

    herm : bool
        If True, use Hermitian transpose of U instead of U 

    Uinv : numpy array
        Inverse of U to avoid matrix inversion if precomputed
        Default is None to compute the inverse on the fly

    Returns
    -------
    ret : numpy array 
        Graph Fourier transform of signal 
    """

    if herm == True:
        hermitian = np.matrix.conjugate(U).T
        ret = hermitian @ signal
    else:
        if Uinv is None:
            ret = np.linalg.inv(U) @ signal
        else:
            ret = Uinv @ signal
    return ret

def inverseGFT(coef: np.ndarray, U: np.ndarray):
    """
    Compute inverse graph Fourier transform of a signal.

    Given the graph Fourier coefficients `coef` and the graph 
    Fourier basis `U`, this function computes the inverse transform 
    to reconstruct the original signal defined on the vertices.

    Parameters
    ----------
    coef : numpy array
        Graph Fourier coefficients of signal

    U : numpy array
        Graph Fourier basis (eigenvectors of graph Laplacian)

    Returns
    -------  
    ret : numpy array
        Reconstructed signal defined on graph vertices
    """

    ret = U @ coef
    return ret

def cn_stabiliser(mat: np.ndarray):
    """
    Stabilise the condition number of the graph Fourier basis by adding an 
    undirected edge between two nodes. Computes the undirected edge which 
    minimizes the condition number when added.

    Parameters
    ----------
    mat : numpy array
        Adjacency matrix of directed graph 

    Returns
    -------
    curpair : tuple
        Indices of nodes to add undirected edge between
    curscore : float  
        Condition number after adding the edge

    """
    r, c = mat.shape

    # Initial condition number
    L = compute_directed_laplacian(mat)
    eigval, eigvect = np.linalg.eig(L)
    U = eigvect[:, np.argsort(np.abs(eigval))]
    cn = np.linalg.cond(U)

    curpair = None
    curscore = cn
    for row in range(r):
        for col in range(c):
            if row == col:
                continue
            tmp = deepcopy(mat)
            tmp[row, col] = 1.0
            tmp[col, row] = 1.0

            L = compute_directed_laplacian(tmp)
            eigval, eigvect = np.linalg.eig(L)
            U = eigvect[:, np.argsort(np.abs(eigval))]
            cn = np.linalg.cond(U)
            if curscore >= cn:
                curscore = cn
                curpair = (row, col)

    if curpair is None:
        print("No undirected edge reduces CN")

    return curpair, curscore


###################################
##### Graph Phase Operations ######
###################################

def apply_phaseshift(phase:np.ndarray, signal:np.ndarray, 
                     U:np.ndarray, V:np.ndarray, Uinv:np.ndarray):
    """
    Apply a phase shift to the frequency domain representation of a signal.
    Generalization of Hilbert Transform with general phase shift in GFT domain
    Parameters
    ----------
    phase : np.ndarray
        The phase shift to apply.
    signal : np.ndarray
        The input signal.
    U : np.ndarray
        The Fourier transform matrix.
    V : np.ndarray
        The eigenvectors of the graph Laplacian.
    Uinv : np.ndarray
        The inverse of the Fourier transform matrix.

    Returns
    -------
    np.ndarray
        The signal with the phase shift applied.
    """
    coefs = GFT(signal, U, Uinv=Uinv)

    cond = V.imag
    filter_p = (np.array((cond < 0), dtype=float) * np.exp(1j * phase) + np.array((cond > 0), dtype=float) * np.exp(-1j * phase))
    filter_P = np.diag(filter_p)
    return inverseGFT(filter_P @ coefs, U)
    
def hilbert_filter(V:np.ndarray):
    """
    Compute filter for Hilbert Transform i.e in GFT domain apply rotation 90

    Parameters
    ----------
    V : numpy.ndarray
        Input graph signal in vertex domain 

    Returns
    -------
    filter_H : numpy.ndarray
        Diagonal filter matrix for Hilbert transform in GFT domain
    """
    cond = V.imag
    filter_h = (np.array((cond < 0), dtype=float) * 1j + np.array((cond > 0), dtype=float) * -1j)
    filter_H = np.diag(filter_h)
    return filter_H

def hilbert_transform(signal:np.ndarray, U:np.ndarray, V:np.ndarray, Uinv:Optional[np.ndarray]=None):
    """
    Compute the Hilbert transform of the input signal in the graph 
    Fourier transform (GFT) domain.

    The signal is first transformed to the GFT domain using the 
    eigenvector matrices U and V. The Hilbert transform filter is 
    applied in the GFT domain. The result is then inverse transformed 
    back to the vertex domain.

    Parameters
    ----------
    signal : numpy.ndarray
        Input signal in the vertex domain

    U : numpy.ndarray
        Eigenvector matrix for graph Fourier transform

    V : numpy.ndarray 
        Eigenvalues for graph Fourier transform

    Uinv : numpy.ndarray, optional
        Inverse eigenvector matrix for inverse graph Fourier transform

    Returns
    -------
    ret : numpy.ndarray
        Hilbert transformed signal in the vertex domain
    """

    xhat = GFT(signal, U, herm=False, Uinv=Uinv)
    filtered = hilbert_filter(V) @ xhat
    ret = inverseGFT(filtered, U)
    return ret

def analytical_signal(signal:np.ndarray, U:np.ndarray, V:np.ndarray, Uinv:Optional[np.ndarray]=None):
    """
    Compute the analytical signal.

    The analytical signal is computed by taking the Hilbert transform of the 
    input signal, and adding it with a 90 degree phase shift to the original 
    signal.

    Parameters
    ----------
    signal : numpy array
        Input signal 
    U : numpy array
        Graph Fourier transform eigenvector matrix
    V : numpy array
        Graph Fourier transform eigenvalues
    Uinv : numpy array, optional
        Inverse graph Fourier transform eigenvector matrix

    Returns
    -------
    ret : numpy array
        Analytical signal
    """

    xh = hilbert_transform(signal, U, V, Uinv=Uinv)
    ret = signal + 1j * xh

    return ret

def inverse_analytical(instant_amp:np.ndarray, instant_phase:np.ndarray, 
                       U:np.ndarray, V:np.ndarray, Uinv:np.ndarray):
    """
    Compute the inverse analytical transform of the given instantaneous
    amplitudes and phases.

    Parameters
    ----------

    Returns
    -------
    """
    
    # 1. Recreate the analytical signal
    xa = instant_amp * np.exp(1j * instant_phase)

    # 2. Compute the inverse Hilbert transform
    H = hilbert_filter(V)
    inv_transform = np.linalg.inv(1 + 1j * U @ H @ Uinv)

    # 3. Apply to analytical signal
    ret = inv_transform @ xa 
    return ret

def graph_phase_unwrapping(phase:np.ndarray, adj:np.ndarray, 
                           vmax:Optional[float]=None, method=1):
    """
    Perform graph phase unwrapping following two methods.
    
    Parameters
    ----------
    phase : numpy.ndarray
        Input signal phases.
    adj : numpy.ndarray
        Adjacency matrix of the graph.
    vmax : float, optional
        Maximum eigenvalue of the adjacency matrix. If not provided, it will be computed.
    method : int, optional
        The method to use for phase unwrapping. 0 for a traversal-based method, 1 for a GSO-based method.
    
    Returns
    -------
    numpy.ndarray
        The unwrapped phases.
    """

    if method == 0:
        start_idx = 0
        parsed_nodes = [start_idx]
        for _ in range(adj.shape[0]-1):
            candidates_idx = np.argsort(np.abs(adj[:,start_idx]))
            for cidx in candidates_idx:
                if (adj[cidx,start_idx]!=0) and (cidx not in parsed_nodes):
                    parsed_nodes.append(cidx)
                    break
        parsed_nodes = np.array(parsed_nodes)
        # Case where some nodes are unaccessible
        if len(parsed_nodes) != adj.shape[0]:
            return
        ret = np.unwrap(phase[parsed_nodes])
        return ret
    
    elif method == 1:
        if vmax is None:
            vmax = np.abs(np.linalg.eigvals(adj)).max()

        ordered_phase = np.zeros_like(phase)
        ordered_phase[0] = phase[0]
        indicator = np.zeros(adj.shape[0])
        indicator[0] = 1.0

        shifted_phase = deepcopy(phase)
        for t in range(adj.shape[0]-1):
            shifted_phase = (adj/vmax) @ shifted_phase
            ordered_phase[t+1] = indicator @ shifted_phase

        ret = np.unwrap(ordered_phase)
        return ret

def graph_instant_frequency_HT(phase:np.ndarray, adj:np.ndarray, 
                               vmax:Optional[float]=None, method:int=1):
    """
    Compute graph instant frequency through first phase unwrapping then a GSO shifted phase minus original phase.

    Parameters
    ----------
    phase : numpy.ndarray
        Input signal phases.
    adj : numpy.ndarray
        Adjacency matrix of the graph.
    vmax : float, optional
        Maximum eigenvalue of the adjacency matrix. If not provided, it will be computed.
    method : int, optional
        Method to use for phase unwrapping, default is 1.

    Returns
    -------
    numpy.ndarray
        Generalized instant frequency on the graph.
    """
    
    if vmax is None:
        vmax = np.abs(np.linalg.eigvals(adj)).max()

    unwrapped_phase = graph_phase_unwrapping(phase, adj, vmax=vmax, method=method)
    frequencies = unwrapped_phase - 1/vmax * adj @ unwrapped_phase
    
    return frequencies

def graph_instant_frequency(angle:np.ndarray, adj:np.ndarray):
    """
    Compute generalized instant frequency on graph support.
    
    Parameters:
    ----------
    angle : numpy.ndarray
        Input signal angles.
    adj : numpy.ndarray
        Adjacency matrix of the graph.
    
    Returns:
    -------
    numpy.ndarray
        Generalized instant frequency on the graph.
    """
    ret = np.zeros_like(angle)
    for i in range(len(angle)):
        # neighbours = np.where(adj[:,i])[0]
        neighbours = np.where(adj[i])[0]
        neighbours_angle = angle[neighbours]
        acc = 0
        for n in neighbours_angle:
            if n < angle[i]: 
                acc += n
            else:
                modified_n = np.unwrap([angle[i], n])[-1]
                acc += modified_n
        ret[i] = acc/len(neighbours_angle) - angle[i]

    ret = np.abs(ret)
    return ret

def demodulating_bydivision(signal:np.ndarray, demodulator:np.ndarray):
    """Compute the demodulation by direct division on node domain.

    Parameters
    ----------
    signal : numpy array
        Input signal to demodulate
    demodulator : numpy array 
        Demodulation signal 

    Returns
    -------
    ret : numpy array
        Demodulated signal
    """

    assert signal.shape == demodulator.shape

    div = deepcopy(demodulator)

    # Taking care of division by 0, if the modulator is 0
    # then we map to 0 by default the original signal
    div[div == 0] = np.inf
    ret = signal / div

    return ret

###################################
####### Graph Signal Shift ########
###################################

def fractional_nodeshift(initial:np.ndarray, U:np.ndarray, Vd:np.ndarray, grain:float, niter:Optional[int]=None, force_normalize:bool=False):
    """
    Compute Fractional nodal shift (i.e continuous GSO) and output a sequence of that shift.
    https://ieeexplore.ieee.org/document/9706433
    Parameters
    ----------
    initial : numpy array
        Initial signal to shift
    U : numpy array
        Graph Fourier transform eigenvector matrix  
    Vd : numpy array
        Graph Fourier transform eigenvalues
    grain : float
        Granularity of the shift
    niter : int, optional
        Number of iterations, by default len(Vd)*grain 
    force_normalize : bool, optional
        Whether to normalize the shifted signal to have same norm as initial, by default False

    Returns
    -------
    D_gft : numpy array
        Sequence of graph Fourier transforms of shifted signals 
    D_signals : numpy array 
        Sequence of shifted signals

    """

    initial_norm = np.linalg.norm(initial)
    D_gft = [GFT(initial, U)]
    D_signals = [initial]
    freqshifter = Vd ** (1 / grain)
    if niter is None:
        niter = len(Vd) * grain

    for _ in tqdm(range(niter)):
        # Apply fractional freq shifter
        gftsig = D_gft[-1]
        # newgftsig = np.conjugate(freqshifter) @ gftsig
        newgftsig = freqshifter @ gftsig
        newshift_signal = inverseGFT(newgftsig, U)

        if force_normalize:
            newshift_signal = (
                newshift_signal / np.linalg.norm(newshift_signal) * initial_norm
            )

        D_gft.append(newgftsig)
        D_signals.append(newshift_signal)

    D_gft = np.asarray(D_gft)
    D_signals = np.asarray(D_signals)
    return D_gft, D_signals

def shifter_phase(V:np.ndarray, sparse:int, rc:Optional[str]=None):
    """Compute shifting diagonal matrix shifting the phase of coefficients.

    Keeps the eigenvectors from the decomposition of adjacency matrix. 
    Computes frequency periodic shifts to match pairs of equal eigenvalues.
    Keeps ratio difference at nearest integer between eigenvalues.
    Returns the shifting vector.

    Parameters
    ----------
    V : numpy array
        Eigenvalues from graph Fourier transform 
    sparse : int
        Sparsity of the graph 
    rc : str, optional
        Rounding scheme for ratios, one of "real1", "real2", "abs", by default None

    Returns
    -------
    shifter : numpy array
        Shifting diagonal matrix
    """

    # Find all pairs of eigenvalues
    n = len(V)
    setspairs = []
    for k in range(n):
        pair = list(np.where(V.real[k] == V.real)[0])
        if pair in setspairs:
            continue
        setspairs.append(pair)

    # Keep ratio of shifts between the eigenvalues
    # Rounding Scheme
    if rc == "real1":
        R = (V.real).astype(int)
    elif rc == "real2":
        R = (V.real / V.real[1]).astype(int)
    elif rc == "abs":
        R = (np.abs(V) / np.abs(V)[1]).astype(int)
    else:
        R = None

    # Crux: Frequency Shifting
    shifter = np.zeros((n, n), dtype=complex)
    diagcount, c = 0, 0
    nbsingle = 0
    for pair in setspairs:
        if len(pair) == 1:
            shifter[diagcount, diagcount] = (-1.0) ** (nbsingle)
            diagcount += 1
            nbsingle += 1
        elif len(pair) == 2:
            # Rotate the transformed pairs with (same angle - opposite direction)
            if R is None:
                shifter[diagcount, diagcount] = np.exp(2j * np.pi * c / sparse)
                shifter[diagcount + 1, diagcount + 1] = np.exp(-2j * np.pi * c / sparse)
            else:
                shifter[diagcount, diagcount] = np.exp(
                    2j * np.pi * R[diagcount] / sparse
                )
                shifter[diagcount + 1, diagcount + 1] = np.exp(
                    -2j * np.pi * R[diagcount] / sparse
                )

            diagcount += 2
        else:
            raise InterruptedError
        c += 1

    return shifter

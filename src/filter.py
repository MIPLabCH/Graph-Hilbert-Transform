"""
Copyright © 2024 Chun Hei Michael Chan, MIPLab EPFL
"""

from src.utils import *
from src.metrics import *
from src.operations import *


def spectral_filter_directed(signal:np.ndarray, kernel:np.ndarray, 
                    U:np.ndarray, Uinv:Optional[np.ndarray]=None):
    """
    Applies a graph filter to a signal on a directed graph.
    
    Paramters
    ---------
        signal (np.ndarray): The input signal to be filtered.
        kernel (np.ndarray): The graph filter kernel.
        U (np.ndarray): The eigenvectors of the graph Laplacian.
        V (np.ndarray): The eigenvectors of the graph Laplacian.
        Uinv (Optional[np.ndarray]): The inverse of the eigenvectors of the graph Laplacian.
    
    Returns
    ---------
        np.ndarray: The filtered signal.
    """
        
    filtered_sig = inverseGFT(kernel @ GFT(signal, U, Uinv=Uinv), U)
    return filtered_sig

def wiener_filter(signal:np.ndarray, U:np.ndarray, Uinv:np.ndarray, 
                  kernel_h:np.ndarray, x_psd:np.ndarray, noise_psd:np.ndarray, return_kernel:bool=False):
    """
    Applies a graph wiener filter to a signal on a (undirected & directed) graph.
    
    Paramters
    ---------
        signal (np.ndarray): The input signal to be filtered.
        U (np.ndarray): The eigenvectors of the graph Laplacian.
        Uinv (np.ndarray): The inverse of the eigenvectors of the graph Laplacian.
        kernel_h (np.ndarray): The graph filter kernel.
        x_psd (np.ndarray): The power spectral density of the signal.
        noise_psd (np.ndarray): The power spectral density of the noise.
    
    Returns
    ---------
        np.ndarray: The filtered signal.
    """
    nsize = kernel_h.shape[0]
    g_kernel = np.zeros(nsize, dtype=complex)
    for n in range(nsize):
        g_kernel[n] = kernel_h[n,n] * x_psd[n,n]**2 / (kernel_h[n,n]**2 * x_psd[n,n]**2 + noise_psd[n,n])
    g_kernel = np.diag(g_kernel)
    filtered = spectral_filter_directed(signal, g_kernel, U, Uinv)
    if return_kernel:
        return filtered, g_kernel
    return filtered

def vandermonde_matrix(v:np.ndarray, dim:int):
    """
    Computes the Vandermonde matrix of a vector.

    Paramters
    ---------
        v (np.ndarray): The vector to compute the Vandermonde matrix of.
        dim (int): The dimension of the Vandermonde matrix.
    Returns
    ---------
    np.ndarray: The Vandermonde matrix.
    """
    vdm = np.zeros((len(v), dim)).astype(complex)
    for sidx in range(dim):
        vdm [:, sidx] = v ** sidx
    return vdm

def get_polynomial_coefficients(A:np.ndarray, kernel:np.ndarray,
                                V:np.ndarray, minpolydeg:float, 
                                normalize_gso:bool=True):
    """
    
    Simply solve for (c_i) the system spectral with filter P (i.e kernel) and A=UVU^{-1}
    P = \sum_i\geq 0 c_i V^i
    Paramters
    ---------
    
    Returns
    ---------

    """
    if normalize_gso:
        Vnorm = V / np.abs(V).max()
    else:
        Vnorm = V / 1.0
    vdm = vandermonde_matrix(Vnorm, minpolydeg)
    c = np.linalg.pinv(vdm) @ kernel

    return vdm, c
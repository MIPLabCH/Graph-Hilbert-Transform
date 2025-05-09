"""
Copyright © 2024 Chun Hei Michael Chan, MIPLab EPFL
"""

from src.utils import *


def TV(signal:np.ndarray, L:np.ndarray, normalize:bool=True):
    """
    Compute total variation with L being the laplacian (can be directed)
    (Dirichlet Energy)

    Parameters
    ----------
    signal : array_like
        The signal to compute total variation for. 
    L : array_like
        The laplacian operator.
    normalize : bool, optional
        Whether to normalize the total variation by the L2 norm of the signal.
        Default is True. 

    Returns
    -------
    tv : float
        The total variation of the signal.
    """

    div = 1
    if normalize:
        div = np.linalg.norm(signal)
        if div < 1e-10:
            return 0

    tv = (np.conjugate(signal).T @ L @ signal) / div
    return tv

def TV2(signal:np.ndarray, L:np.ndarray, norm:str="L1", normalize:bool=False, lbd_max:Optional[float]=None):
    """
    Compute total variation with L being the laplacian (can be directed)
    Consider the normed difference

    Parameters
    ----------
    signal : array_like
        The signal to compute total variation for.
    L : array_like
        The laplacian operator.
    norm : {'L1', 'L2'}, optional
        The norm to use for the difference. Default is 'L1'.
    normalize : bool, optional
        Whether to normalize the total variation by the L2 norm of the signal.
        Default is True.

    Returns
    -------
    tv : float
        The total variation of the signal.
    """

    div = 1
    if normalize:
        div = np.linalg.norm(signal)
        if div < 1e-10:
            return 0

    if norm == "L1":
        if lbd_max is None:
            lbd_max = np.abs(np.linalg.eigvals(L)).max()
        tv = np.abs(signal - L/lbd_max @ signal).sum() / div
    else:
        tv = np.linalg.norm(signal - L @ signal) / div
    return tv


def TV3(signal:np.ndarray, L:np.ndarray, norm:str="L2", normalize:bool=False):
    """
    Compute total variation of a signal using the Laplacian operator.
    (2-Dirichlet Energy)

    Parameters
    ----------
    signal : ndarray
        The signal for which to compute total variation.
    L : ndarray
        The graph Laplacian operator.  
    norm : {'L1', 'L2'}, optional
        The norm to use. Default is 'L2'.
    normalize : bool, optional  
        Whether to normalize by the L2 norm of the signal. Default is False.

    Returns
    -------
    tv : float
        The total variation of the signal.
    """

    div = 1
    if normalize:
        div = np.linalg.norm(signal)
        if div < 1e-10:
            return 0

    if norm == "L1":
        tv = np.abs(L @ signal).sum() / div
    else:
        tv = np.linalg.norm(L @ signal) / div
    return tv
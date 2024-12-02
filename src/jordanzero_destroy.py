"""
Copyright © 2024 Chun Hei Michael Chan, MIPLab EPFL
"""

from src.utils import *
from src.operations import *

from scipy import linalg

def find_best_pair(A:np.ndarray, k:np.ndarray, vl:np.ndarray, 
                   vr:np.ndarray, opt_eps:float=1e-5):
    
    """
    Find the best pairs of indices (i, j) in the matrix A that maximize the product of the corresponding left and right eigenvectors for the eigenvalue with index k.

    Parameters
    ----------
    A : np.ndarray
        The input matrix.
    k : np.ndarray
        The index of the eigenvalue to consider.
    vl : np.ndarray
        The left eigenvectors of A.
    vr : np.ndarray
        The right eigenvectors of A.
    opt_eps : float, optional
        The tolerance for selecting the best pairs, by default 1e-5.

    Returns
    -------
    np.ndarray
        The best pairs of indices (i, j) that maximize the product of the corresponding left and right eigenvectors.
    """

    mask = (A == 0).astype(float)
    cross_values = np.outer(np.abs(vl[:,k]), np.abs(vr[:,k]))
    cross_values = cross_values * mask

    best_pairs = np.asarray(np.where(np.abs(cross_values - cross_values.max()) < opt_eps)).T
    return best_pairs


def destroy_jordan_blocks(A:np.ndarray, prefer_nodes:list=[]):
    """
    Destroy the Jordan blocks in the input matrix A by setting the smallest entries to 1.

    Parameters
    ----------
    A : np.ndarray
        The input matrix.
    prefer_nodes : list, optional
        A list of preferred node indices to prioritize when selecting the entries to set to 1, by default [].

    Returns
    -------
    np.ndarray
        The modified matrix A with the Jordan blocks destroyed.
    """

    ret = deepcopy(A)
    _, vl, vr = linalg.eig(ret, left=True)
    n = len(A)
    while np.linalg.matrix_rank(vr) < n:
        D = np.nan_to_num(np.arccos(np.abs(vr.T@vr)))
        k = np.argmax(np.sum(D < 1e-6, axis=1))
        best_pairs = find_best_pair(ret, k, vl, vr)
        both = []
        one = []
        for pair in best_pairs:
            if (pair[0] in prefer_nodes) and (pair[1] in prefer_nodes):
                i,j = pair
                both.append((i,j))
            elif (pair[0] in prefer_nodes) or (pair[1] in prefer_nodes):
                i,j = pair
                one.append((i,j))

        if len(both) != 0: i,j = both[0]
        elif len(one)!= 0: i,j = one[0]
        else: i,j = best_pairs[0]
        ret[i,j] = 1
        _, vl ,vr = linalg.eig(ret, left=True)

    return ret



def destroy_zero_eigenvals(A:np.ndarray, prefer_nodes:list=[], eps:float=1e-6, tol:float=1e-4, verbose:bool=False):
    """
    Destroy the zero eigenvalues in the input matrix A by setting the smallest entries to 1.

    Parameters
    ----------
    A : np.ndarray
        The input matrix.
    prefer_nodes : list, optional
        A list of preferred node indices to prioritize when selecting the entries to set to 1, by default [].
    eps : float, optional
        The tolerance value for considering an eigenvalue as zero, by default 1e-6.

    Returns
    -------
    np.ndarray
        The modified matrix A with the zero eigenvalues destroyed.
    """
    
    ret = deepcopy(A)
    D, vl, vr = linalg.eig(ret, left=True)
    repeat_count = 0
    prev_rank = -1
    while np.min(np.abs(D)) < eps:
        # k = np.argmin(np.abs(D))
        
        possible_index = np.where((np.abs(D) - np.min(np.abs(D))) < tol)[0]
        cur_rank = len(possible_index)
        if verbose:
            print(f"Dimension of null space={cur_rank}")

        if cur_rank == prev_rank:
            repeat_count += 1
        else:
            repeat_count = 0
        if repeat_count >= 10:
            if verbose: print("Remove preferential nodes")
            best_pairs = find_best_pair(ret, np.argmin(np.abs(D)), vl, vr)
        else:
            best_pairs = [find_best_pair(ret, possible_index[k], vl, vr) for k in range(len(possible_index))]
            best_pairs = np.concatenate(best_pairs)

        both = []
        one = []
        for pair in best_pairs:
            if (pair[0] in prefer_nodes) and (pair[1] in prefer_nodes):
                i,j = pair
                both.append((i,j))
            elif (pair[0] in prefer_nodes) or (pair[1] in prefer_nodes):
                i,j = pair
                one.append((i,j))

        if len(both) != 0: i,j = both[0]
        elif len(one)!= 0: i,j = one[0]
        else: i,j = best_pairs[0]
        ret[i,j] = 1

        # Update
        D, vl ,vr = linalg.eig(ret, left=True)
        prev_rank = cur_rank


    return ret
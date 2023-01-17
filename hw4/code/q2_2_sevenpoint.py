import numpy as np
import sympy as sym
import matplotlib.pyplot as plt

from helper import displayEpipolarF, calc_epi_error, toHomogenous, _singularize
# Insert your package here


'''
Q2.2: Seven Point Algorithm for calculating the fundamental matrix
    Input:  pts1, 7x2 Matrix containing the corresponding points from image1
            pts2, 7x2 Matrix containing the corresponding points from image2
            M, a scalar parameter computed as max (imwidth, imheight)
    Output: Farray, a list of estimated 3x3 fundamental matrixes.
    
    HINTS:
    (1) Normalize the input pts1 and pts2 scale paramter M.
    (2) Setup the seven point algorithm's equation.
    (3) Solve for the least square solution using SVD. 
    (4) Pick the last two colum vector of vT.T (the two null space solution f1 and f2)
    (5) Use the singularity constraint to solve for the cubic polynomial equation of  F = a*f1 + (1-a)*f2 that leads to 
        det(F) = 0. Solving this polynomial will give you one or three real solutions of the fundamental matrix. 
        Use np.polynomial.polynomial.polyroots to solve for the roots
    (6) Unscale the fundamental matrixes and return as Farray
'''
def sevenpoint(pts1, pts2, M):
    T = np.diag([1 / M, 1 / M, 1])
    N = pts1.shape[0]
    ones_arr = np.ones((N,1))
    pts1 = np.hstack((pts1,ones_arr))
    pts2 = np.hstack((pts2,ones_arr))
    norm1 = np.matmul(T,pts1.T).T
    norm2 = np.matmul(T,pts2.T).T

    A = np.ones((N,9))
    for i in range(N):
        x1 = norm1[i,0]
        x2 = norm2[i,0]
        y1 = norm1[i,1]
        y2 = norm2[i,1]
        A[i,:] = [x1*x2, x1*y2, x1, y1*x2, y1*y2, y1, x2, y2, 1]

    A = np.asarray(A)
    D, V = np.linalg.eig(np.dot(A.T, A))
    f1 = np.reshape(V[:, -1], (3, 3))
    f2 = np.reshape(V[:, -2], (3, 3))
    f1 = f1 / f1[2, 2]
    f2 = f2 / f2[2, 2]

    a = sym.symbols('a')
    func = a*f1+(1-a)*f2
    func = sym.Matrix(func)
    d = func.det()
    c0 = d.coeff(a,0)
    c1 = d.coeff(a,1)
    c2 = d.coeff(a,2)
    c3 = d.coeff(a,3)
    C = np.asarray([c0,c1,c2,c3]).astype(float)
    sol = np.polynomial.polynomial.polyroots(C)

    F_list = []
    for root in sol:
        if np.isreal(root):
            F = root*f1+(1-root)*f2
            F = np.matmul(T.T, np.matmul(F, T))
            F = F / F[2, 2]
            F_list.append(F)

    return F_list


if __name__ == "__main__":
        
    correspondence = np.load('data/some_corresp.npz') # Loading correspondences
    intrinsics = np.load('data/intrinsics.npz') # Loading the intrinscis of the camera
    K1, K2 = intrinsics['K1'], intrinsics['K2']
    pts1, pts2 = correspondence['pts1'], correspondence['pts2']
    im1 = plt.imread('data/im1.png')
    im2 = plt.imread('data/im2.png')

    # indices = np.arange(pts1.shape[0])
    # indices = np.random.choice(indices, 7, False)
    indices = np.array([82, 19, 56, 84, 54, 24, 18])

    M = np.max([*im1.shape, *im2.shape])
    Farray = sevenpoint(pts1[indices, :], pts2[indices, :], M)
    F = Farray[0]
    print("F: ",F)
    print()
    np.savez('q2_2.npz', F, M, pts1, pts2)

    # fundamental matrix must have rank 2!
    assert(np.linalg.matrix_rank(F) == 2)
    displayEpipolarF(im1, im2, F)

    # Simple Tests to verify your implementation:
    # Test out the seven-point algorithm by randomly sampling 7 points and finding the best solution. 
    np.random.seed(1) #Added for testing, can be commented out
    
    pts1_homogenous, pts2_homogenous = toHomogenous(pts1), toHomogenous(pts2)

    max_iter = 500
    pts1_homo = np.hstack((pts1, np.ones((pts1.shape[0], 1))))
    pts2_homo = np.hstack((pts2, np.ones((pts2.shape[0], 1))))

    ress = []
    F_res = []
    choices = []
    M=np.max([*im1.shape, *im2.shape])
    for i in range(max_iter):
        choice = np.random.choice(range(pts1.shape[0]), 7)
        pts1_choice = pts1[choice, :]
        pts2_choice = pts2[choice, :]
        Fs = sevenpoint(pts1_choice, pts2_choice, M)
        for F in Fs:
            choices.append(choice)
            res = calc_epi_error(pts1_homo,pts2_homo, F)
            F_res.append(F)
            ress.append(np.mean(res))
            
    min_idx = np.argmin(np.abs(np.array(ress)))
    F = F_res[min_idx]
    print("Error:", ress[min_idx])

    assert(F.shape == (3, 3))
    assert(F[2, 2] == 1)
    assert(np.linalg.matrix_rank(F) == 2)
    assert(np.mean(calc_epi_error(pts1_homogenous, pts2_homogenous, F)) < 1)
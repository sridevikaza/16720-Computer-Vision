import numpy as np
import matplotlib.pyplot as plt

from helper import displayEpipolarF, calc_epi_error, toHomogenous, refineF, _singularize

# Insert your package here

'''
Q2.1: Eight Point Algorithm
    Input:  pts1, Nx2 Matrix
            pts2, Nx2 Matrix
            M, a scalar parameter computed as max (imwidth, imheight)
    Output: F, the fundamental matrix

    HINTS:
    (1) Normalize the input pts1 and pts2 using the matrix T.
    (2) Setup the eight point algorithm's equation.
    (3) Solve for the least square solution using SVD.
    (4) Use the function `_singularize` (provided) to enforce the singularity condition.
    (5) Use the function `refineF` (provided) to refine the computed fundamental matrix.
        (Remember to usethe normalized points instead of the original points)
    (6) Unscale the fundamental matrix
'''
def eightpoint(pts1, pts2, M):
    # Replace pass by your implementation
    T = np.diag([1/M,1/M,1])
    N = pts1.shape[0]
    ones_arr = np.ones((N,1))
    pts1 = np.hstack((pts1,ones_arr))
    pts2 = np.hstack((pts2,ones_arr))
    norm1 = np.matmul(T,pts1.T).T
    norm2 = np.matmul(T,pts2.T).T

    # compute A
    A = np.ones((N,9))
    for i in range(N):
        x1 = norm1[i,0]
        x2 = norm2[i,0]
        y1 = norm1[i,1]
        y2 = norm2[i,1]
        A[i,:] = [x1*x2, x1*y2, x1, y1*x2, y1*y2, y1, x2, y2, 1]

    # SVD
    D, V = np.linalg.eig(np.dot(A.T, A))
    idx = np.argmin(D)
    F = np.reshape(V[:, idx], (3, 3)).T

    # singularize and refine
    F = _singularize(F)
    F = refineF(F,norm1[:,0:2],norm2[:,0:2])

    # denormalize
    F = np.matmul(T.T,np.matmul(F, T))

    # scale
    F = F / F[2,2]
    
    return F


if __name__ == "__main__":

    correspondence = np.load('data/some_corresp.npz') # Loading correspondences
    intrinsics = np.load('data/intrinsics.npz') # Loading the intrinscis of the camera
    K1, K2 = intrinsics['K1'], intrinsics['K2']
    pts1, pts2 = correspondence['pts1'], correspondence['pts2']
    im1 = plt.imread('data/im1.png')
    im2 = plt.imread('data/im2.png')

    M = np.max([*im1.shape, *im2.shape])
    F = eightpoint(pts1, pts2, M)
    print(F)
    np.savez('results/q2_1.npz', F=F, M=M)

    # Q2.1
    displayEpipolarF(im1, im2, F)

    # Simple Tests to verify your implementation:
    pts1_homogenous, pts2_homogenous = toHomogenous(pts1), toHomogenous(pts2)

    assert(F.shape == (3, 3))
    assert(F[2, 2] == 1)
    assert(np.linalg.matrix_rank(F) == 2)
    assert(np.mean(calc_epi_error(pts1_homogenous, pts2_homogenous, F)) < 1)

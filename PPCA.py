import numpy as np
import argparse
import random

# Ahora todo varias veces
def has_converged(C_old, C_new, ss_old, ss_new, tol=1e-4):
    delta_C = np.linalg.norm(C_new - C_old, ord='fro')
    delta_ss = abs(ss_new - ss_old)
    return delta_C < tol and delta_ss < tol


def main(args):
    random.seed(42)
    np.random.seed(42)
    # Leer archivo TXT como RDD
    pathcsv = args.input
    Y_p = np.genfromtxt(pathcsv, delimiter=",")  # convierte a numpy array
    D = Y_p.shape[1]  # number of columns
    d = args.dim  # dimension of the projection
    N = Y_p.shape[0]
    C_p = np.random.normal(loc=0.0, scale=1.0, size=(D, d))
    ss_p = np.random.normal(loc=0.0, scale=1.0)
    I = np.eye(d)
    C_old =  np.inf
    ss_old = np.inf
    Ym = np.sum(Y_p, axis=0) / N
    Yc = Y_p - Ym
    max_iters = args.maxIters
    num_iters = 0
    while not has_converged(C_old, C_p, ss_old, ss_p) and num_iters < max_iters:
        M = C_p.T @ C_p + ss_p * I
        M_inv = np.linalg.inv(M)
        X_p = Yc @ C_p @ M_inv
        XtX_p = X_p.T @ X_p + (ss_p * M_inv)
        YtX_p = Yc.T @ X_p
        C_old = C_p
        C_p = YtX_p @ np.linalg.inv(XtX_p)
        ss2_p = np.trace(XtX_p @ C_p.T @ C_p)
        ss3_p = 0
        for i in range(0,N):
            ss3_p += X_p[i,:] @ C_p.T @ Yc[i,:].T
        ss_old = ss_p
        ss_p = (np.linalg.norm(Yc, ord='fro')**2 + ss2_p - 2 * ss3_p) / N / D
        num_iters += 1
        print(f"{num_iters} ss : {ss_p}")

    np.savetxt(args.output, C_p, delimiter=",")




if __name__ == '__main__':
    """Run PPCA"""
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, default='', help='Input file txt')
    parser.add_argument('--dim', type=int, default=3, help='Number of output dimensions')
    parser.add_argument('--maxIters', type=int, default=100, help='Maximum iterations')
    parser.add_argument('--output', type=str, default='', help='Output file')
    args = parser.parse_args()
    main(args)

import pyspark
import numpy as np
import argparse
import glob
import os
from pyspark.sql import SparkSession
from pyspark.accumulators import AccumulatorParam
from utils import meanJob, center_data, FJob, YtXSparkJob, ss3Job
import random


# Ahora todo varias veces
def has_converged(C_old, C_new, ss_old, ss_new, tol=1e-4):
    delta_C = np.linalg.norm(C_new - C_old, ord='fro')
    delta_ss = abs(ss_new - ss_old)
    return delta_C < tol and delta_ss < tol


def main(args):
    # inicializar spark
    spark = SparkSession.builder.appName("sPCA").getOrCreate()
    sc = spark.sparkContext  # así obtienes el SparkContext moderno
    random.seed(42)
    np.random.seed(42)

    # Leer archivo TXT como RDD
    rdd = sc.textFile(args.input)  # carga como RDD de líneas
    Y = rdd.map(lambda line: [float(x) for x in line.split(",")])
    D = len(Y.first())  # number of columns
    d = args.dim  # dimension of the projection
    N = Y.count()

    # start probabilistic PCA
    C = np.random.normal(loc=0.0, scale=1.0, size=(D, d))
    ss = np.random.normal(loc=0.0, scale=1.0)
    Ym = np.array(meanJob(Y))
    ss1 = FJob(Y, Ym)
    I = np.eye(d)

    max_iters = args.maxIters
    C_old =  np.inf
    ss_old = np.inf
    num_iters = 0
    while not has_converged(C_old, C, ss_old, ss) and num_iters < max_iters:
        # Compute M = Cᵀ * C + ss * I
        M = C.T @ C + ss * I

        # Compute CM = C * M⁻¹
        M_inv = np.linalg.inv(M)
        CM = C @ M_inv

        # Compute Xm = Ym * CM
        Xm = Ym @ CM  # result has shape (d,)
        YtX, XtX = YtXSparkJob(Y, Ym, Xm, CM, D, d, sc)
        XtX += ss * M_inv
        XtX_inv = np.linalg.inv(XtX)
        C_old = C
        C = YtX @ XtX_inv
        ss2 = np.trace(XtX @ C.T @ C)
        ss3 = ss3Job(Y, Ym, Xm, CM, C, sc)
        ss_old = ss
        ss = (ss1**2 + ss2 -2 * ss3) / N / D
        num_iters +=1
        print(f"{num_iters} ss : {ss}")

    # Columns of C are the principal components
    np.savetxt(args.output, C, delimiter=",")




if __name__ == '__main__':
    """Run sPCA"""
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, default='', help='Input file txt')
    parser.add_argument('--dim', type=int, default=3, help='Number of output dimensions')
    parser.add_argument('--maxIters', type=int, default=100, help='Maximum iterations')
    parser.add_argument('--output', type=str, default='', help='Output file')
    args = parser.parse_args()
    main(args)
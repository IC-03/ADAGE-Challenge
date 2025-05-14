import os
from pyspark.sql import SparkSession
import numpy as np
import random
import argparse
from pyspark.accumulators import AccumulatorParam

#################### clases #################################
# Clase acumuladora (usando listas de listas)
class MatrixAccumulatorParam(AccumulatorParam):
    def zero(self, value):
        rows = len(value)
        cols = len(value[0])
        return [[0.0 for _ in range(cols)] for _ in range(rows)]

    def addInPlace(self, val1, val2):
        for i in range(len(val1)):
            for j in range(len(val1[0])):
                val1[i][j] += val2[i][j]
        return val1

# Wrapper de la clase acumuladora
class MatrixAccumulatorWrapper:
    def __init__(self, sc, shape):
        self.shape = shape
        self.accumulator = sc.accumulator(
            [[0.0 for _ in range(shape[1])] for _ in range(shape[0])],
            MatrixAccumulatorParam()
        )

    def addRow(self, row_index, row_contribution):
        temp = [[0.0 for _ in range(self.shape[1])] for _ in range(self.shape[0])]
        temp[row_index] = list(row_contribution)
        self.accumulator.add(temp)

    def add(self, matrix):
        self.accumulator.add(matrix)

    def get(self):
        return self.accumulator.value


################# funciones ##########################
# Calcular la media de cada columna
def meanJob(Y):
    suma_vectores = Y.reduce(lambda a, b: [x + y for x, y in zip(a, b)])
    n = Y.count()
    media = [x / n for x in suma_vectores]
    return media

# Center data con la media de cada columna
def center_data(Y, Ym):
    return Y.map(lambda fila: [x - m for x, m in zip(fila, Ym)])

# Computa la norma de forbenius de forma distribuida
def FJob(Y_rdd, Ym):
    diff_squared = Y_rdd.map(lambda row: sum((xi - mi) ** 2 for xi, mi in zip(row, Ym)))
    fro_squared_sum = diff_squared.reduce(lambda a, b: a + b)
    return np.sqrt(fro_squared_sum)


# calcula YtX y XtX de forma distribuida 
def YtXSparkJob(Y, Ym, Xm, CM, D, d, sc):
    YtXSum = MatrixAccumulatorWrapper(sc, shape=(D, d))
    XtXSum = MatrixAccumulatorWrapper(sc, shape=(d, d))
    Ym = np.array(Ym)
    Xm = np.array(Xm)
    CM = np.array(CM)

    # Broadcast variables to use inside workers
    Ym_b = sc.broadcast(Ym)
    Xm_b = sc.broadcast(Xm)
    CM_b = sc.broadcast(CM)

    def compute_partial(Yi):
        try:
            Yi = np.array(Yi)

            CM = CM_b.value
            Ym = Ym_b.value
            Xm = Xm_b.value

            Xi = (Yi @ CM) - (Ym @ CM)

            YtX_i = np.outer(Yi - Ym, Xi - Xm)
            XtX_i = np.outer(Xi, Xi)

            YtXSum.add(YtX_i)
            XtXSum.add(XtX_i)
        except Exception as e:
            import traceback
            print("Error in compute_partial:")
            traceback.print_exc()
            raise e


    Y.foreach(compute_partial)
    YtX = YtXSum.accumulator.value
    XtX = XtXSum.accumulator.value
    return YtX, XtX

# calcular ss3 de forma distribuida
def ss3Job(Y, Ym, Xm, CM, C, sc):
    ss3 = sc.accumulator(0)
    Ym = np.array(Ym)
    Xm = np.array(Xm)
    CM = np.array(CM)
    C = np.array(C)

    # Broadcast variables to use inside workers
    Ym_b = sc.broadcast(Ym)
    Xm_b = sc.broadcast(Xm)
    CM_b = sc.broadcast(CM)
    C_b = sc.broadcast(C)

    def calculate_ss3(Yi):
        try: 
            Yi = np.array(Yi)
            CM = CM_b.value
            Ym = Ym_b.value
            Xm = Xm_b.value
            C = C_b.value
        
            Xi = Yi @ CM - Ym @ CM
            #Xi = Xi - Xm
            cy = C.T @ Yi.T
            xcy = Xi @ cy
            ss3.add(xcy)
        except Exception as e:
            import traceback
            print("Error in calculate_ss3:")
            traceback.print_exc()
            raise e
    Y.foreach(calculate_ss3)
    return ss3.value


############# start spark session #############################

# for client mode
spark = SparkSession.builder.appName("sPCA").getOrCreate()

# for cluster mode only  
#os.environ['PYSPARK_PYTHON'] = "./environment/env/bin/python"
#spark = SparkSession.builder.config(
#    "spark.yarn.dist.archives", "env.tar.gz#environment"
#).appName("sPCA").getOrCreate()
###


sc = spark.sparkContext

# Get the arguments 
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, default='', help='Input file txt')
    parser.add_argument('--dim', type=int, default=3, help='Number of output dimensions')
    parser.add_argument('--maxIters', type=int, default=100, help='Maximum iterations')
    parser.add_argument('--output', type=str, default='', help='Output file')
    return parser.parse_args()

# Ahora todo varias veces
def has_converged(C_old, C_new, ss_old, ss_new, tol=1e-4):
    delta_C = np.linalg.norm(C_new - C_old, ord='fro')
    delta_ss = abs(ss_new - ss_old)
    return delta_C < tol and delta_ss < tol


def main(sc):
    args = parse_args()
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

    # loop and update C and ss until the error is very small or the number of iterations is big
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
    sc.parallelize(C).map(lambda row: ",".join(map(str, row))).coalesce(1).saveAsTextFile(args.output + "_C")
    # Define a function to reduce dimensionality by multiplying with C
    def reduce_dimensionality(row):
        return np.dot(row, C)

    # Apply dimensionality reduction to each row in Y
    reduced_rdd = Y.map(reduce_dimensionality)
    # Save the reduced data as a single output file
    reduced_rdd.map(lambda row: ",".join(map(str, row))) \
        .coalesce(1) \
        .saveAsTextFile(args.output + "_reduced")

main(sc)
spark.stop()
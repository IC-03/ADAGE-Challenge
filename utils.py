import numpy as np
from pyspark.sql import SparkSession
from pyspark.accumulators import AccumulatorParam
from accumulators import MatrixAccumulatorWrapper

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
#            XtX_i = np.outer(Xi - Xm, Xi - Xm)
            XtX_i = np.outer(Xi, Xi)

            #YtX_i = np.outer(Yi, Xi_minus_Xm) - np.outer(Ym, Xi_minus_Xm)
            #XtX_i = np.outer(Xi, Xi_minus_Xm) - np.outer(Xm, Xi_minus_Xm)
            #non_zero_indices = np.nonzero(Yi)[0]
            #for idx in non_zero_indices:
            #    YtXSum.addRow(idx, YtX_i[idx, :])
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
    
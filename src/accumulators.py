# accumulators.py

from pyspark.accumulators import AccumulatorParam

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

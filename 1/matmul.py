Matrix = list[list[int]]


def t(matrix: Matrix) -> Matrix:
    if not matrix or not matrix[0]:
        return []

    rows = len(matrix)
    cols = len(matrix[0])

    return [[matrix[j][i] for j in range(rows)] for i in range(cols)]


def matmul(m1: Matrix, m2: Matrix) -> Matrix|None:
    if not m1 or not m2 or not m1[0] or not m2[0]:
        return None

    rows1, cols1 = len(m1), len(m1[0])
    rows2, cols2 = len(m2), len(m2[0])

    if cols1 != rows2:
        return None

    result = []
    for i in range(rows1):
        row = []
        for j in range(cols2):
            total = sum(m1[i][k] * m2[k][j] for k in range(cols1))
            row.append(total)
        result.append(row)

    return result

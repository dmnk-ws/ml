from constants import MATRIX_A, VECTOR_X, VECTOR_Y, VECTOR_Z
from matmul import matmul, t


def main() -> None:
    tasks = {
        "Ay": matmul(MATRIX_A, VECTOR_Y),
        "Ax": matmul(MATRIX_A, VECTOR_X),
        "x^Ty": matmul(t(VECTOR_X), VECTOR_Y),
        "xA": matmul(VECTOR_X, MATRIX_A),
        "Ay^T": matmul(MATRIX_A, t(VECTOR_Y)),
        "x^T(Ay)": matmul(t(VECTOR_X), matmul(MATRIX_A, VECTOR_Y)),
        "xy^T": matmul(VECTOR_X, t(VECTOR_Y)),
        "Az": matmul(MATRIX_A, VECTOR_Z),
        "zx^T": matmul(VECTOR_Z, t(VECTOR_X)),
        "AA": matmul(MATRIX_A, MATRIX_A),
        "AA^T": matmul(MATRIX_A, t(MATRIX_A)),
    }

    for i, (description, result) in enumerate(tasks.items(), 1):
        print(f"{i}. {description} = {result}")


if __name__ == "__main__":
    main()
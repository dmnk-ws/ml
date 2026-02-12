import numpy as np #1

def task2() -> None:
    print(np.version.version)

def task3() -> None:
    A = np.zeros(10)
    print(A)

def task4() -> None:
    A = np.ones([10, 10])
    print(A.nbytes)

def task5() -> None:
    # print(np.add.__doc__)
    print(np.info(np.add))

def task6() -> None:
    A = np.zeros(10)
    A[4] = 1
    print(A)

def task7() -> None:
    A = np.arange(10, 50)
    print(A)

def task8() -> None:
    A = np.arange(0, 50)
    print(A)
    B = A[::-1]
    print(B)

def task9() -> None:
    A = np.arange(9).reshape(3, 3)
    print(A)

def task10() -> None:
    A = np.nonzero([1,2,0,0,4,0])
    print(A)

def task11() -> None:
    A = np.identity(3)
    print(A)
    B = np.eye(3)
    print(B)

def task12() -> None:
    A = np.random.random([3, 3, 3])
    print(A)

def task13() -> None:
    A = np.random.random([10, 10])
    print(A)
    print(A.max(), A.min())

def task14() -> None:
    A = np.random.random(30)
    print(A)
    print(A.mean())

def task15() -> None:
    A = np.ones([10, 10])
    A[1:-1,1:-1] = 0 # A[row_start:row_stop,col_start:col_stop]
    print(A)

def task16() -> None:
    # A = np.ones([10, 10])
    # A[0:1,:] = 0
    # A[-1:,:] = 0
    # A[:,0:1] = 0
    # A[:,-1:] = 0
    # print(A)

    # fancy indexing
    B = np.ones([10, 10])
    B[[0, -1],:] = 0
    B[:, [0, -1]] = 0
    print(B)

def task17() -> None:
    print(0 * np.nan) # nan - Arithmetic with NaN gives NaN
    print(np.nan == np.nan) # False - NaN is never equal to anything
    print(np.inf > np.nan) # False - Comparisons with NaN are False
    print(np.nan - np.nan) # nan - Undefined operation
    print(np.nan in {np.nan}) # True - Hashing behavior in sets
    print(0.3 == 3 * 0.1) # False - Floating-point rounding error

def task18() -> None:
    A = np.diag(np.arange(1, 5), k=-1)
    print(A)

def task19() -> None:
    A = np.zeros([8, 8])
    A[1::2,::2] = 1
    A[::2, 1::2] = 1
    print(A)

def task20() -> None:
    print(np.unravel_index(99, (6,7,8)))

def task21() -> None:
    A = np.array([[0., 1.], [1., 0.]])
    print(np.tile(A, (4, 4)))

def task22() -> None:
    A = np.random.random((5, 5))
    A = (A - np.mean(A)) / np.std(A)
    print(A)
    print(A.mean(), A.std())

def task23() -> None:
    color = np.dtype([("r", np.ubyte),
                      ("g", np.ubyte),
                      ("b", np.ubyte),
                      ("a", np.ubyte)])
    print(color)

def task24() -> None:
    A = np.ones((5,3)) * np.arange(1, 16).reshape(5,3)
    B = np.ones((3,2)) * np.arange(7, 1, -1).reshape(3,2)
    print(np.matmul(A, B))
    print(A @ B)

def task25() -> None:
    A = np.arange(1, 16)
    A[(3 < A) & (A < 8)] *= -1
    print(A)

def task26() -> None:
    print(sum(range(5), -1))
    # from numpy import *
    # SyntaxError: import * only allowed at module level
    # print(sum(range(5), -1))

def task27() -> None:
    Z = np.arange(1, 3)
    print(Z ** Z)
    print(2 << Z >> 2)
    print(Z <- Z)
    print(1j * Z)
    print(Z / 1 / 1)
    print(Z<Z>Z) # ValueError: The truth value of an array with more than one element is ambiguous. Use a.any() or a.all()

def task28() -> None:
    print(np.array(0) / np.array(0))
    print(np.array(0) // np.array(0))
    print(np.array([np.nan]).astype(int).astype(float))

def main() -> None:
    # task2()
    # task3()
    # task4()
    # task5()
    # task6()
    # task7()
    # task8()
    # task9()
    # task10()
    # task11()
    # task12()
    # task13()
    # task14()
    # task15()
    # task16()
    # task17()
    # task18()
    # task19()
    # task20()
    # task21()
    # task22()
    # task23()
    # task24()
    # task25()
    # task26()
    # task27()
    task28()

if __name__ == "__main__":
    main()

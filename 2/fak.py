def fak(n):
    return 1 if n == 0 else n * fak(n-1)

if __name__ == '__main__':
    for x in range(1, 7):
        print(fak(x))
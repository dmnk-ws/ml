def slice_first(x: list) -> tuple[int, list]:
    return x[0], x[1:]

def sum_dif_mod(x: list) -> list:
    return [x[0] + x[-1], x[0] - x[-1], x[0] % x[-1]]

def sum_list(x: list) -> int:
    return sum(x)

def odd(x: list) -> list:
    return [x[i] for i in range(1, len(x), 2)]

def invert_skip_first(x: list) -> list:
    return x[:0:-1]

def invert(x: list) -> list:
    return x[::-1]

def odd_mask(x: list) -> list:
    return [1 if i % 2 == 1 else 0 for i in range(len(x))]
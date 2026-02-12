from utils import *

def main() -> None:
    nums = [x for x in range(50)]

    tasks = {
        1: slice_first(nums),
        2: sum_dif_mod(nums),
        3: sum_list(nums),
        4: odd(nums),
        5: invert_skip_first(nums),
        6: invert(nums),
        7: odd_mask(nums)
    }

    for k, v in tasks.items():
        print(f"{k}. {v}")


if __name__ == "__main__":
    main()
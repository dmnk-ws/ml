def pr(iterable, i: int) -> None:
    for index, element in enumerate(iterable):
        if index >= i:
            break
        print(element, end=" ")
    print("\n")


def main() -> None:
    tasks = {
        "Tuple": ((1,2,3), 3),
        "List": ([1,2,3], 3),
        "String": ("abc", 3),
        "Dictionary": ({"a": 1, "b": 2, "c": 3}, 3),
        "Iterator": (range(3), 3)
    }

    for i, (k, v) in enumerate(tasks.items(), 1):
        print(f"{i}. {k}")
        pr(*v)

if __name__ == '__main__':
    main()
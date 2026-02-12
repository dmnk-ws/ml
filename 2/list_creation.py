def main() -> None:
    tasks = {
        1: [x for x in range(0, 11, 2)],
        2: [x for x in range(100) if x % 15 == 0],
        3: [x for x in range(15, 0, -1) if x % 2 == 1],
        4: ["xx"] * 5,
        5: ["string" + "X" * x for x in range(5, 15)],
        6: ["1", 2, 3.0, 4],
        7: [x for x in range(100) if str(x).find("3") != -1]
    }

    for k, v in tasks.items():
        print(f"{k}. {v}")


if  __name__ == "__main__":
    main()
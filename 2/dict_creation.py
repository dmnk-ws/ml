def main() -> None:
    nums = {}

    for i in range(1, 11):
        nums[i] = str(i)

    print("Keys:")
    for key in nums.keys():
        print(key)
    print()

    print("Values:")
    for value in nums.values():
        print(value)
    print()

    print("Keys and values:")
    for key, value in nums.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
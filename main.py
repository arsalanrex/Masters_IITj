import sys
import random


def miller_rabin(n, k):
    """
    Perform the Miller-Rabin primality test on an integer n.

    :param n: The integer to test for primality.
    :param k: The number of iterations (tests) to perform.
    :return: 1 if n is probably prime, 0 if n is composite.
    """
    # Step 1: Handle edge cases
    if n == 2 or n == 3:
        return 1
    if n % 2 == 0 or n < 2:
        return 0

    # Step 2: Write n-1 as d * 2^r
    r = 0
    d = n - 1
    while d % 2 == 0:
        d //= 2
        r += 1

    # Step 3: Perform k iterations of the test
    for _ in range(k):
        a = random.randint(2, n - 2)
        x = pow(a, d, n)

        if x == 1 or x == n - 1:
            continue

        for _ in range(r - 1):
            x = pow(x, 2, n)
            if x == n - 1:
                break
        else:
            return 0

    return 1


def main():
    if len(sys.argv) != 2:
        print("Usage: python miller_rabin.py <number>")
        return

    try:
        n = int(sys.argv[1])
    except ValueError:
        print("Error: The input must be an integer.")
        return

    k = 5  # Number of trials to perform

    result = miller_rabin(n, k)
    print(result)


if __name__ == "__main__":
    main()

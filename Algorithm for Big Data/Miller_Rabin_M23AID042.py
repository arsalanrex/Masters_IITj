import sys
import random


# this will give a probable result of whether the number is prime or not
# 1 means probably prime, 0 means composite

def miller_rabin(n, k):

    # Step 1: handling small edge cases directly
    if n == 2 or n == 3:
        return 1  # 2 and 3 are prime numbers
    if n % 2 == 0 or n < 2:
        return 0  # Any even number or number less than 2 is not prime

    # Step 2: Write n - 1 as d * 2^r by finding r and d
    # where d is an odd number
    r = 0
    d = n - 1
    while d % 2 == 0:  # Keep dividing d by 2 until it is odd
        d //= 2
        r += 1

    # Step 3: Perform the Miller-Rabin test k times with different bases
    for _ in range(k):
        # Step 3.1: Choosing random integer 'a' in the range [2, n-2]
        a = random.randint(2, n - 2)

        # Step 3.2: Compute b = a^d % n using modular exponentiation
        x = pow(a, d, n)

        # Step 3.3: Check if x is 1 or n - 1
        if x == 1 or x == n - 1:
            continue  # 'n' may be prime in this round, proceed to the next round

        # Step 3.4: Repeat the squaring process r-1 times
        for _ in range(r - 1):
            x = pow(x, 2, n)  # Square x and reduce it modulo n
            if x == n - 1:
                break  # If x becomes n - 1, this round suggests 'n' may be prime
        else:
            # If none of the checks passed, n is composite
            return 0

    # If all rounds passed, 'n' is probably prime
    return 1


def main():

    # n: The integer to test for primality.
    # k: The number of iterations (tests) to perform.
    # return: 1 if n is probably prime, 0 if n is composite.

    # Ensure there is exactly one command-line argument besides the script name
    if len(sys.argv) != 2:
        print("Only one integer argument is allowed.")
        return

    try:
        # Convert the input argument to an integer
        n = int(sys.argv[1])
    except ValueError:
        # Print error if the input cannot be converted to an integer
        print("The input must be an integer.")
        return

    # number of trials for the Miller-Rabin test
    # we have to balance trade-off between performance and the confidence level of results
    k = 5  # This can be adjusted for higher certainty

    # Miller-Rabin primality test and printing the result
    result = miller_rabin(n, k)
    print(result)






if __name__ == "__main__":
    main()

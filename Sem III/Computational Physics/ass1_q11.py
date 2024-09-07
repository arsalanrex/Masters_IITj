# Bisection Method
# The bisection method is a root-finding method that applies to any continuous function for which one knows two values with opposite signs. The method consists of repeatedly bisecting the interval defined by these values and then selecting the subinterval in which the function changes sign, and therefore must contain a root. The method is a simple example of a bracketing method.

def f(x):
    return x ** 3 - 2 * x - 5


def bisection_method(a, b, tol):
    if f(a) * f(b) >= 0:
        print("The Bisection method fails.")
        return None

    iteration = 0
    a_values = []
    b_values = []
    c_values = []
    tolerances = []

    # Print the table header
    print(f"{'Iteration':<10} | {'a':<8} | {'b':<8} | {'c':<8} | {'f(c)':<8} | {'Tol':<10}")
    print('-' * 60)

    while (b - a) / 2 > tol:
        c = round((a + b) / 2, decimals)

        # Append the current values of a, b, c, and tolerance
        a_values.append(round(a, decimals))
        b_values.append(round(b, decimals))
        c_values.append(round(c, decimals))
        tolerances.append(round((b - a) / 2, decimals))

        # Print current iteration details in tabular form
        iteration += 1
        print(
            f"{iteration:<10} | {round(a, decimals):<8} | {round(b, decimals):<8} | {round(c, decimals):<8} | {round(f(c), decimals):<8} | {tolerances[-1]:<10}")

        # Update the interval based on the sign of f(c)
        if f(c) == 0:  # We've found the exact root
            break
        elif f(a) * f(c) < 0:  # The root lies between a and c
            b = c
        else:  # The root lies between c and b
            a = c

    return c, a_values, b_values, c_values, tolerances


# Given values
a = 2
b = 3
tol = 0.01
decimals = 4

root, a_values, b_values, c_values, tolerances = bisection_method(a, b, tol)
if root is not None:
    print(f"\nThe root is approximately: {root}")

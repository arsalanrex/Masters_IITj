# Newton-Raphson Method
# The Newton-Raphson method is a root-finding method that uses linear approximation to iteratively find the root of a real-valued function. The method requires that the function be differentiable and that the derivative is known. The formula for the method is given by:
# x(n+1) = x(n) - f(x(n)) / f'(x(n))

import sympy as sp

# Define the symbolic variable and function
x = sp.Symbol('x')
f_symbolic = x ** 2 - 4 * x - 7

# Define the derivative of the function symbolically
f_prime_symbolic = sp.diff(f_symbolic, x)

# Convert symbolic function and derivative to numerical functions
f = sp.lambdify(x, f_symbolic, 'math')
f_prime = sp.lambdify(x, f_prime_symbolic, 'math')
print(f"f(x)  = {f_symbolic}")
print(f"f'(x) = {f_prime_symbolic}\n")
print("x(n+1) = x(n) - f(x(n))")
print("               --------")
print("                f'(x(n)\n\n")


def newton_raphson_method(x0, tol):
    # Compute initial x1
    x1 = x0 - f(x0) / f_prime(x0)

    iteration = 0
    x_values = []
    tolerances = []
    f_prime_values = []

    # Print the table header
    print(f"{'Iteration':<10} | {'x0':<8} | {'x1':<8} | {'f(xn)':<8} | {'f_p(xn)':<8} | {'Tol':<10}")
    print('-' * 60)

    while abs(x1 - x0) > tol:
        # Append the current values of x0, f(x1), f'(x1), and tolerance
        x_values.append(round(x0, decimals))
        tolerances.append(round(abs(x1 - x0), decimals))
        f_prime_values.append(round(f_prime(x1), decimals))

        # Print current iteration details in tabular form
        iteration += 1
        print(
            f"{iteration:<10} | {round(x0, decimals):<8} | {round(x1, decimals):<8} | {round(f(x0), decimals):<8} | {round(f_prime(x0), decimals):<8} | {round(abs(x1 - x0), decimals):<10}")

        # Update x0 and x1 for the next iteration
        x0 = x1
        x1 = x0 - f(x0) / f_prime(x0)

    return x1, x_values, tolerances


# Given values
x0 = 5
tol = 0.01
decimals = 4

root, x_values, tolerances = newton_raphson_method(x0, tol)
if root is not None:
    print(f"\nThe root is approximately: {round(root, decimals)}")

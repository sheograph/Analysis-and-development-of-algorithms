import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize
import random

f_calc = np.zeros(3)


def cubic_func(x):
    f_calc[0] += 1
    return np.power(x, 3)


def absolute_func(x):
    f_calc[1] += 1
    return np.absolute(x - 0.2)


def sin_func(x):
    f_calc[2] += 1
    return x * np.sin(1 / x)


def exhaustive_search(func, bounds, eps):
    i = 0
    x_min = bounds[0]
    func_min = func(x_min)
    for x in np.arange(bounds[0], bounds[1], eps):
        i += 1
        if func(x) < func_min:
            func_min = func(x)
            x_min = x
    return x_min, func_min, i


def dichotomy_search(func, bounds, eps):
    i = 0
    delta = eps / 2
    a = bounds[0]
    b = bounds[1]
    while b - a >= eps:
        i += 1
        x_1 = (a + b - delta) / 2
        x_2 = (a + b + delta) / 2

        if func(x_1) <= func(x_2):
            b = x_2
        else:
            a = x_1

    x_min = np.around((a + b) / 2, decimals=5)
    func_min = func(x_min)
    return x_min, func_min, i


def golden_section_search(func, bounds, eps):
    i = 1
    a = bounds[0]
    b = bounds[1]

    x_1 = a + (3 - np.sqrt(5)) * (b - a) / 2
    x_2 = b + (-3 + np.sqrt(5)) * (b - a) / 2
    func_x1 = func(x_1)
    func_x2 = func(x_2)

    while b - a >= eps:
        if func_x1 <= func_x2:
            b = x_2
            x_2 = x_1
            x_1 = a + (3 - np.sqrt(5)) * (b - a) / 2
            func_x2 = func_x1
            func_x1 = func(x_1)
        else:
            a = x_1
            x_1 = x_2
            x_2 = b + (-3 + np.sqrt(5)) * (b - a) / 2
            func_x1 = func_x2
            func_x2 = func(x_2)
        i += 1

    x_min = np.around((a + b) / 2, decimals=5)
    func_min = func(x_min)
    return x_min, func_min, i


def lab21_plot(eps):
    methods = (exhaustive_search, dichotomy_search, golden_section_search)
    tested_funcs = ((cubic_func, [0, 1]), (absolute_func, [0, 1]), (sin_func, [0.01, 1]))
    point_sizes = [100, 50, 25]
    for ind, tested_func in enumerate(tested_funcs):
        counts_ = []
        iterations = []
        for point_size, method in enumerate(methods):
            x_min, func_min, i = method(tested_func[0], tested_func[1], eps)
            print(str(method.__name__) + ' for ' + str(tested_func[0].__name__) + ' called ' + str(
                int(f_calc[ind])) + ' times for ' + str(i) + ' iterations')
            print(str(method.__name__) + ' x_min: ' + str(x_min))
            print(str(method.__name__) + ' func_min(x_min): ' + str(func_min) + '\n')
            counts_.append(f_calc[ind])
            iterations.append(i)
            f_calc[ind] = 0
            plt.scatter(x_min, func_min, label=str(method.__name__), s=point_sizes[point_size])
        x = np.arange(tested_func[1][0], tested_func[1][1] + eps, eps)
        y = np.array([tested_func[0](x_i) for x_i in x])
        plt.plot(x, y)
        plt.xlabel('x')
        plt.ylabel(str(tested_func[0].__name__) + '(x)')
        plt.legend()
        plt.savefig(str(tested_func[0].__name__))
        plt.show()

        plt.bar([method.__name__ for method in methods], counts_)
        plt.xlabel('Method')
        plt.ylabel('Function calls')
        plt.title('f-calculations for ' + str(tested_func[0].__name__) + '(x)')
        plt.savefig(str(tested_func[0].__name__) + ' bars')
        plt.show()

        plt.bar([method.__name__ for method in methods], iterations)
        plt.xlabel('Method')
        plt.ylabel('Iterations')
        plt.title('Number of iterations for ' + str(tested_func[0].__name__) + '(x)')
        plt.savefig(str(tested_func[0].__name__) + ' bars')
        plt.show()



def linear_approximant(x, a, b):
    return a * x + b


def rational_approximant(x, a, b):
    return a / (1 + b * x)


def LSE_lin(*args):   # Least Squares Error
    a = args[0][0]
    b = args[0][1]
    xdata = [linear_approximant(x, a, b) for x in x_k]
    return np.sum([(xdata[i] - y_k[i]) * (xdata[i] - y_k[i]) for i in range(len(y_k))])


def LSE_rat(*args):
    a = args[0][0]
    b = args[0][1]
    xdata = [rational_approximant(x, a, b) for x in x_k]
    return np.sum([(xdata[i] - y_k[i]) * (xdata[i] - y_k[i]) for i in range(len(y_k))])


def lab22_plot(N, eps, alpha, beta, x_k, y_k):
    res1_lin = scipy.optimize.brute(LSE_lin, ranges=(slice(0, 1, 1/(N+1)), (slice(0, 1, 1/(N+1)))))
    res2_lin = scipy.optimize.minimize(LSE_lin, [0.5, 0.5], method='CG', options={'eps': eps})
    res3_lin = scipy.optimize.minimize(LSE_lin, [0.5, 0.5], method='Nelder-Mead')

    plt.plot(x_k, y_k, 'o')
    plt.plot(x_k, [linear_approximant(x, alpha, beta) for x in x_k], label='Generating line')
    plt.plot(x_k, [linear_approximant(x, res1_lin[0], res1_lin[1]) for x in x_k], label='Exhaustive search')
    plt.plot(x_k, [linear_approximant(x, res2_lin.x[0], res2_lin.x[1]) for x in x_k], label='Gauss')
    plt.plot(x_k, [linear_approximant(x, res3_lin.x[0], res3_lin.x[1]) for x in x_k], label='Nelder-Mead')
    plt.title('Linear approximation')
    plt.legend()
    plt.savefig('Linear approximation')
    plt.show()

    res1_rat = scipy.optimize.brute(LSE_rat, ranges=(slice(0, 1, 1/(N+1)), (slice(0, 1, 1/(N+1)))))
    res2_rat = scipy.optimize.minimize(LSE_rat, [0.5, 0.5], method='CG', options={'eps': eps})
    res3_rat = scipy.optimize.minimize(LSE_rat, [0.5, 0.5], method='Nelder-Mead')

    plt.plot(x_k, y_k, 'o')
    plt.plot(x_k, [linear_approximant(x, alpha, beta) for x in x_k], label='Generating line')
    plt.plot(x_k, [rational_approximant(x, res1_rat[0], res1_rat[1]) for x in x_k], label='Exhaustive search')
    plt.plot(x_k, [rational_approximant(x, res2_rat.x[0], res2_rat.x[1]) for x in x_k], label='Gauss')
    plt.plot(x_k, [rational_approximant(x, res3_rat.x[0], res3_rat.x[1]) for x in x_k], label='Nelder-Mead')
    plt.title('Rational approximation')
    plt.legend()
    plt.savefig('Rational approximation')
    plt.show()


N = 100
eps = 0.001
alpha = random.random()
beta = random.random()
noise = np.random.normal(0, 1, N + 1)
x_k = np.array([k / N for k in range(N + 1)])
y_k = np.array([alpha * x_k[k] + beta + noise[k] for k in range(len(x_k))])


# lab21_plot(eps)
lab22_plot(N, eps, alpha, beta, x_k, y_k)
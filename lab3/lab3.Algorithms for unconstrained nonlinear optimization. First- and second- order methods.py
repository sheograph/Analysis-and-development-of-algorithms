import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import scipy.optimize
matplotlib.use('TkAgg')
import random

f_calc = np.zeros(3)


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


def linear_approximant(x, a, b):
    return a * x + b


def rational_approximant(x, a, b):
    return a / (1 + b * x)


def LSE_lin(*args):   # Least Squares Error
    try:
        a = args[0][0]
        b = args[0][1]
    except:
        a = args[0]
        b = args[1]
    xdata = [linear_approximant(x, a, b) for x in x_k]
    return np.sum([(xdata[i] - y_k[i]) * (xdata[i] - y_k[i]) for i in range(len(y_k))])


def LSE_rat(*args):
    try:
        a = args[0][0]
        b = args[0][1]
    except:
        a = args[0]
        b = args[1]
    xdata = [linear_approximant(x, a, b) for x in x_k]
    xdata = [rational_approximant(x, a, b) for x in x_k]
    return np.sum([(xdata[i] - y_k[i]) * (xdata[i] - y_k[i]) for i in range(len(y_k))])


def exhaustive_search(func):
    a = np.linspace(0, 1, 101)
    b = np.linspace(0, 1, 101)
    a_best, b_best = None, None
    min_error = 10 ** 6
    for i in range(len(a)):
        print(i)
        for j in range(len(b)):
            err = func(a[i], b[j])
            if err < min_error:
                min_error = err
                a_best = a[i]
                b_best = b[j]
    return a_best, b_best, min_error


def gauss(func, eps):
    a = 1/2
    b = 1/2
    a_grid = np.linspace(0, 1, 101)
    b_grid = np.linspace(0, 1, 101)
    min_f = func(a, b)
    step = 1
    while step > eps/2:
        a_best = a
        b_best = b
        for a_ in a_grid:
            err = func(a_, b)
            if err < min_f:
                min_f = err
                a_best = a_
        step = abs(a - a_best)
        a = a_best
        for b_ in b_grid:
            err = func(a, b_)
            if err < min_f:
                min_f = err
                b_best = b_
        step = min(step, abs(b - b_best))
        b = b_best
    return a, b, min_f


def lab22_plot(N, eps, alpha, beta, x_k, y_k):
    res1_lin = exhaustive_search(LSE_lin)
    res2_lin = gauss(LSE_lin, eps)
    res3_lin = scipy.optimize.minimize(LSE_lin, [0.5, 0.5], method='Nelder-Mead')

    plt.plot(x_k, y_k, 'o')
    plt.plot(x_k, [linear_approximant(x, alpha, beta) for x in x_k], label='Generating line')
    plt.plot(x_k, [linear_approximant(x, res1_lin[0], res1_lin[1]) for x in x_k], label='Exhaustive search')
    plt.plot(x_k, [linear_approximant(x, res2_lin[0], res2_lin[1]) for x in x_k], label='Gauss')
    plt.plot(x_k, [linear_approximant(x, res3_lin.x[0], res3_lin.x[1]) for x in x_k], label='Nelder-Mead')
    plt.title('Linear approximation')
    plt.legend()
    plt.savefig('Linear approximation')
    plt.show()

    res1_rat = exhaustive_search(LSE_rat)
    res2_rat = gauss(LSE_rat, eps)
    res3_rat = scipy.optimize.minimize(LSE_rat, [0.5, 0.5], method='Nelder-Mead')

    plt.plot(x_k, y_k, 'o')
    plt.plot(x_k, [linear_approximant(x, alpha, beta) for x in x_k], label='Generating line')
    plt.plot(x_k, [rational_approximant(x, res1_rat[0], res1_rat[1]) for x in x_k], label='Exhaustive search')
    plt.plot(x_k, [rational_approximant(x, res2_rat[0], res2_rat[1]) for x in x_k], label='Gauss')
    plt.plot(x_k, [rational_approximant(x, res3_rat.x[0], res3_rat.x[1]) for x in x_k], label='Nelder-Mead')
    plt.title('Rational approximation')
    plt.legend()
    plt.savefig('Rational approximation')
    plt.show()



def conjugate_gradient_descent(function, args):
    return scipy.optimize.minimize(
        function,
        x0=(0, 0),
        method='CG',
        args=(args,),
        tol=epsilon
    )

def newtons_method(function, args):
    return scipy.optimize.minimize(
        function,
        x0=(0, 0),
        method='Newton-CG',
        jac=jacobian(function),
        args=(args,),
        tol=epsilon
    )

def levenberg_marquardt_algorithm(function, x, y):
    return scipy.optimize.curve_fit(
        function,
        xdata=x,
        ydata=y,
        method='lm'
    )

def lab3_plot(func, str, N, eps, alpha, beta, x_k, y_k):
    res1_lin = scipy.optimize.minimize(func, x0=(0, 0), method='BFGS', tol=eps)
    res2_lin = scipy.optimize.minimize(func, [0.5, 0.5], method='CG', options={'eps': eps})
    res3_lin = scipy.optimize.minimize(func, [0.5, 0.5], method='Nelder-Mead')

    plt.plot(x_k, y_k, 'o')
    plt.plot(x_k, [linear_approximant(x, alpha, beta) for x in x_k], label='Generating line')
    plt.plot(x_k, [linear_approximant(x, res1_lin.x[0], res1_lin.x[1]) for x in x_k], label='Gradient descent')
    plt.plot(x_k, [linear_approximant(x, res2_lin.x[0], res2_lin.x[1]) for x in x_k], label='conjugate_gradient_descent')
    plt.plot(x_k, [linear_approximant(x, res3_lin.x[0], res3_lin.x[1]) for x in x_k], label='Nelder-Mead')
    plt.title(str)
    plt.legend()
    plt.savefig(str)
    plt.show()



N = 100
eps = 0.001
alpha = random.random()
beta = random.random()
noise = np.random.normal(0, 1, N + 1)
x_k = np.array([k / N for k in range(N + 1)])
y_k = np.array([alpha * x_k[k] + beta + noise[k] for k in range(len(x_k))])

# lab22_plot(N, eps, alpha, beta, x_k, y_k)

lab3_plot(LSE_lin, 'Linear approximation', N, eps, alpha, beta, x_k, y_k)
lab3_plot(LSE_rat, 'Rational approximation', N, eps, alpha, beta, x_k, y_k)




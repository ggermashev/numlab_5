import numpy as np
from matplotlib import pyplot as plt

start = 0.0
end = 1.0
inits = [0.0, 4/3]

h = 0.05
count = int((end-start) / h) + 1
nodes = np.linspace(start, end, count)

rude_h = 0.1
rude_count = int((end-start) / rude_h) + 1
rude_nodes = np.linspace(start, end, rude_count)


def matrix(m, x):
    return np.vstack([m[1], (x * np.cosh(x) ** 2 - np.tanh(x)) / 3 - np.cosh(x) ** 2 * m[0] + np.tanh(x) * m[1]])


def euler(nodes, inits):
    h = nodes[1] - nodes[0]
    ans = np.zeros((len(inits), len(nodes)))
    ans[:, 0] = inits
    for i in range(1, len(nodes)):
        ans[:, [i]] = ans[:, [i-1]] + h * matrix(ans[:, i - 1], nodes[i-1])
    return ans


def runge_kutta(nodes, inits):
    h = nodes[1] - nodes[0]
    ans = np.zeros((len(inits), len(nodes)))
    ans[:, 0] = inits
    for i in range(0, len(nodes) - 1):
        k1 = matrix(ans[:, i], nodes[i])
        k2 = matrix(ans[:, [i]] + 0.5 * k1 * h, nodes[i] + 0.5 * h)
        k3 = matrix(ans[:, [i]] + 0.5 * k2 * h, nodes[i] + 0.5 * h)
        k4 = matrix(ans[:, [i]] + k3 * h, nodes[i] + h)
        ans[:, [i+1]] = ans[:, [i]] + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
    return ans


def adams(nodes, inits):
    h = nodes[1] - nodes[0]
    ans = np.zeros((len(inits), len(nodes)))
    ans[:, 0:3] = runge_kutta(nodes[:3], inits)
    for i in range(2, len(nodes) - 1):
        k1 = matrix(ans[:, i], nodes[i])
        k2 = matrix(ans[:, i - 1], nodes[i - 1])
        k3 = matrix(ans[:, i-2], nodes[i-2])
        ans[:, [i+1]] = ans[:, [i]] + h / 12 * (23 * k1 - 16 * k2 + 5 * k3)
    return ans


def answer(x):
    return np.sin(np.sinh(x)) + x/3


# solution in [euler, runge_kutta, adams]
def get_error(nodes, inits, solution):
    return np.max(np.abs(answer(nodes) - solution(nodes, inits)[0]))


ans = [answer(x) for x in nodes]
euler_ans = euler(nodes, inits)
runge_kutta_ans = runge_kutta(nodes, inits)
adams_ans = adams(nodes, inits)

plt.plot(nodes, ans, label='answer')
plt.plot(nodes, euler_ans[0], label='euler')
plt.plot(nodes, runge_kutta_ans[0], label='runge-kutta')
plt.plot(nodes, adams_ans[0], label='adams')
plt.legend()
plt.show()

errors_euler = []
errors_runge_kutta = []
errors_adams = []
steps = []
steps_count = np.arange(50, 5, -5)
for s_c in steps_count:
    count = s_c
    nodes = np.linspace(start, end, count)
    steps.append(nodes[1] - nodes[0])
    ans = [answer(x) for x in nodes]
    euler_ans = euler(nodes, inits)
    runge_kutta_ans = runge_kutta(nodes, inits)
    adams_ans = adams(nodes, inits)

    errors_euler.append(get_error(nodes, inits, euler))
    errors_runge_kutta.append(get_error(nodes, inits, runge_kutta))
    errors_adams.append(get_error(nodes, inits, adams))

y_x = [x for x in steps]
y_3x = [x ** 3 for x in steps]
y_4x = [x ** 4 for x in steps]

plt.loglog(steps, errors_euler, label='euler errors')
plt.loglog(steps, y_x, label='x')
plt.legend()
plt.show()

plt.loglog(steps, errors_runge_kutta, label='runge-kutta errors')
plt.loglog(steps, y_3x, label='4x')
plt.legend()
plt.show()

plt.loglog(steps, errors_adams, label='adams errors')
plt.loglog(steps, y_4x, label='3x')
plt.legend()
plt.show()


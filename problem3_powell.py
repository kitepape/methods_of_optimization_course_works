import numpy as np
import scipy.optimize as opt
float_ = 1e-10
def powell(x):
    return (x[0]+10*x[1])**2+5*(x[2]-x[3])**2+(x[1]-2*x[2])**2+10*(x[0]-x[3])**4

def grad_powell(x):
    return np.array([
        2*(x[0]+10*x[1])+40*(x[0]-x[3])**3,
        20*(x[0]+10*x[1])+2*(x[1]-2*x[2]),
        10*(x[2]-x[3])-4*(x[1]-2*x[2]),
        -10*(x[2]-x[3])-40*(x[0]-x[3])**3
    ])

def solve_powell():
    x0 = np.array([3, -1, 0, 1])
    res = opt.minimize(powell, x0, method='powell', jac=grad_powell, options={'disp': True})
    print("近似最优解为：", res.x)
    print("函数值为：", powell(res.x))
    print("最优解的梯度为：", grad_powell(res.x))


if __name__ == '__main__':
    solve_powell()
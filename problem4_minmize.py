import numpy as np
import scipy.optimize as opt

def f(x):
    return (x[0]-2)**2+(x[1]-1)**2

def grad_f(x):
    return np.array([
        2*(x[0]-2),
        2*(x[1]-1)
    ])
# 约束条件,其type字典键值有：

constraint = (
    {
        'type': 'ineq',
        'fun': lambda x: -0.25*x[0]**2-x[1]**2+1},
    {
        'type': 'eq',
        'fun': lambda x: x[0]-2*x[1]+1}
)

def solve_f():
    x0 = np.array([0, 0])
    res = opt.minimize(f, x0, jac=grad_f, constraints=constraint,method='SLSQP'
                       options={'disp': True})

    print("近似最优解为：", res.x)
    print("函数值为：", f(res.x))
    print("最优解的梯度为：", grad_f(res.x))

if __name__ == '__main__':
    solve_f()
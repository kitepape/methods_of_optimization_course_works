import numpy as np
import matplotlib.pyplot as plt
eplison1 = 1e-2
eplison2 = 1e-3

def fib(n):
    ls = [1, 1]
    for i in range(1000):
        ls.append(ls[i]+ls[i+1])
    return ls[n]
def f(x):
    result = 100 * ((x[1] - x[0] ** 2) ** 2 + (1 - x[0]) ** 2)
    return result
def grad_f(xk):
    grad_f = np.array([
        400 * xk[0] * (xk[0]**2-xk[1]) + 2 * (xk[0] - 1),
        200 * (xk[1] - xk[0] ** 2)
    ])
    return grad_f

def get_n(length):
    n = 0
    while True:
        if fib(n) < np.linalg.norm(length) / eplison2:
            n += 1
        else:
            break
    return n

def get_next_fib(a0,b0):
    n = get_n(b0-a0)
    a, b, lambdak, miuk, dk = [a0], [b0], [], [], []
    lambdak.append(b[0] - (fib(n)/fib(n+1))*(b0-a0))
    miuk.append(a[0] + (fib(n)/fib(n+1))*(b0-a0))
    k2=-1
    while True:
        k2 += 1
        aa = a[k2]
        lk = lambdak[k2]
        l_f = f(lk)
        bb = b[k2]
        mk = miuk[k2]
        r_f = f(mk)
        if l_f>r_f:
            if np.linalg.norm(b[k2]-lambdak[0])<eplison2:
                flag = 0
                break
            else:
                a.append(lambdak[k2])
                b.append(b[k2])
                length = b[k2+1]-a[k2+1]
                lambdak.append(miuk[k2])
                miuk.append(a[k2+1]+(fib(n-k2)/fib(n-k2+1))*length)
        else:
            if np.linalg.norm(miuk[k2]-a[k2])<eplison2:
                flag = 1
                break
            else:
                a.append(a[k2])
                b.append(miuk[k2])
                length = b[k2+1]-a[k2+1]
                miuk.append(lambdak[k2])
                lambdak.append(b[k2+1]-(fib(n-k2)/fib(n-k2+1))*length)

    nextx = miuk[k2] if flag==0 else lambdak[k2]
    return nextx
def get_next_wolfe(x0,d0,c=1e-4,rho=0.8):
    alpha = 0.1
    while True:
        if f(x0+alpha*d0)>f(x0)+c*alpha*np.dot(grad_f(x0),d0):
            alpha *= rho
        else:
            break
    return x0 + alpha*d0
def gradown():
    k = -1
    x0 = np.array([-1.2, 1])
    x,d = [x0],[]
    d0 = -grad_f(x0)
    print(f"初始坐标为{x0}")
    print(f"初始梯度为{d0}")
    print(f"初始函数值为{f(x0):.4f}")

    while True:
        k+=1
        gk =grad_f(x[k])
        if np.linalg.norm(gk)<eplison1:
            break
        else:
            d.append(-gk)
            a0 = x[k]
            b0 = x[k]+100*d[k]
            nextx = get_next_fib(a0,b0)
            x.append(nextx)
            print(f"第{k+1}次迭代，坐标为：{x[k+1]}")
            print(f"函数值为：{f(x[k+1]):.6f}")
    print(f"近似最优解为：{x[-1]}\n"
          f"近似最优值为：{f(x[-1]):.6f}")
    plt.plot([x[i][0] for i in range(len(x))], [x[i][1] for i in range(len(x))], 'r-o')
def wolfe():
    k = -1
    x0 = np.array([-1.2, 1])
    x,d = [x0],[]
    d0 = -grad_f(x0)
    print(f"初始坐标为{x0}")
    print(f"初始梯度为{d0}")
    print(f"初始函数值为{f(x0):.6f}")

    while True:
        k+=1
        gk =grad_f(x[k])
        if np.linalg.norm(gk)<eplison1 or k>10000:
            break
        else:
            d.append(-gk)
            nextx = get_next_wolfe(x[k],10*d[k])
            x.append(nextx)
            print(f"第{k+1}次迭代，坐标为：{x[k+1]}")
            print(f"梯度为：{grad_f(x[k+1])}")
            print(f"函数值为：{f(x[k+1]):.4f}")
        print(f"近似最优解为：{x[-1]}\n"
              f"近似最优值为：{f(x[-1]):.6f}")
    plt.plot([x[i][0] for i in range(len(x))], [x[i][1] for i in range(len(x))], 'r-o')
def problem2(x):
    if x == 0:
        gradown()
    else:
        wolfe()
    plt.show()

if __name__ == '__main__':
    problem2(1)
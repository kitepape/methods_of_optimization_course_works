import matplotlib.pyplot as plt
import numpy as np
import scipy.optimize as opt

float_ = 1e-6

class Active_set:
    def __init__(self,G=None,g=None,A=None,b=None,x0=None,func=None):
        self.G = G
        self.g = g
        self.A = A
        self.b = b
        self.x = [x0]
        self.func = func
    def solve_eq(self,Gt,gt,At,bt):
        if not (At.shape[0]==1 and At.shape[1]==0):
            KKT_upper = np.hstack((Gt,-At.T))
            KKT_lower = np.hstack((-At,np.zeros_like(np.eye(At.shape[0]))))
            KKT = np.matrix(np.vstack((KKT_upper,KKT_lower)))
            right = -np.matrix(np.hstack((gt, np.matrix(bt)))).T
        else:
            KKT = Gt
            right = -gt.T
        res = KKT.I @ right
        d = res[:Gt.shape[0], :]
        lam = res[Gt.shape[0]:, :]
        for i in range(d.shape[0]):
            if -float_<d[i,0]<float_:
                d[i,0] =0
        return d,lam

    def solve(self):
        m,n = self.A.shape
        W = []
        tmp = self.A @ self.x[0]
        k = -1
        for i in range(m):
            tmp1 = tmp[i,0]
            tmp2 = self.b[i,0]
            if np.all(tmp1==tmp2):
                W.append(i)
        c=0
        while True:
            k+=1
            print("第{}次迭代，解向量为：{}".format(
                k, list(map(lambda x: eval(f"{x[0, 0]:.3f}"), self.x[k]))))
            print(f"工作集为：{W}")
            print(f"函数值为：{self.func(self.x[k])}")
            if k>100:break
            At = np.matrix(np.array([self.A[i] for i in W]))
            bt = np.zeros(At.shape[0])
            d,lam = self.solve_eq(self.G,self.g(self.x[k]),At,bt)
            if np.all(-float_<d) and np.all(d<float_):
                if np.all(lam>0):

                    break
                else:
                    self.x.append(self.x[k])
                    neg = W[np.argmin(lam)]
                    W.remove(neg)
                    continue
            else:
                wait = [((self.b[i,0]-self.A[i] @ self.x[k])/(self.A[i] @ d))[0,0]
                        if i not in W and ((self.A[i] @ d)[0,0]<0) else 101 for i in range(m)]
                if (101>min(wait)>=1):
                    ahpla = 1
                else:
                    ahpla = min(wait)
                if -float_<ahpla<float_: ahpla=0
                if ahpla==0: c+=1
                self.x.append(self.x[k]+ahpla*d)
                if ahpla<1:
                    if c<2:
                        tmp1 = [wait[i] for i in range(len(wait))]
                        tmp0 = np.array(tmp1)
                        addr = np.argmin(tmp0)
                        W.append(addr)
                    else:
                        tmp1 = [wait[i] for i in range(len(wait))]
                        tmp0 = np.array(tmp1)
                        addr = np.argmin(tmp0)
                        tmp1 = [wait[i] if i != addr else 101 for i in range(len(wait))]
                        tmp0 = np.array(tmp1)
                        addr = np.argmin(tmp0)
                        W.append(addr)
                        c = 0


def test():
    func = lambda x: (x[0] - 1) ** 2 + (x[1] - 2.5) ** 2
    G = np.matrix([
        [2, 0],
        [0, 2]
    ])
    g = lambda x: np.matrix([2 * (x[0, 0] - 1), 2 * (x[1, 0] - 2.5)])
    A = np.matrix([
        [1, -2],
        [-1, -2],
        [-1, 2],
        [1, 0],
        [0, 1]
    ])
    b = np.matrix([[-2], [-6], [-2], [0], [0]])
    x = np.matrix([[2],
                   [0]])
    act = Active_set(G, g, A, b, x, func)
    act.solve()
def pro5():
    func = lambda x: (x[0, 0] + 6) ** 2 + (x[0, 0] + x[1, 0] + 6) ** 2 + (x[0, 0] - x[2, 0] + 8) ** 2
    G = np.matrix([
        [6, 2, -2],
        [2, 2, 0],
        [-2, 0, 2]
    ])
    g = lambda x: np.matrix([
        2 * (x[0, 0] + 6) + 2 * (x[0, 0] + x[1, 0] + 6) + 2 * (x[0, 0] - x[2, 0] + 8),
        2 * (x[0, 0] + x[1, 0] + 6),
        -2 * (x[0, 0] - x[2, 0] + 8)
    ])
    A = np.matrix([
        [0, -1, -1],
        [-1, 0, -1],
        [-1, -1, 0],
        [-1, -1, -1],
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 1]
    ])
    b = np.matrix([[-7], [-7], [-9], [-21], [0], [0], [0]])
    x = np.matrix([[4.5], [4.5], [2.5]])
    x1 = np.matrix([[0], [0], [0]])
    act = Active_set(G, g, A, b, x, func)
    act.solve()
if __name__ == '__main__':
    pro5()



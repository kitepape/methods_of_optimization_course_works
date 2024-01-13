import numpy as np
def normal_simplex(c, A, b):
    print("本次求解使用一般单纯形法")
    row, col = A.shape
    cj = np.hstack((c, np.zeros(row)))
    k = 0
    B = [i + col for i in range(row)]
    table = np.concatenate((A, np.eye(row), b.reshape((row,1))),axis=1)
    c_b = np.array([cj[i] for i in B])
    sigmoid = np.array([cj[i] - c_b @ table[:, i] for i in range(row+col)])
    while True:
        k+=1
        standards = []
        if np.all(sigmoid>=0):
            print(f"最优解基变量：   {B}")
            x_ = [table[B.index(i), row+col] if i in B else 0 for i in range(cj.shape[0])]
            print(f"最优解基变量取值：{x_[:col]}")
            print(f"最优解函数值：{cj @ x_:.2f}")
            break
        in_i = np.argmin(sigmoid)
        if np.all(table[:,col]<0):
            print("没有最优解")
            return
        for i in range(row):
            if (table[i,in_i]<=0):
                continue
            else:
                value = table[i,table.shape[1]-1] / table[i,in_i]
                standards.append((B[i],value))
        standards.sort(key=lambda x:x[1])
        out_j = standards[0][0]
        B = [i if i!=out_j else in_i for i in B]

        main_var = table[B.index(in_i), in_i]
        table[B.index(in_i),:] /= main_var
        for i in range(row):
            if i!=B.index(in_i):
                table[i,:] -= table[i,in_i] * table[B.index(in_i),:]
        c_b =np.array([sigmoid[i] for i in B])
        sigmoid -= c_b @ table[:,:row+col]
        print(f"第{k}次迭代基变量：   {B}")
        x_ = [table[B.index(i), row+col] if i in B else 0 for i in range(row+col)]
        print(f"第{k}次迭代基变量取值：{x_[:col]}")
        print(f"第{k}次迭代函数值：{cj @ x_:.2f}")
    print()
def dual_simplex(c, A, b):
    print("本次求解使用对偶单纯形法")
    row, col = A.shape
    cj = np.hstack((c, np.zeros(row)))
    k = 0
    B = [i + col for i in range(row)]
    table = np.concatenate((A, np.eye(row), b.reshape((row, 1))), axis=1)
    c_b = np.array([cj[i] for i in B])
    sigmoid = np.array([cj[i] - c_b @ table[:, i] for i in range(row + col)])
    while True:
        k += 1
        standards = []
        if np.all(table[:,row+col] >= 0):
            print(f"最优解基变量：   {B}")
            x_ = [table[B.index(i), row + col] if i in B else 0 for i in range(cj.shape[0])]
            print(f"最优解基变量取值：{x_[:col]}")
            print(f"最优解函数值：{cj @ x_:.2f}")
            break

        out_j = B[np.argmin(table[:,col+row])]
        for i in range(row):
            if table[B.index(out_j),i]<0:
                value = abs(sigmoid[i]/table[B.index(out_j),i])
                standards.append((i,value))
        standards.sort(key=lambda x:x[1])
        in_i = standards[0][0]
        B = [i if i != out_j else in_i for i in B]

        main_var = table[B.index(in_i), in_i]
        table[B.index(in_i), :] /= main_var
        for i in range(row):
            if i != B.index(in_i):
                table[i, :] -= table[i, in_i] * table[B.index(in_i), :]
        c_b = np.array([sigmoid[i] for i in B])
        sigmoid -= c_b @ table[:, :row + col]
        print(f"第{k}次迭代基变量：   {B}")
        x_ = [table[B.index(i), row + col] if i in B else 0 for i in range(row + col)]
        print(f"第{k}次迭代基变量取值：{x_[:col]}")
        print(f"第{k}次迭代函数值：{cj @ x_:.2f}")
    print()
def simplex_method(c, A, b):
    """
    单纯形法求解线性规划问题
    :param c: 目标函数系数向量
    :param A: 约束条件系数矩阵
    :param b: 约束条件值向量
    :return: 无
    """
    if np.all(b>0):
        normal_simplex(c,A,b)
    elif np.all(c>0):
        dual_simplex(c,A,b)
    else:
        print("单纯形法与对偶单纯形法均不能应用于该问题")
def test():
    c = np.array([-2,-3])
    A = np.array([
        [-1,1],
        [-2,1],
        [4,1],
    ])
    b = np.array([3,2,16])

    c1 = np.array([6.8,3])
    A1 = np.array([
        [-2.6,-1],
        [-3.8,-3],
        [-1.6,-1],
        [-6,-10]
    ])
    b1 = np.array([-800,-1000,-100,-6000])
    simplex_method(c,A,b)
    simplex_method(c1,A1,b1)

def problem1():
    c0 = np.array([3,1,1])
    A0 = np.array([
        [2,1,1],
        [1,-1,-1]
    ])
    b0 = np.array([2,-1])
    simplex_method(c0,A0,b0)
    print()
    c1 = np.array([2,3])
    A1 = np.array([
        [1,8],
        [4,0],
        [0,4]
    ])
    b1 = np.array([8, 16, 12])
    simplex_method(c1, A1, b1)
if __name__ == '__main__':
    problem1()


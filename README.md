# methods_of_optimization_course_works

WHU‘s method of optimization course works

2022年大二下半学期武汉大学计算机学院ai专业最优化课程大作业

代码涉及

1）单纯形法，包括教材中的例子，以及作业；

2）rosenbrock函数的斐波那契精确搜索优化和wolfe准则不精确搜索优化，
其中由于rosenbrock函数的性质，以及浮点数计算误差，
斐波那契精确搜索优化过程中函数值可能会小幅上升，但最后会收敛到近似最优解，
然而wolfe准则不精确搜索优化则无法使得该函数下降到最优解；

3）调用scipy库对powell函数进行无约束优化，本代码使用该库提供的'powell'方法；

4）调用scipy库对最小二乘函数进行约束优化，本代码使用该库提供的'SLSQP'方法；

5）凸二次规划的有效集方法，包括教材用例，以及作业。

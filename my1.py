from functions1 import *
"""
Created on Sun Oct 15 15:10:20 2023
By Tian Jiafu
平面问题求解器，可以求解任意四边形平面应力问题
输出x/y/xy三个方向引力
"""


################ 前处理(参数输入) ######################
E = 200*10**9  # 材料.弹性模量(Pa)
prxy = 0.3     # 材料.泊松比
dens = 7850    # 材料.体密度(Kg/m^3)

t = 0.01       # 实常数.板厚(m)

### 网格划分（平面四边形四结点单元） ###
Pk = np.array([[0,-3],           # 单位(m)
               [15,0],
               [15,3],
              [0,3]])
N1 = 20                        # x方向划分单元个数
N2 = 4                         # y方向划分单元个数
Nl,El,x_coor,y_coor = mesh_rec(Pk,N1,N2,True)

### 荷载输入 ###
g = 0                          # 重力加速度(N/Kg)
Qx = np.array([[0,0,0,0]])  # 边界面力,线性变化：[fxi,fxj,Ni,Nj] (Pa)
Qy = np.array([[-20*10**6,-20*10**6,5,105]])  # 边界面力,线性变化：[fyi,fyj,Ni,Nj] (Pa)
### 位移边界条件输入 ###
DoN = np.array([[1,1,1,0,0],
               [2,1,1,0,0],
                [3,1,1,0,0],
                [4,1,1,0,0],
                [5,1,1,0,0]])

##################### 计算CAE ##################################
### D矩阵，材料本构关系（平面应力问题） ###
D = E/(1-prxy**2)*np.array([[1,prxy,0],
                            [prxy,1,0],
                            [0,0,(1-prxy)/2]])


### 系统总体刚度矩阵 ###
K = global_K(El,Nl,D,t)

### 系统总体荷载列阵 ###
Fsx = boundary_Fx(Qx, Nl, Pk, N2,t)   # x方向面力
Fsy = boundary_Fy(Qy, Nl, Pk, N2,t)   # x方向面力
F = global_F(Fsx,Fsy, El, Nl, t, dens, g)

#### 求解 ####
Disall = solution(K,F,DoN)
print(Disall.max())
print(Disall.min())
############################## 后处理 #######################
#### 变形图 ####
alpha = 1           # 变形显示放大系数
x_final,y_final = defor_show(Disall,x_coor,y_coor,N1,N2,alpha)
Nl_final = np.hstack((x_final,y_final))
#### 应力云图 ####
stype = 'xx'
fignum = 2
shownum = False
stress_show(Disall,El,Nl,Nl_final,D,stype,fignum)
line_show(x_final,y_final,N1+1,N2+1)

stype = 'yy'
fignum = 3
shownum = False
stress_show(Disall,El,Nl,Nl_final,D,stype,fignum)
line_show(x_final,y_final,N1+1,N2+1)

stype = 'xy'
fignum = 4
shownum = False
stress_show(Disall,El,Nl,Nl_final,D,stype,fignum)
line_show(x_final,y_final,N1+1,N2+1)

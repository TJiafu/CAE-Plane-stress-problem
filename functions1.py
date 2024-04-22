import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.colors as mcolors


def mesh_rec(Pk, N1, N2,show):

    ### Notes ###
    N1 += 1
    N2 += 1
    x_coor = np.zeros([N1 * N2, 1])
    y_coor = np.zeros([N1 * N2, 1])
    index = np.zeros([N1 * N2, 1])
    Pk_side = np.zeros([2, 2])
    for i in range(N1):
        Pk_side[0, :] = [(Pk[1, 0] - Pk[0, 0]) / (N1 - 1) * i + Pk[0, 0],
                         (Pk[1, 1] - Pk[0, 1]) / (N1 - 1) * i + Pk[0, 1]]
        Pk_side[1, :] = [(Pk[2, 0] - Pk[3, 0]) / (N1 - 1) * i + Pk[3, 0],
                         (Pk[2, 1] - Pk[3, 1]) / (N1 - 1) * i + Pk[3, 1]]
        for j in range(N2):
            x_coor[N2 * i + j] = (Pk_side[1, 0] - Pk_side[0, 0]) / (N2 - 1) * j + Pk_side[0, 0]
            y_coor[N2 * i + j] = (Pk_side[1, 1] - Pk_side[0, 1]) / (N2 - 1) * j + Pk_side[0, 1]
            index[N2 * i + j] = N2 * i + j + 1
    Nl = np.hstack((index, x_coor, y_coor))

    ### Elements ###
    El = np.zeros([(N1-1)*(N2-1),4])   # 平面四边行四节点单元
    n = 0
    for i in range(N1-1):
        for j in range(N2-1):
            El[n,:] = np.array([index[i*N2+j],index[(i+1)*N2+j],index[(i+1)*N2+j+1],index[i*N2+j+1]]).T
            n += 1
    ### Show ###
    fig_num = 1
    shownum = True
    if show:
        mash_show(x_coor, y_coor, N1, N2, fig_num, shownum)
    return Nl,El,x_coor,y_coor


def mash_show(x_coor,y_coor,N1,N2,fig_num,shownum):
    mpl.use("TkAgg")
    plt.figure(fig_num)  # ,figsize=(10,N2/N1*10)
    if shownum:
        n = 1
        for i in range(N1):
            for j in range(N2):
                plt.annotate(n,xy=(x_coor[N2*i+j],y_coor[N2*i+j]))
                n += 1
        m = 1
        for i in range(N1-1):
            for j in range(N2-1):
                plt.annotate(m, xy=((x_coor[(i+1)*N2+j] + x_coor[N2 * i + j])/2,
                                    (y_coor[i*N2+j+1] + y_coor[N2 * i + j])/2))
                m += 1
    for i in range(N1-1):
        for j in range(N2-1):
            plt.plot(np.array([x_coor[N2 * i + j],x_coor[(i+1)*N2+j]]),
                     np.array([y_coor[N2 * i + j],y_coor[(i+1)*N2+j]]),c="red")
            plt.plot(np.array([x_coor[(i+1)*N2+j+1],x_coor[(i+1)*N2+j]]),
                     np.array([y_coor[(i+1)*N2+j+1],y_coor[(i+1)*N2+j]]),c="red")
            plt.plot(np.array([x_coor[(i+1)*N2+j+1],x_coor[i*N2+j+1]]),
                     np.array([y_coor[(i+1)*N2+j+1],y_coor[i*N2+j+1]]),c="red")
            plt.plot(np.array([x_coor[i*N2+j+1],x_coor[N2 * i + j]]),
                     np.array([y_coor[i*N2+j+1],y_coor[N2 * i + j]]),c="red")
    plt.axis("equal")
    plt.show()


def element_K(Nl,nl,D,t):            # nl:某单元的节点号，eg：[1,2,4,5]
    E_coor = np.zeros([4, 2])
    Ke = np.zeros([8, 8])
    for k in range(4):
        E_coor[k, :] = Nl[int(nl[k] - 1), 1:]
    Guess_point = np.array([-(3/5)**0.5,0,(3/5)**0.5])
    Guess_weight = np.array([5,8,5])/9
    for p,a in enumerate(Guess_point):
        for q,b in enumerate(Guess_point):
            dNab = np.array([[b/4 - 1/4, 1/4 - b/4, b/4 + 1/4, -b/4 - 1/4],
                             [a/4 - 1/4, -a/4 - 1/4, a/4 + 1/4, 1/4 - a/4]])
            J = dNab @ E_coor  # 雅可比矩阵
            invJ = np.linalg.inv(J)
            B = np.zeros([3,8])       # 应变矩阵
            for m in range(4):
                dNxyi = invJ@dNab[:,m]
                B[:,m*2:m*2+2] = np.array([[dNxyi[0],0],
                                           [0,dNxyi[1]],
                                           [dNxyi[1],dNxyi[0]]])
            Fab = B.T@D@B*t*np.linalg.det(J)      # 8x8
            Ke = Ke + Guess_weight[p]*Guess_weight[q]*Fab
    return Ke


def global_K(El,Nl,D,t):  # 参数：单元位置矩阵、结点坐标、本构关系矩阵、板厚
    NoE = np.size(El, 0)  # 单元数
    NoN = np.size(Nl,0)   # 节点数
    K = np.zeros([NoN*2,NoN*2])
    ### 单元刚度矩阵组装 ###
    for i in range(NoE):
        nl = El[i,:]
        Ke = element_K(Nl,nl,D,t)
        print(i)
        for p in range(4):
            for q in range(4):
                row = nl[p]
                column = nl[q]
                value = Ke[p*2:p*2+2,q*2:q*2+2]
                K[int(row - 1)*2:int(row - 1)*2+2, int(column - 1)*2:int(column - 1)*2+2] \
                    = K[int(row - 1)*2:int(row - 1)*2+2, int(column - 1)*2:int(column - 1)*2+2] + value
    return K


def boby_Fe(dens, g, nl, Nl, t):
    gxy = np.array([[0], [-dens * g]])  # 体力(gx=0,gy=-9.8) (N/m^3)
    E_coor = np.zeros([4, 2])
    G = np.zeros([8, 2])
    for i in range(4):
        E_coor[i, :] = Nl[int(nl[i] - 1), 1:]
        G[2 * i:2 * i + 2] = np.array([[1, 0],
                                       [0, 1]])
    E_coor = np.hstack((E_coor, np.array([[1], [1], [1], [1]])))
    A1 = np.delete(E_coor, 3, axis=0)
    A2 = np.delete(E_coor, 1, axis=0)
    Ae = (np.linalg.det(A1) + np.linalg.det(A2)) / 2
    Pbe = 1 / 4 * t * Ae * G @ gxy
    return Pbe


def boundary_Fx(Qx, Nl, Pk, N2,t):
    NoN = np.size(Nl, 0)       # 节点数
    NoQx = np.size(Qx, 0)      # x方向面力个数
    Qex = np.zeros([4])
    Fsx = np.zeros([NoN*2, 1])
    for i in range(NoQx):    # 第i个x方向面力
        xi = Nl[Qx[i,2]-1,1]
        xj = Nl[Qx[i,3]-1,1]
        yi = Nl[Qx[i,2]-1,2]
        yj = Nl[Qx[i,3]-1,2]
        if xj-xi == 0:
            if xi == (Pk[0,0]+Pk[1,0])/2 or xi == (Pk[3,0]+Pk[2,0])/2:
                Nadd = N2+1
            else:
                Nadd = 1
        else:
            if (yj-yi)/(xj-xi) == (Pk[0,1]-Pk[1,1])/(Pk[0,0]-Pk[1,0]) or \
               (yj-yi)/(xj-xi) == (Pk[3,1]-Pk[2,1])/(Pk[3,0]-Pk[2,0]):
                Nadd = N2 + 1
            else:
                Nadd = 1
        NoE_Q = int((Qx[i,3]-Qx[i,2])/Nadd)    # 第i个x方向面力作用的单元数
        Qadd = (Qx[i,1]-Qx[i,0])/NoE_Q    # 面力沿单元增长的步长
        for j in range(int(NoE_Q)):         # 第i个x方向面力的第j个受力单元
            Qex[2:4] = [Qx[i,2]+Nadd*j,Qx[i,2]+Nadd*j+Nadd]
            Qex[0:2] = [Qx[i,0]+Qadd*j,Qx[i,0]+Qadd*j+Qadd]
            xei = Nl[int(Qex[2] - 1), 1]
            xej = Nl[int(Qex[3] - 1), 1]
            yei = Nl[int(Qex[2] - 1), 2]
            yej = Nl[int(Qex[3] - 1), 2]
            ls = ((yej - yei) ** 2 + (xej - xei) ** 2) ** 0.5
            fei = ls*t*np.array([Qex[0]/2+(Qex[1]-Qex[0])/6])
            fej = ls*t*np.array([Qex[0]/2+(Qex[1]-Qex[0])/3])
            Fsx[int(Qex[2]-1)*2] = Fsx[int(Qex[2]-1)*2] + fei
            Fsx[int(Qex[3]-1)*2] = Fsx[int(Qex[3]-1)*2] + fej
    return Fsx


def boundary_Fy(Qy, Nl, Pk, N2,t):
    NoN = np.size(Nl, 0)       # 节点数
    NoQy = np.size(Qy, 0)      # y方向面力个数
    Qey = np.zeros([4])
    Fsy = np.zeros([NoN*2, 1])
    for i in range(NoQy):    # 第i个y方向面力
        xi = Nl[Qy[i,2]-1,1]
        xj = Nl[Qy[i,3]-1,1]
        yi = Nl[Qy[i,2]-1,2]
        yj = Nl[Qy[i,3]-1,2]
        if xj-xi == 0:
            if xi == (Pk[0,0]+Pk[1,0])/2 or xi == (Pk[3,0]+Pk[2,0])/2:
                Nadd = N2+1
            else:
                Nadd = 1
        else:
            if (yj-yi)/(xj-xi) == (Pk[0,1]-Pk[1,1])/(Pk[0,0]-Pk[1,0]) or \
               (yj-yi)/(xj-xi) == (Pk[3,1]-Pk[2,1])/(Pk[3,0]-Pk[2,0]):
                Nadd = N2 + 1
            else:
                Nadd = 1
        NoE_Q = int((Qy[i,3]-Qy[i,2])/Nadd)    # 第i个y方向面力作用的单元数
        Qadd = (Qy[i,1]-Qy[i,0])/NoE_Q        # 面力沿单元增长的步长
        for j in range(int(NoE_Q)):         # 第i个y方向面力的第j个受力单元
            Qey[2:4] = [Qy[i,2]+Nadd*j,Qy[i,2]+Nadd*j+Nadd]
            Qey[0:2] = [Qy[i,0]+Qadd*j,Qy[i,0]+Qadd*j+Qadd]
            xei = Nl[int(Qey[2] - 1), 1]
            xej = Nl[int(Qey[3] - 1), 1]
            yei = Nl[int(Qey[2] - 1), 2]
            yej = Nl[int(Qey[3] - 1), 2]
            ls = ((yej - yei) ** 2 + (xej - xei) ** 2) ** 0.5
            fei = ls*t*np.array([Qey[0]/2+(Qey[1]-Qey[0])/3])
            fej = ls*t*np.array([Qey[0]/2+(Qey[1]-Qey[0])*2/3])
            Fsy[int(Qey[2]-1)*2+1] = Fsy[int(Qey[2]-1)*2+1] + fei
            Fsy[int(Qey[3]-1)*2+1] = Fsy[int(Qey[3]-1)*2+1] + fej
    return Fsy


def global_F(Fsx,Fsy, El, Nl, t, dens, g):
    NoE = np.size(El, 0)       # 单元数
    NoN = np.size(Nl, 0)       # 节点数
    Fb = np.zeros([NoN*2, 1])
    for i in range(NoE):
        nl = El[i,:]
        Fbe = boby_Fe(dens, g, nl, Nl, t)
        for j in range(4):
            row = nl[j]-1
            value = Fbe[j*2:j*2+2]
            Fb[int(row)*2:int(row)*2+2,:] = Fb[int(row)*2:int(row)*2+2,:] + value
    F = Fb + Fsx + Fsy
    return F


def solution(K,F,DoN):
    #### 乘大数法，改变刚度矩阵奇异性 ####
    m = np.size(DoN,axis=0)
    alpha = np.maximum(K,-K).max() * 10**4
    for i in range(m):
        index = DoN[i,0]
        for j in range(2):
            if DoN[i,j+1] == 1:
                dis = DoN[i,j+3]
                F[int(index - 1)*2+j] = alpha * K[int(index - 1)*2+j, int(index - 1)*2+j] * dis
                K[int(index - 1)*2+j, int(index - 1)*2+j] = alpha * K[int(index - 1)*2+j, int(index - 1)*2+j]
    Disall = np.linalg.inv(K)@F
    return Disall


def defor_show(Disall,x_coor,y_coor,N1,N2,alpha):
    NoN = (N1 + 1) * (N2 + 1)
    Dis_ex = alpha * Disall
    x_final = np.zeros([NoN, 1])
    y_final = np.zeros([NoN, 1])
    for i in range(NoN):
        x_final[i] = Dis_ex[i * 2] + x_coor[i]
        y_final[i] = Dis_ex[i * 2 + 1] + y_coor[i]
    fig_num = 1
    shownum = False
    mash_show(x_final, y_final, N1 + 1, N2 + 1, fig_num, shownum)
    return x_final,y_final


def stressE_const(Disall, nl, Nl, D):
    dis_Nu = np.zeros([4,1])
    dis_Nv = np.zeros([4,1])
    mitr_Ecoor = np.zeros([4,4])
    for i,N in enumerate(nl):
        N -= 1
        dis_Nu[i] = Disall[int(N)*2]
        dis_Nv[i] = Disall[int(N)*2+1]
        mitr_Ecoor[i,:] = [1,Nl[int(N),1],Nl[int(N),2],Nl[int(N),1]*Nl[int(N),2]]
    beta1 = np.linalg.inv(mitr_Ecoor)@dis_Nu
    beta2 = np.linalg.inv(mitr_Ecoor)@dis_Nv
    const = D@np.array([[beta1[1,0]],[beta2[2,0]],[beta1[2,0]+beta2[1,0]]])
    const_x = D@np.array([[0],[beta2[3,0]],[beta1[3,0]]])
    const_y = D@np.array([[beta1[3,0]],[0],[beta2[3,0]]])
    return const,const_x,const_y


def stressX(x,y,const,const_x,const_y):
    return const[0,0] + const_x[0,0]*x + const_y[0,0]*y


def stressY(x,y,const,const_x,const_y):
    return const[1,0] + const_x[1,0]*x + const_y[1,0]*y


def stressXY(x,y,const,const_x,const_y):
    return const[2,0] + const_x[2,0]*x + const_y[2,0]*y


def stress_Normalized(Disall, El, Nl, D,  stype):
    global_z = np.zeros([np.size(El,0),5*5])
    for i,nl in enumerate(El):
        E_coor = np.zeros([4,2])
        for j,N in enumerate(nl):
            E_coor[j,:] = Nl[int(N-1),1:3]
        N,E,x,y = mesh_rec(E_coor, 4, 4, False)
        const,const_x,const_y = stressE_const(Disall, nl, Nl, D)
        if stype == 'xx':
            z = stressX(x, y, const,const_x,const_y)
        elif stype == 'yy':
            z = stressY(x, y, const, const_x, const_y)
        else:
            z = stressXY(x, y, const, const_x, const_y)
        global_z[i,:] = global_z[i,:] + z.T
    print(global_z.max(),global_z.min())
    globalz_Normalized = (global_z - global_z.min()) / (global_z.max() - global_z.min())
    return globalz_Normalized


def truncate_colormap(cmap, minval=0.0, maxval=1.0, n=-1):
    if n == -1:
        n = cmap.N
    new_cmap = mcolors.LinearSegmentedColormap.from_list('trunc({name},{a:.2f},{b:.2f})'
                                                         .format(name=cmap.name,a=minval,b=maxval),
                                                         cmap(np.linspace(minval,maxval,n)))
    return new_cmap


def stress_show(Disall,El,Nl,Nl_final,D,stype,fignum):
    plt.figure(fignum)
    plt.title('stress' + stype)
    globalz_Normalized = stress_Normalized(Disall, El, Nl, D, stype)
    for i, nl in enumerate(El):
        E_coor = np.zeros([4, 2])
        for j, N in enumerate(nl):
            E_coor[j, :] = Nl_final[int(N - 1), :]
        N, E, x, y = mesh_rec(E_coor, 4, 4, False)
        x = x.ravel("C")
        y = y.ravel("C")
        c = globalz_Normalized[i,:]
        cmap = truncate_colormap(plt.get_cmap('jet'),c.min(),c.max())
        plt.tripcolor(x,y,c,cmap=cmap,shading="gouraud")


def line_show(x_coor,y_coor,N1,N2):
    for i in range(N1 - 1):
        for j in range(N2 - 1):
            plt.plot(np.array([x_coor[N2 * i + j], x_coor[(i + 1) * N2 + j]]),
                     np.array([y_coor[N2 * i + j], y_coor[(i + 1) * N2 + j]]), 'k-',linewidth=0.5)
            plt.plot(np.array([x_coor[(i + 1) * N2 + j + 1], x_coor[(i + 1) * N2 + j]]),
                     np.array([y_coor[(i + 1) * N2 + j + 1], y_coor[(i + 1) * N2 + j]]), 'k-',linewidth=0.5)
            plt.plot(np.array([x_coor[(i + 1) * N2 + j + 1], x_coor[i * N2 + j + 1]]),
                     np.array([y_coor[(i + 1) * N2 + j + 1], y_coor[i * N2 + j + 1]]), 'k-',linewidth=0.5)
            plt.plot(np.array([x_coor[i * N2 + j + 1], x_coor[N2 * i + j]]),
                     np.array([y_coor[i * N2 + j + 1], y_coor[N2 * i + j]]), 'k-',linewidth=0.5)
    plt.axis("equal")
    plt.show()

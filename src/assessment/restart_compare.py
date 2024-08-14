import pytools as pt
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from multiprocessing import Pool
import sys,os
from scipy.constants import k, m_e, m_p


arglen = len(sys.argv)
if arglen == 1:
    print("Usage: restart_base.vlsv restart_test.vlsv method [stride] [nprocs] ")
    print(" Calculate VDF measures between two restart file VDFs")
    print(" method is used to annotate outputs and will create a corresponding folder")
    print(" stride (default 1) strides over cellIDs (file layout)")
    print(" nprocs (default 1) is number of cores used via multiprocessing")

    # khi_base = "/scratch/project_2010750/assessment/khi_small/"

    # baseline_folder = "/scratch/project_2010750/assessment/khi_small/control/"
    # baseline_restart = "restart.0000400.2024-08-08_15-03-48.vlsv"

    # test_folder = "/scratch/project_2010750/assessment/khi_small/khi_small_rec/"
    # test_restart = "output_zfp_restart.0000400.2024-08-05_11-49-39.vlsv"

    # method ='mlp'
    # test_restart = "output_mlp_restart.0000400.2024-08-05_11-49-39.vlsv"


elif arglen == 4 or arglen == 5 or arglen == 6:
    baseline_folder=""
    baseline_restart = sys.argv[1]
    test_folder=""
    test_restart = sys.argv[2]
    method = sys.argv[3]
else:
    print("Bad number of input arguments, expected 4..6. Run without arguments for help.")
    sys.exit(1)

try:
    os.mkdir(method)
except FileExistsError:
    pass



if arglen == 5 or arglen == 6:
    stride = max(int(sys.argv[4]),1)
else:
    stride = 1
    
if arglen == 6:
    poolsize = max(int(sys.argv[5]),1)
else:
    poolsize = 1


base = pt.vlsvfile.VlsvReader(baseline_folder+baseline_restart)
test = pt.vlsvfile.VlsvReader(test_folder+test_restart)


cells = base.read_variable("CellID")[::stride]


pop = "proton"
threshold = 1e-17
size = base.get_velocity_mesh_size(pop)
distribution = np.zeros(4 * 4 * 4 * int(size[0]) * int(size[1]) * int(size[2]))
vids = np.arange(4 * 4 * 4 * int(size[0]) * int(size[1]) * int(size[2]))
v0s = base.get_velocity_cell_coordinates(vids, pop)


extent = base.get_velocity_mesh_extent(pop)
dvx = (extent[3] - extent[0]) / (4 * size[0])
dvy = (extent[4] - extent[1]) / (4 * size[1])
dvz = (extent[5] - extent[2]) / (4 * size[2])
dV = dvx*dvy*dvz  
# generate a velocity space 
vids = np.arange(4 * 4 * 4 * int(size[0]) * int(size[1]) * int(size[2]))

#Vertti Tarvus 2024
def epsilon_M2(vk, vs, D_vals, dV, B, n, v0, T_para, T_perp):
    # Graham et al 2021
    distribution = np.zeros(4 * 4 * 4 * int(size[0]) * int(size[1]) * int(size[2]))

    bhat = B/np.linalg.norm(B)
    vperp2hat = np.cross(bhat,v0)/np.linalg.norm(np.cross(bhat,v0))
    vperp1hat = np.cross(vperp2hat, bhat)/np.linalg.norm(np.cross(vperp2hat,bhat))    
    R = np.array([bhat, vperp1hat, vperp2hat])
    vsb = np.matmul(R,vs.T).T
    vsb_mean = np.matmul(R,v0.T).T
    v0_para = vsb_mean[0]
    v0_perp = vsb_mean[1]
    
    vb = np.matmul(R,v0s.T).T
    vT_para = np.sqrt(2 * k * T_para / m_p)
    distribution[vk] = D_vals

    
    distribution[distribution<threshold] = 0
    eM_model_f = n * T_para / (np.pi**1.5 * vT_para**3 * T_perp) * np.exp(
        -(vb[:,0] - v0_para)**2/vT_para**2
        -((vb[:,1] - v0_perp)**2 + vb[:,2]**2)/(vT_para**2 * (T_perp/T_para))
        )
    
    epsilon = np.linalg.norm(distribution - eM_model_f, ord=1)
    epsilon *= dV / (2 * n)
    
    return epsilon


def calc_moments(vs, D_vals, dV, B):
    n = np.sum(np.array(D_vals))*dV
    v0 = np.average(vs, axis=0,weights=np.array(D_vals)*dV)
    # Generate a parallel-perpedicular coordinate basis and corresponding velocity array
    bhat = B/np.linalg.norm(B)
    vperp2hat = np.cross(bhat,v0)/np.linalg.norm(np.cross(bhat,v0))
    vperp1hat = np.cross(vperp2hat, bhat)/np.linalg.norm(np.cross(vperp2hat,bhat))
    
    R = np.array([bhat, vperp1hat, vperp2hat])
    vsb = np.matmul(R,vs.T).T
    vsb_mean = np.matmul(R,v0.T).T
    v0_para = vsb_mean[0]
    v0_perp = vsb_mean[1]

    P_diag = m_p * np.sum((vsb - vsb_mean) * (vsb - vsb_mean) * np.array(D_vals)[:,np.newaxis],axis=0) * dV
    T = np.sum(P_diag) / (3.0 * n * k)
    T_para = P_diag[0] / (n * k)
    T_perp = (P_diag[1] + P_diag[2]) / (2.0 * n * k)
    return n, v0, T, T_para, T_perp


moments_base = []

print("Analysing", len(cells), "cells")
def analyse(args):
    reader, c = args
    vmap = reader.read_velocity_cells(c, pop=pop)
    V = reader.get_velocity_cell_coordinates(list(vmap.keys()),pop=pop)
    f = np.array(list(vmap.values()))
    dV = np.prod(base.get_velocity_mesh_dv())
    B = reader.read_variable('vg_b_vol',c)
    n, v0, T, T_para, T_perp = calc_moments(V, f, dV, B)
    # eM = epsilon_M2(list(vmap.keys()),V, f, dV, B, n, v0, T_para, T_perp)
    # eM = pt.calculations.epsilon_M(reader,c,pop="proton",m=m_p, bulk=None, B=None,
    #             model="bimaxwellian",
    #             normorder=1, norm=2, 
    #             dummy=None)
    return np.array([n, v0[0],v0[1],v0[2],T,T_para,T_perp, 0])

# moments_base = np.array([analyse(base, c) for c in cells])
with Pool(poolsize) as p:
    moments_base = np.array(p.map(analyse, [(base,c) for c in cells]))
# print(cells)
# print(np.array([pt.calculations.epsilon_M(base,c,pop="proton",m=m_p, bulk=None, B=None,
#                 model="bimaxwellian",
#                 normorder=1, norm=2, 
#                 dummy=None) for c in cells]))

# print(cells)
# print(moments_base)
# print(moments_base[:,7])
# sys.exit()

print("Analysing", len(cells), "cells")
with Pool(poolsize) as p:
    moments_test = np.array(p.map(analyse, [(test,c) for c in cells]))

# moments_test = np.array([analyse(test, c) for c in cells])


# for i,c in enumerate(cells):
#     print(i)
#     vmap = test.read_velocity_cells(c, pop=pop)
#     f = np.array(list(vmap.values()))
#     V = test.get_velocity_cell_coordinates(list(vmap.keys()),pop=pop)
#     dV = np.prod(test.get_velocity_mesh_dv())
#     rho = np.sum(f*dV)
#     U = dV*np.sum(V*f[:,np.newaxis],axis = 0)/rho
#     vx = U[0]
#     vy = U[1]
#     vz = U[2]
#     moments_test.append(np.array([rho,vx,vy,vz]))


# print(moments_test)
rho_base = moments_base[:,0]
rho_test = moments_test[:,0]
# rho_diff = rho_test - rho_base

ux_diff = moments_test[:,1] - moments_base[:,1]
uy_diff = moments_test[:,2] - moments_base[:,2]
uz_diff = moments_test[:,3] - moments_base[:,3]


def print_delta(base, test, name):
    diff = test - base
    print(name + " absolute [min(abs(delta)),Lp1,Lp2,max(abs(delta))]:", [np.linalg.norm(diff,lp) for lp in [-np.inf,1,2,np.inf]])
    print(name + " normalized differences [min(abs(ndelta)), max(abs(ndelta))]:          ",[np.linalg.norm(diff/(base+test),lp) for lp in [-np.inf,np.inf]])

    fig,ax = plt.subplots(1,1)
    h,x,y,im = plt.hist2d(base,test,bins=100, norm = 'log', cmap='copper_r')
    # ax.axis("equal")
    p0 = [min(x[0],y[0]),min(x[0],y[0])]
    p1 = [max(x[-1],y[-1]),max(x[-1],y[-1])]
    plt.plot([p0[0],p1[0]],[p0[1],p1[1]], 'k:')
    ax.grid()
    plt.title(method+": "+name)
    ax.set_xlabel("baseline")
    ax.set_ylabel("test")
    plt.colorbar(label="counts")
    plt.savefig(method+"/"+method+"_"+name+"_correlation.png", dpi=150)
    plt.close()


    
print_delta(rho_base, rho_test, "n")
# print("min(abs(delta))","1","2","max(abs(delta))")
# print("rho absolute [min(abs(delta)),Lp1,Lp2,max(abs(delta))]:", [np.linalg.norm(rho_diff,lp) for lp in [-np.inf,1,2,np.inf]])
# print("rho rel [min(abs(ndelta)), max(abs(ndelta))]:          ",[np.linalg.norm(rho_diff/(moments_base[:,0]+moments_test[:,0]),lp) for lp in [-np.inf,np.inf]])

print_delta(moments_base[:,1], moments_test[:,1], 'Ux')
print_delta(moments_base[:,2], moments_test[:,2], 'Uy')
print_delta(moments_base[:,3], moments_test[:,3], 'Uz')
print_delta(moments_base[:,4], moments_test[:,4], 'T')
print_delta(moments_base[:,5], moments_test[:,5], 'T_para')
print_delta(moments_base[:,6], moments_test[:,6], 'T_perp')
# print_delta(moments_base[:,7], moments_test[:,7], 'e_M')


# print(moments_test[:,3])
# print(moments_base[:,3])

# print(moments_test[:,3]-moments_base[:,3])


# print("Ux: ",[np.linalg.norm(ux_diff,lp) for lp in [0,-np.inf,1,2,np.inf]])


# print("Uy: ",[np.linalg.norm(uy_diff,lp) for lp in [0,-np.inf,1,2,np.inf]])
# print("Uz: ",[np.linalg.norm(uz_diff,lp) for lp in [0,-np.inf,1,2,np.inf]])
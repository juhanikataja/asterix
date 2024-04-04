import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from sklearn.mixture import GaussianMixture
from scipy.optimize import curve_fit
from scipy import ndimage

def extract_vdf(file,cid,box=-1):
    import numpy as np
    import pytools as pt
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm
    from matplotlib.colors import LogNorm

    assert(cid>0)
    f = pt.vlsvfile.VlsvReader(file)
    # -- read phase space density
    vcells = f.read_velocity_cells(cid) 
    keys = list(vcells.keys())
    values = list(vcells.values())

    # -- generate a velocity space
    size = f.get_velocity_mesh_size()
    vids = np.arange(4 * 4 * 4 * int(size[0]) * int(size[1]) * int(size[2]))

    # -- put phase space density into array
    dist = np.zeros_like(vids,dtype=float) 
    dist[keys] = values

    # -- sort vspace by velocity
    v = f.get_velocity_cell_coordinates(vids)

    i = np.argsort(v[:,0],kind='stable')
    v = v[i]
    #vids = vids[i]
    dist = dist[i]

    j = np.argsort(v[:,1],kind='stable')
    v = v[j]
    #vids = vids[j]
    dist = dist[j]

    k = np.argsort(v[:,2],kind='stable')
    v = v[k]
    #vids = vids[k]
    dist = dist[k]
    dist = dist.reshape(4*int(size[0]),4*int(size[1]),4*int(size[2]))
    vdf=dist
    i,j,k = np.unravel_index(np.argmax(vdf), vdf.shape)
    len=box
    data=vdf[(i-len):(i+len),(j-len):(j+len),(k-len):(k+len)]
    return np.array(data,dtype=np.float32)


def plot_vdfs(a,b):
    nx,ny,nz=np.shape(a)
    fig, ax = plt.subplots(2, 3, figsize=[12,6])
    
    slicer2d = np.s_[:,:,nz//2]
    slicer1d = np.s_[:,ny//2,nz//2]
    im1=ax[0,0].imshow(a[slicer2d],norm=colors.LogNorm(vmin=1e-15))
    im2=ax[0,1].imshow(b[slicer2d],norm=colors.LogNorm(vmin=1e-15))
    ax[1,0].semilogy(b[slicer1d],label="Reconstructed")
    ax[1,0].semilogy(a[slicer1d],label="Original")
    ax2 = ax[1,0].twinx()
    ax2.plot(b[slicer1d]-a[slicer1d],label="Difference recon-orig",color='k')
    yabs_max = abs(max(ax2.get_ylim(), key=abs))
    ax2.set_ylim(ymin=-yabs_max, ymax=yabs_max)
    im4=ax[1,1].imshow(np.abs(a[slicer2d] - b[slicer2d]))
    plt.colorbar(im1)
    plt.colorbar(im2)
    plt.colorbar(im4)
    ax[0,0].set_title("Original VDF")
    ax[0,1].set_title("Reconstructed VDF")
    ax[1,0].set_title("Profile")
    ax[1,1].set_title("Absolute Diff")
    ax[1,0].legend()
    ax2.legend()
    
#     lapl_0 = ndimage.laplace(a)
    lapl_0 = ndimage.gaussian_laplace(a,0.5)
    #27-point stencil
    k = np.array([[[2,3,2],[3,6,3]  ,[2,3,2]],
                  [[3,6,3],[6,-88,8],[3,6,3]],
                  [[2,3,2],[3,6,3]  ,[2,3,2]]])/26
#     lapl_0 = ndimage.convolve(a, k)
    im5 = ax[0,2].imshow(lapl_0[slicer2d],cmap='seismic')#norm=colors.SymLogNorm(1e-15,vmin=-1e-12,vmax=1e-12))
#     im6 = ax[1,2].imshow(np.abs(lapl_0-ndimage.gaussian_laplace(b,1))[slicer2d])
    plt.colorbar(im5)
    im6 = ax[1,2].imshow((lapl_0**2/a)[slicer2d], norm=colors.LogNorm(vmin=1e-17,vmax=1e-13),cmap='seismic')
    plt.colorbar(im6)
    ax[0,2].set_title("Discrete laplacian, original")
    ax[1,2].set_title("Discrete laplacian**2/vdf, original")
#     ax[1,2].set_title("Abs. diff of laplacians")
    plt.tight_layout()
    plt.show()

def scale_vdf(vdf):
    vdf=vdf.astype(np.float64).flatten()
    vdf[vdf<1e-16]=1e-16
    vdf = np.log10(vdf)
    return vdf;

def unscale_vdf(vdf):
    vdf = 10 ** vdf
    vdf[vdf <= 1e-16] = 0
    return vdf

def print_comparison_stats(a,b):
    def get_moments(vdf):
        density = np.sum(vdf)
        mean_velocity_x = np.sum(vdf * np.arange(vdf.shape[0])) / density
        mean_velocity_y = np.sum(vdf * np.arange(vdf.shape[1])) / density
        mean_velocity_z = np.sum(vdf * np.arange(vdf.shape[2])) / density
        return density, (mean_velocity_x, mean_velocity_y, mean_velocity_z)
    
    def relative_norms(a, b):
        diff = a - b
        l1_norm = np.sum(np.abs(diff)) / np.sum(np.abs(a))
        l2_norm = np.linalg.norm(diff) / np.linalg.norm(b)
        return l1_norm, l2_norm
        
    density1, mean1 = get_moments(a)
    density2, mean2 = get_moments(b)
    
    # Calculate relative percentage difference in moments
    relative_diff_r = np.abs(density1 - density2) / np.mean([density1, density2]) * 100.0
    relative_diff_v = np.linalg.norm(np.array(mean1) - np.array(mean2)) / np.linalg.norm(np.mean([mean1, mean2], axis=0)) * 100.0
    print(f"Moment Stats (R,Vm)= {np.round(relative_diff_r,3),np.round(relative_diff_v,3)} %.")
    l1,l2=relative_norms(a,b)
    print(f"L1,L2 rNorms= {np.round(l1,3),np.round(l2,3)}.")
    

#Ivan GMM
def run_gmm(vdf_3d,n_pop,norm_range):
    flat_data=vdf_3d.flatten()

    xbins,ybins,zbins=vdf_3d.shape[0],vdf_3d.shape[1],vdf_3d.shape[2]
    vx,vy,vz=np.linspace(0,xbins,xbins),np.linspace(0,ybins,ybins),np.linspace(0,zbins,zbins)

    normal_unit=np.max(vdf_3d)/norm_range
    point_x,point_y,point_z=[],[],[]
    total_count=0

    for n in range(int(flat_data.size)):
        k = int(n / (xbins*ybins))
        j = int((n - k*xbins*ybins)/xbins)
        i = n-xbins*(j + ybins*k)

        Npart_in_bin=flat_data[n]//normal_unit
        total_count=total_count+Npart_in_bin

        for kk in range(0,int(Npart_in_bin),1):
            point_x.append(vx[i])
            point_y.append(vy[j])
            point_z.append(vz[k])

    ## APPLYING GMM
    point_cloud=[point_x,point_y,point_z]
    point_cloud=np.array(point_cloud).T

    clf = GaussianMixture(n_components=n_pop, covariance_type="full")
    gmm=clf.fit(point_cloud)

    return gmm.means_,gmm.weights_,gmm.covariances_,normal_unit

### reconstruction of ANISOTROPIC gmm
from numba import jit
@jit(fastmath=True)
def multivariate_normal(x, mean, covariance_matrix):
    n = len(x)
    det_covariance = np.linalg.det(covariance_matrix)
    inv_covariance = np.linalg.inv(covariance_matrix)
    #print('inv_covariance',inv_covariance)
    constant_term = 1 / ((2 * np.pi) ** (n / 2) * np.sqrt(det_covariance))
    exponent_term = np.exp(-0.5 * np.dot(np.dot((x - mean).T, inv_covariance), (x - mean)))
    pdf_value = constant_term * exponent_term
    return pdf_value

def reconstruct_vdf(n_pop,means,covs,weights,n_bins,v_min,v_max):

    vx=np.linspace(v_min,v_max,n_bins)
    vy=np.linspace(v_min,v_max,n_bins)
    vz=np.linspace(v_min,v_max,n_bins)

    vx3, vy3, vz3 = np.meshgrid(vx,vy,vz,indexing='ij')
    flat_grid_x=vx3.flatten()
    flat_grid_y=vy3.flatten()
    flat_grid_z=vz3.flatten()

    #### reconstruction
    f_mult={}
    for n_p in range(n_pop):
        print('reconstruction: n pop done', n_p)
        f={}
        for kkk in range(flat_grid_x.size):
            f[kkk] =  multivariate_normal([flat_grid_x[kkk],flat_grid_y[kkk],flat_grid_z[kkk]], means[n_p], covs[n_p])
        f_mult[n_p]=f

    f_tot=np.zeros(len(f_mult[0]))
    for n_p in range(n_pop):
        f_tot += weights[n_p]*np.array(list(f_mult[n_p].values()))

    #### build 3D array
    f_3d=np.zeros([n_bins,n_bins,n_bins])
    count=0
    for k in range(n_bins):
        for j in range(n_bins):
            for i in range(n_bins):
                f_3d[i,j,k] = f_tot[count]
                count += 1
    return f_3d


from numba import jit
@jit(fastmath=True)
def gaussian_3d(xyz, A, x0, y0, z0, sigma_x, sigma_y, sigma_z):
    x, y, z = xyz
    return A * np.exp(
        -((x - x0)**2 / (2 * sigma_x**2) + (y - y0)**2 / (2 * sigma_y**2) + (z - z0)**2 / (2 * sigma_z**2))
    )

### create three base vectors
@jit(fastmath=True)
def get_v_ax(v_min,v_max,n_bins):
    vx=np.linspace(v_min,v_max,n_bins)
    vy=np.linspace(v_min,v_max,n_bins)
    vz=np.linspace(v_min,v_max,n_bins)
    return [vx,vy,vz]


### Maxwell fit
def max_fit(vdf_3d,v_min,v_max,n_bins,guess):
    ### create velocity mesh
    #v_min,v_max,n_bins=0,200,200
    v_ax=get_v_ax(v_min,v_max,n_bins)
    vxx, vyy, vzz = np.meshgrid(v_ax[0], v_ax[1], v_ax[2], indexing='ij')

    # Flatten the data for curve fitting
    flatten_data = vdf_3d.flatten()
    flatten_xyz = np.vstack([vxx.flatten(), vyy.flatten(), vzz.flatten()])

    ##### fit 3D Gausssian // initial guess is required //  need to write searching module
    initial_guess = guess
    params, covariance = curve_fit(gaussian_3d, flatten_xyz, flatten_data, p0=initial_guess)

    #fitted_params = [params[0]] + list(params[1:])
    max_fit_3d = gaussian_3d((vxx, vyy, vzz), *params)
    return max_fit_3d,params


### hermite polynomial
def herm_phys(v,m):
    #hermit = special.hermite(m, monic=False)
    hermit = np.polynomial.hermite.Hermite.basis(m)(v)
    return hermit

### physical hermite polynomial
def herm_phys_spec(v,u,vth,m):
    y= (np.exp(-(v-u)**2 / (2*vth**2) )*herm_phys((v-u)/vth,m) ) / (np.sqrt( (2**m)*np.math.factorial(m)*np.sqrt(np.pi)*vth))
    return y

### flattening of the velocity mesh
def get_flat_mesh(v_min,v_max,n_bins):
    vx = np.linspace(v_min, v_max, n_bins)
    vy = np.linspace(v_min, v_max, n_bins)
    vz = np.linspace(v_min, v_max, n_bins)
    vxx, vyy, vzz = np.meshgrid(vx, vy, vz, indexing='ij')
    flatten_xyz = np.vstack([vxx.flatten(), vyy.flatten(), vzz.flatten()])
    return flatten_xyz

### create array of hermite polynomial according to local v_bulk and v_thermal
def herm_mpl_arr(m_pol,v_ax,u,vth):
    vx,vy,vz = v_ax
    herm_x,herm_y,herm_z = np.zeros([m_pol,vx.shape[0]]),np.zeros([m_pol,vy.shape[0]]),np.zeros([m_pol,vz.shape[0]])
    for i in range(m_pol):
        herm_x[i,:] = herm_phys_spec(vx,u[0],vth[0],i)
        herm_y[i,:] = herm_phys_spec(vy,u[1],vth[1],i)
        herm_z[i,:] = herm_phys_spec(vz,u[2],vth[2],i)
    print('array with base polynomials created')
    return [herm_x,herm_y,herm_z]


# forward transform / calculation of the coefficients for hermite decomposition
def coefficient_matrix(vdf_3d_flat,mm,herm_array,v_xyz):
    # vxx, vyy, vzz = v_xyz[0], v_xyz[1], v_xyz[2]
    dv = 1
    result = np.zeros((mm, mm, mm))
    for mx in range(mm):
        herm_mx = herm_array[0,mx,:]
        for my in range(mm):
            herm_my = herm_array[1,my,:]
            for mz in range(mm):
                herm_mz = herm_array[2,mz,:]
                #result[mx, my, mz] = np.sum( np.sum( np.sum(vdf_3d_flat*herm_mx*dv)* herm_my*dv)*herm_mz*dv )
                result[mx, my, mz] = np.einsum('i,i,i', vdf_3d_flat*herm_mx, herm_my, herm_mz) * dv**3
    return result

# inverse transform / reconstruction of the original
@jit(fastmath=True)
def inv_herm_trans(mm_matrix,herm_array, v_xyz):
    vx,vy,vz=v_xyz
    f=np.zeros_like(vx)
    for mx in range(mm_matrix.shape[0]):
        print('mode number',mx)
        for my in range(mm_matrix.shape[1]):
            for mz in range(mm_matrix.shape[2]):
                #f += mm_matrix[mx, my, mz] * herm_phys_spec(vx,u[0],vth[0],mx) * herm_phys_spec(vy,u[1],vth[1],my)* herm_phys_spec(vz,u[2],vth[2],mz)
                f += mm_matrix[mx, my, mz] * herm_array[0,mx,:] * herm_array[1,my,:]* herm_array[2,mz,:]
    return f





### 3D case
import numpy as np
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture


def run_gmm(vdf_3d,n_pop,norm_range):
    flat_data=vdf_3d.flatten()

    xbins,ybins,zbins=vdf_3d.shape[0],vdf_3d.shape[1],vdf_3d.shape[2]    
    vx,vy,vz=np.linspace(0,xbins,xbins),np.linspace(0,ybins,ybins),np.linspace(0,zbins,zbins)

    normal_unit=np.max(vdf_3d)/norm_range
    point_x,point_y,point_z=[],[],[]
    total_count=0

    for n in range(int(flat_data.size)): 
        k = int(n / (xbins*ybins))
        j = int((n - k*xbins*ybins)/xbins)
        i = n-xbins*(j + ybins*k)    
                
        Npart_in_bin=flat_data[n]//normal_unit        
        total_count=total_count+Npart_in_bin

        for kk in range(0,int(Npart_in_bin),1):
            point_x.append(vx[i])
            point_y.append(vy[j])
            point_z.append(vz[k])

    ## APPLYING GMM
    point_cloud=[point_x,point_y,point_z]
    point_cloud=np.array(point_cloud).T
    
    clf = GaussianMixture(n_components=n_pop, covariance_type="full")
    gmm=clf.fit(point_cloud)

    return gmm.means_,gmm.weights_,gmm.covariances_,normal_unit



### reconstruction of ANISOTROPIC gmm
@jit(fastmath=True)
def multivariate_normal(x, mean, covariance_matrix):    
    n = len(x)
    det_covariance = np.linalg.det(covariance_matrix)
    inv_covariance = np.linalg.inv(covariance_matrix)
    #print('inv_covariance',inv_covariance)
    #constant_term = 1 / ((2 * np.pi) ** (n / 2) * np.sqrt(det_covariance))
    exponent_term = np.exp(-0.5 * np.dot(np.dot((x - mean).T, inv_covariance), (x - mean)))
    pdf_value = exponent_term
    #pdf_value = constant_term*exponent_term
    return pdf_value


def reconstruct_vdf(n_pop,means,covs,weights,n_bins,v_min,v_max):

    vx=np.linspace(v_min,v_max,n_bins)
    vy=np.linspace(v_min,v_max,n_bins)
    vz=np.linspace(v_min,v_max,n_bins)

    vx3, vy3, vz3 = np.meshgrid(vx,vy,vz,indexing='ij')    
    flat_grid_x=vx3.flatten()
    flat_grid_y=vy3.flatten()
    flat_grid_z=vz3.flatten()

    #### reconstruction
    f_mult={}
    for n_p in range(n_pop):
        print('reconstruction: n pop done', n_p)
        f={}
        for kkk in range(flat_grid_x.size):
            f[kkk] =  multivariate_normal([flat_grid_x[kkk],flat_grid_y[kkk],flat_grid_z[kkk]], means[n_p], covs[n_p])
        f_mult[n_p]=f

    f_tot=np.zeros(len(f_mult[0]))
    for n_p in range(n_pop):
        f_tot += weights[n_p]*np.array(list(f_mult[n_p].values()))

    #### build 3D array
    f_3d=np.zeros([n_bins,n_bins,n_bins])
    count=0
    for k in range(n_bins):
        for j in range(n_bins):
            for i in range(n_bins):
                f_3d[i,j,k] = f_tot[count]
                count += 1    
    return f_3d             

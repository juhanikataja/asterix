#some math to calculate the bulk velocity and its location in the VDF space
def get_bulk_v(vdf,f,sparse=1e-16):
    from scipy.integrate import simps
    import numpy as np
    mesh=f.get_velocity_mesh_extent()
    sz=f.get_velocity_mesh_size()
    wid=int(f.get_WID())
    v_x = np.linspace(mesh[0], mesh[3], int(wid*sz[0]))
    v_y = np.linspace(mesh[1], mesh[4], int(wid*sz[1]))
    v_z = np.linspace(mesh[2], mesh[5], int(wid*sz[2]))
    Vx, Vy, Vz = np.meshgrid(v_x, v_y, v_z, indexing='ij')
    term_1_x = simps(simps(simps(Vx * vdf, v_z), v_y), v_x)
    term_1_y = simps(simps(simps(Vy * vdf, v_z), v_y), v_x)
    term_1_z = simps(simps(simps(Vz * vdf, v_z), v_y), v_x)
    term_2 = simps(simps(simps(vdf, v_z), v_y), v_x)
    bulk_v = np.array([term_1_x / term_2,term_1_y / term_2,term_1_z / term_2])    
    index_x = (np.abs(v_x - bulk_v[0])).argmin()
    index_y = (np.abs(v_y - bulk_v[1])).argmin()
    index_z = (np.abs(v_z - bulk_v[2])).argmin()
    loc=[index_x,index_y,index_z]
    return bulk_v ,loc

def get_vdf_bounding_box(vdf,sparse):
    import numpy as np
    #Get BBOX
    valid_indices = np.argwhere(vdf > sparse)
    bounding_box=[valid_indices[:, 0].min(), valid_indices[:, 1].min(),         
                  valid_indices[:, 2].min(), valid_indices[:, 0].max(),         
                  valid_indices[:, 1].max(), valid_indices[:, 2].max()]
    return bounding_box

def pad_array(input,shape,val=0):
    import numpy as np
    assert len(input.shape) == len(shape) 
    width=[]
    for c,dim in enumerate(input.shape):
        target=shape[c]
        left = (target - dim) // 2
        right = target - left - dim
        width.append((left,right))
    return np.pad(input,width,mode="constant",constant_values=val)
        
def extract(f, cid,sparsity=1e-16,restrict_box=True):
    import numpy as np
    assert cid > 0

    # -- read phase space density
    vcells = f.read_velocity_cells(cid)
    keys = list(vcells.keys())
    values = list(vcells.values())

    # -- generate a velocity space
    size = f.get_velocity_mesh_size()
    vids = np.arange(4 * 4 * 4 * int(size[0]) * int(size[1]) * int(size[2]))

    # -- put phase space density into array
    dist = np.zeros_like(vids, dtype=float)
    dist[keys] = values

    # -- sort vspace by velocity
    v = f.get_velocity_cell_coordinates(vids)
    kk = f.get_velocity_cell_coordinates(vids)

    i = np.argsort(v[:, 0], kind="stable")
    v = v[i]
    vids = vids[i]
    dist = dist[i]

    j = np.argsort(v[:, 1], kind="stable")
    v = v[j]
    vids = vids[j]
    dist = dist[j]

    k = np.argsort(v[:, 2], kind="stable")
    v = v[k]
    vids = vids[k]
    dist = dist[k]
    dist = dist.reshape(4 * int(size[0]), 4 * int(size[1]), 4 * int(size[2]))

    vdf = dist
    bulk_v,bulk_v_loc=get_bulk_v(vdf,f,sparsity)
    bbox=get_vdf_bounding_box(vdf,sparsity)
    bbox_max_side=np.max([bbox[3]-bbox[0]+1,bbox[4]-bbox[1]+1,bbox[5]-bbox[2]+1])
    len=bbox_max_side//2
    if (not restrict_box):
        return bulk_v_loc,np.array(vdf, dtype=np.double),len
    data = vdf[(bulk_v_loc[0] - len) : (bulk_v_loc[0] + len), (bulk_v_loc[1] - len) : (bulk_v_loc[1] + len), (bulk_v_loc[2] - len) : (bulk_v_loc[2] + len)]
    # print(f"Extracted VDF shape = {np.shape(data)}")
    return bulk_v_loc,np.array(data, dtype=np.double),len


if __name__ == "__main__":
    import os, sys

    if len(sys.argv) < 3:
        print(f"Usage: ./{sys.argv[0]} <vlsv_file> <cell_id>")
        sys.exit(1)
    file = sys.argv[1]
    cid = int(sys.argv[2])
    f = pt.vlsvfile.VlsvReader(file)
    extract(f, cid, 25)

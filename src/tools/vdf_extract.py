
def extract(f,cid,len=25):
    import numpy as np
    import pytools as pt
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm
    from matplotlib.colors import LogNorm
    assert(cid>0)

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
    kk = f.get_velocity_cell_coordinates(vids)

    i = np.argsort(v[:,0],kind='stable')
    v = v[i]
    vids = vids[i]
    dist = dist[i]

    j = np.argsort(v[:,1],kind='stable')
    v = v[j]
    vids = vids[j]
    dist = dist[j]

    k = np.argsort(v[:,2],kind='stable')
    v = v[k]
    vids = vids[k]
    dist = dist[k]
    dist = dist.reshape(4*int(size[0]),
                        4*int(size[1]),
                        4*int(size[2]))



    vdf=dist
    if len>0:
        i,j,k = np.unravel_index(np.argmax(vdf), vdf.shape)
        data=vdf[(i-len):(i+len),(j-len):(j+len),(k-len):(k+len)]
    else:
        i=j=k=0
        data=vdf[:,:,:]
    print(f"Extracted VDF shape = {np.shape(data)}")
    return [i,j,k],np.array(data,dtype=np.double)

if __name__=="__main__":
    import os,sys
    if ( len(sys.argv)<3 ):
        print(f"Usage: ./{sys.argv[0]} <vlsv_file> <cell_id>")
        sys.exit(1)
    file=sys.argv[1]
    cid = int(sys.argv[2])  
    f = pt.vlsvfile.VlsvReader(file)
    extract(f,cid,25)

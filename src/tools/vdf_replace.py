import sys,os
from tqdm import tqdm

if (len(sys.argv)!=2):
    print(len(sys.argv))
    print("ERROR: Wrong usage!")
    print(f"USAGE: ./{sys.argv[0]} <vlsv file>")
    sys.exit()

import pytools as pt
import numpy as np
import vdf_extract
import mlp_compress

def reconstruct_cid(f,cid):
    vdf=vdf_extract.extract(f,cid,25)
    sparsity=1.e-16;
    if f.check_variable('MinValue'):
        sparsity = f.read_variable("proton"+"/EffectiveSparsityThreshold",cid)   
    reconstructed_vdf=np.reshape(mlp_compress.compress_mlp_from_vec(vdf.flatten(),12,5,2,50,50,sparsity),(50,50,50))
    return cid,reconstructed_vdf
    

def reconstruct_vdfs(file):
    f=pt.vlsvfile.VlsvReader(file)
    cids=np.array(np.arange(1,1+np.prod(f.get_spatial_mesh_size())),dtype=int)
    print("Cell IDs to replace =",cids);
    for cid in cids:
        print(f"Extracting {cid}")
        key,recon=reconstruct_cid(f,cid)


        
file=sys.argv[1]
reconstruct_vdfs(file)
 


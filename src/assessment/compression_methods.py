import numpy as np
import vdf_extract
import mlp_compress
import shutil
import struct
import sys,os
import warnings
import numpy as np
import ctypes
import pyzfp,zlib
import mlp_compress
import tools
import pytools

def sparsify(vdf,sparsity):
    vdf[vdf<sparsity]=0.0
    return
    

# MLP with fourier features
def reconstruct_cid_fourier_mlp(f, cid,sparsity):
    order = 48
    epochs = 50
    hidden_layers=[75,50,50,10]
    max_indexes, vdf,len = vdf_extract.extract(f, cid,sparsity)
    nx, ny, nz = np.shape(vdf)
    assert nx == ny == nz
    reconstructed_vdf = np.reshape(
        mlp_compress.compress_mlp_from_vec(
            vdf.flatten(), order, epochs,np.array(hidden_layers,dtype=np.uint64) , nx, sparsity
        ),
        (nx, ny, nz),
    )
    mesh = f.get_velocity_mesh_size()
    final_vdf = np.zeros((int(4 * mesh[0]), int(4 * mesh[1]), int(4 * mesh[2])))
    final_vdf[
        max_indexes[0] - len : max_indexes[0] + len,
        max_indexes[1] - len : max_indexes[1] + len,
        max_indexes[2] - len : max_indexes[2] + len,
    ] = reconstructed_vdf
    sparsify(final_vdf,sparsity)
    return cid, np.array(final_vdf, dtype=np.float32)

# MLP
def reconstruct_cid_mlp(f, cid,sparsity):
    order = 0
    epochs = 50
    hidden_layers=[75,50,50,10]
    max_indexes, vdf,len = vdf_extract.extract(f, cid,sparsity)
    nx, ny, nz = np.shape(vdf)
    assert nx == ny == nz
    reconstructed_vdf = np.reshape(
        mlp_compress.compress_mlp_from_vec(
            vdf.flatten(), order, epochs, np.array(hidden_layers,dtype=np.uint64) , nx, sparsity
        ),
        (nx, ny, nz),
    )
    mesh = f.get_velocity_mesh_size()
    final_vdf = np.zeros((int(4 * mesh[0]), int(4 * mesh[1]), int(4 * mesh[2])))
    final_vdf[
        max_indexes[0] - len : max_indexes[0] + len,
        max_indexes[1] - len : max_indexes[1] + len,
        max_indexes[2] - len : max_indexes[2] + len,
    ] = reconstructed_vdf
    sparsify(final_vdf,sparsity)
    return cid, np.array(final_vdf, dtype=np.float32)

# ZFP
def reconstruct_cid_zfp(f, cid,sparsity):
    tolerance = 1e-13
    max_indexes, vdf,len = vdf_extract.extract(f, cid,sparsity)
    nx, ny, nz = np.shape(vdf)
    assert nx == ny == nz
    compressed_vdf = pyzfp.compress(vdf, tolerance=tolerance)
    reconstructed_vdf = pyzfp.decompress(compressed_vdf,vdf.shape,vdf.dtype,tolerance)
    mesh = f.get_velocity_mesh_size()
    final_vdf = np.zeros((int(4 * mesh[0]), int(4 * mesh[1]), int(4 * mesh[2])))
    final_vdf[
        max_indexes[0] - len : max_indexes[0] + len,
        max_indexes[1] - len : max_indexes[1] + len,
        max_indexes[2] - len : max_indexes[2] + len,
    ] = reconstructed_vdf
    sparsify(final_vdf,sparsity)
    return cid, np.array(final_vdf, dtype=np.float32)

# Spherical Harmonics
def reconstruct_cid_sph(f, cid,sparsity):
    degree =10 
    max_indexes, vdf,len = vdf_extract.extract(f, cid,sparsity)
    nx, ny, nz = np.shape(vdf)
    assert nx == ny == nz
    reconstructed_vdf=mlp_compress.compress_sph_from_vec(vdf.flatten(),degree,nx)
    reconstructed_vdf=np.array(reconstructed_vdf,dtype=np.double)
    reconstructed_vdf= np.reshape(reconstructed_vdf,np.shape(vdf),order='C')
    mesh = f.get_velocity_mesh_size()
    final_vdf = np.zeros((int(4 * mesh[0]), int(4 * mesh[1]), int(4 * mesh[2])))
    final_vdf[
        max_indexes[0] - len : max_indexes[0] + len,
        max_indexes[1] - len : max_indexes[1] + len,
        max_indexes[2] - len : max_indexes[2] + len,
    ] = reconstructed_vdf
    sparsify(final_vdf,sparsity)
    return cid, np.array(final_vdf, dtype=np.float32)


# Octree
def reconstruct_cid_oct(f, cid,sparsity):
    from juliacall import Main as jl
    max_indexes, vdf ,len= vdf_extract.extract(f, cid,sparsity)
    nx, ny, nz = np.shape(vdf)
    assert nx == ny == nz
    jl.Pkg.activate("src/jl_env")
    jl.Pkg.instantiate()
    jl.include("src/octree.jl")
    A, b, img, reco, cell, tree = jl.VDFOctreeApprox.compress(vdf, maxiter=500, alpha=0.0, beta=1.0, nu=2, tol=3e-1, verbose=False)
    reconstructed_vdf=np.array(reco,dtype=np.double)
    reconstructed_vdf= np.reshape(reconstructed_vdf,np.shape(vdf),order='C')
    mesh = f.get_velocity_mesh_size()
    final_vdf = np.zeros((int(4 * mesh[0]), int(4 * mesh[1]), int(4 * mesh[2])))
    final_vdf[
        max_indexes[0] - len : max_indexes[0] + len,
        max_indexes[1] - len : max_indexes[1] + len,
        max_indexes[2] - len : max_indexes[2] + len,
    ] = reconstructed_vdf
    sparsify(final_vdf,sparsity)
    return cid, np.array(final_vdf, dtype=np.float32)


# PCA
def reconstruct_cid_pca(f, cid,sparsity):
    from sklearn.decomposition import PCA
    n =10 
    max_indexes, vdf,len = vdf_extract.extract(f, cid,sparsity)
    nx, ny, nz = np.shape(vdf)
    assert nx == ny == nz
    vdf[vdf<sparsity]=sparsity
    vdf = np.log10(vdf)
    arr=vdf.copy()
    arr = arr.reshape(arr.shape[0], -1)
    cov_matrix = np.cov(arr, rowvar=False)
    eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)
    pca = PCA(n_components=n)
    compressed = pca.fit_transform(arr)
    #reconstruct the data
    recon = pca.inverse_transform(compressed)
    nx,ny,nz=np.shape(vdf)
    recon=np.reshape(recon,(nx,ny,nz))
    reconstructed_vdf = 10 ** recon
    reconstructed_vdf[reconstructed_vdf <= sparsity] = 0
    reconstructed_vdf=np.array(reconstructed_vdf,dtype=np.double)
    reconstructed_vdf= np.reshape(reconstructed_vdf,np.shape(vdf),order='C')
    mesh = f.get_velocity_mesh_size()
    final_vdf = np.zeros((int(4 * mesh[0]), int(4 * mesh[1]), int(4 * mesh[2])))
    final_vdf[
        max_indexes[0] - len : max_indexes[0] + len,
        max_indexes[1] - len : max_indexes[1] + len,
        max_indexes[2] - len : max_indexes[2] + len,
    ] = reconstructed_vdf
    sparsify(final_vdf,sparsity)
    return cid, np.array(final_vdf, dtype=np.float32)


#CNN
def reconstruct_cid_cnn(f, cid,sparsity):
    import torch
    import torch.nn as nn
    import torch.optim as optim

    class CNN(nn.Module):
        def __init__(self):
            super(CNN, self).__init__()
            self.conv1 = nn.Conv3d(1, 16, kernel_size=3, padding=1)
            self.bn1 = nn.BatchNorm3d(16)
            self.conv2 = nn.Conv3d(16, 32, kernel_size=3, padding=1)
            self.bn2 = nn.BatchNorm3d(32)
            self.conv3 = nn.Conv3d(32, 64, kernel_size=3, padding=1)
            self.bn3 = nn.BatchNorm3d(64)
            self.conv4 = nn.Conv3d(64, 1, kernel_size=3, padding=1)
            self.relu = nn.ReLU()
        
        def forward(self, x):
            x = self.relu(self.bn1(self.conv1(x)))
            x = self.relu(self.bn2(self.conv2(x)))
            x = self.relu(self.bn3(self.conv3(x)))
            x = self.conv4(x)
            return x

    def train_and_reconstruct(input_array, num_epochs=30, learning_rate=0.001, batch_size=32):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        input_tensor = torch.tensor(input_array, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)  # Move input tensor to device
        model = CNN().to(device) 
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
        for epoch in range(num_epochs):
            for i in range(0, input_tensor.size(0), batch_size):
                optimizer.zero_grad()
                batch_input = input_tensor[i:i+batch_size]
                output_tensor = model(batch_input)
                loss = criterion(output_tensor, batch_input)
                loss.backward()
                optimizer.step()
    
        with torch.no_grad():
            output_tensor = model(input_tensor)
        reconstructed_array = output_tensor.squeeze(0).squeeze(0).cpu().numpy()
    
        param_size = 0
        for param in model.parameters():
            param_size += param.nelement() * param.element_size()
        buffer_size = 0
        for buffer in model.buffers():
            buffer_size += buffer.nelement() * buffer.element_size()
        size = (param_size + buffer_size)   
        return reconstructed_array, size
    
    epochs=10
    max_indexes, vdf,len = vdf_extract.extract(f, cid,sparsity)
    nx, ny, nz = np.shape(vdf)
    assert nx == ny == nz
    vdf[vdf<sparsity]=sparsity
    vdf = np.log10(vdf)
    input_array=vdf
    reconstructed_vdf,total_size= train_and_reconstruct(input_array,epochs)
    reconstructed_vdf = 10 ** reconstructed_vdf
    reconstructed_vdf[reconstructed_vdf <= sparsity] = 0
    reconstructed_vdf=np.array(reconstructed_vdf,dtype=np.double)
    reconstructed_vdf= np.reshape(reconstructed_vdf,np.shape(vdf),order='C')
    mesh = f.get_velocity_mesh_size()
    final_vdf = np.zeros((int(4 * mesh[0]), int(4 * mesh[1]), int(4 * mesh[2])))
    final_vdf[
        max_indexes[0] - len : max_indexes[0] + len,
        max_indexes[1] - len : max_indexes[1] + len,
        max_indexes[2] - len : max_indexes[2] + len,
    ] = reconstructed_vdf
    sparsify(final_vdf,sparsity)
    return cid, np.array(final_vdf, dtype=np.float32)


# GMM
def reconstruct_cid_gmm(f, cid,sparsity):
    n_pop=5
    norm_range=300
    max_indexes, vdf,len = vdf_extract.extract(f, cid,sparsity)
    nx, ny, nz = np.shape(vdf)
    assert nx == ny == nz
    means,weights,covs,norm_unit=tools.run_gmm(vdf,n_pop,norm_range)
    n_bins=nx
    v_min,v_max=0,nx
    reconstructed_vdf=tools.reconstruct_vdf(n_pop,means,covs,weights,n_bins,v_min,v_max)
    reconstructed_vdf=reconstructed_vdf*norm_unit*norm_range
    reconstructed_vdf=np.array(reconstructed_vdf,dtype=np.double)
    reconstructed_vdf= np.reshape(reconstructed_vdf,np.shape(vdf),order='C')
    mesh = f.get_velocity_mesh_size()
    final_vdf = np.zeros((int(4 * mesh[0]), int(4 * mesh[1]), int(4 * mesh[2])))
    final_vdf[
        max_indexes[0] - len : max_indexes[0] + len,
        max_indexes[1] - len : max_indexes[1] + len,
        max_indexes[2] - len : max_indexes[2] + len,
    ] = reconstructed_vdf
    sparsify(final_vdf,sparsity)
    return cid, np.array(final_vdf, dtype=np.float32)


# DWT
def reconstruct_cid_dwt(f, cid,sparsity):
    import pywt
    max_indexes, vdf,len = vdf_extract.extract(f, cid,sparsity)
    nx, ny, nz = np.shape(vdf)
    assert nx == ny == nz
    threshold = sparsity
    orig_shape = vdf.shape
    vdf[np.isnan(vdf)] = 0
    comp_type = np.float32
    norm = 1# np.nanmax(vdf)/quant
    vdf /= norm
    vdf[vdf<0]=0
    wavelet = 'db4'#'bior1.3'
    dwtn_mlevel = pywt.dwtn_max_level(vdf.shape,wavelet)
    level_delta = 2
    coeffs3 = pywt.wavedecn(vdf,wavelet=wavelet, level = dwtn_mlevel-2)
    coeffs3_comp = coeffs3.copy()
    zeros = 0
    nonzeros = 0
    for i,a in enumerate(coeffs3_comp):
        print(type(a))
        zero_app = False
        # print(a.shape)
        if(type(a) == type(np.ndarray(1))):
            coeffs3_comp[i] = a
            mask = np.abs(a) < threshold
            zeros += np.sum(mask)
            nonzeros += np.sum(~mask)
            # nonzeros += np.prod(a.shape)
            coeffs3_comp[i][mask] = 0
        else:
            for k,v in a.items():
                mask = np.abs(v) < threshold
                coeffs3_comp[i][k] = v
                coeffs3_comp[i][k][mask] = 0
                zeros += np.sum(mask)
                nonzeros += np.sum(~mask)

    reconstructed_vdf = pywt.waverecn(coeffs3_comp,wavelet=wavelet)*norm    
    reconstructed_vdf=np.array(reconstructed_vdf,dtype=np.double)
    reconstructed_vdf= np.reshape(reconstructed_vdf,np.shape(vdf),order='C')
    mesh = f.get_velocity_mesh_size()
    final_vdf = np.zeros((int(4 * mesh[0]), int(4 * mesh[1]), int(4 * mesh[2])))
    final_vdf[
        max_indexes[0] - len : max_indexes[0] + len,
        max_indexes[1] - len : max_indexes[1] + len,
        max_indexes[2] - len : max_indexes[2] + len,
    ] = reconstructed_vdf
    sparsify(final_vdf,sparsity)
    return cid, np.array(final_vdf, dtype=np.float32)


# DCT
def reconstruct_cid_dct(f, cid,sparsity):
    from scipy.fft import dctn, idctn
    blocksize = 8
    keep_n = 4
    max_indexes, vdf,len = vdf_extract.extract(f, cid,sparsity)
    nx, ny, nz = np.shape(vdf)
    assert nx == ny == nz
    orig_shape = vdf.shape
    vdf[np.isnan(vdf)] = 0

    paddings = (np.ceil(np.array(vdf.shape)/8)).astype(int)*8 - vdf.shape
    paddings = ((0,paddings[0]),(0,paddings[1]),(0,paddings[2]))
    vdf = np.pad(vdf, paddings)

    block_data = np.zeros_like(vdf)
    for i in range(0,vdf.shape[0], blocksize):
        for j in range(0, vdf.shape[1], blocksize):
            for k in range(0, vdf.shape[2], blocksize):
                block_data[i:i+blocksize,j:j+blocksize, k:k+blocksize] = dctn(vdf[i:i+blocksize,j:j+blocksize, k:k+blocksize])

    zeroed = np.zeros_like(block_data)
    for i in range(keep_n):
        for j in range(keep_n):
            for k in range(keep_n):
                zeroed[i::blocksize,j::blocksize,k::blocksize] = block_data[i::blocksize,j::blocksize,k::blocksize]


    volume_compressed = np.prod(keep_n*np.ceil(np.array(vdf.shape)/8))
    volume_orig = np.prod(vdf.shape)
    compression = volume_orig/volume_compressed

    vdf_rec = np.zeros_like(vdf)
    for i in range(0,vdf.shape[0], blocksize):
        for j in range(0, vdf.shape[1], blocksize):
            for k in range(0, vdf.shape[2], blocksize):
                vdf_rec[i:i+blocksize,j:j+blocksize, k:k+blocksize] = idctn(zeroed[i:i+blocksize,j:j+blocksize, k:k+blocksize])

    reconstructed_vdf = vdf_rec[0:orig_shape[0],0:orig_shape[1],0:orig_shape[2]]
    reconstructed_vdf=np.array(reconstructed_vdf,dtype=np.double)
    reconstructed_vdf= np.reshape(reconstructed_vdf,orig_shape,order='C')
    mesh = f.get_velocity_mesh_size()
    final_vdf = np.zeros((int(4 * mesh[0]), int(4 * mesh[1]), int(4 * mesh[2])))
    final_vdf[
        max_indexes[0] - len : max_indexes[0] + len,
        max_indexes[1] - len : max_indexes[1] + len,
        max_indexes[2] - len : max_indexes[2] + len,
    ] = reconstructed_vdf
    sparsify(final_vdf,sparsity)
    return cid, np.array(final_vdf, dtype=np.float32)

def reconstruct_cid_vqvae(f, cid,sparsity):
    import torch
    import torch.nn as nn
    import torch.optim as optim

    import vqvae.vqvae_tools as vq
    model_checkpoint='state.ptch'
    tolerance = 1e-13
    max_indexes, vdf,len = vdf_extract.extract(f, cid,sparsity,restrict_box=False)
    vdf[vdf<sparsity]=sparsity
    vdf = np.log10(vdf)
    nx, ny, nz = np.shape(vdf)
    assert nx == ny == nz

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    use_ema = True
    model_args = {
        "in_channels": 1,
        "num_hiddens": 128,
        "num_downsampling_layers": 2,
        "num_residual_layers": 2,
        "num_residual_hiddens": 32,
        "embedding_dim": 64,
        "num_embeddings": 512,
        "use_ema": use_ema,
        "decay": 0.99,
        "epsilon": 1e-5,
    }
    model = vq.VQVAE(**model_args).to(device)
    ckpt=torch.load(model_checkpoint)
    new_ckpt = {}
    for k, v in ckpt.items():
        new_ckpt[k.replace('module.', '', 1)] = v
    model.load_state_dict(new_ckpt)
    model.eval() 
    with torch.no_grad():
        vdf=np.array(vdf,dtype=np.float32)
        input = torch.from_numpy(vdf).unsqueeze(0).unsqueeze(0).to(device)
        out = model(input)
        reconstructed_vdf = out["x_recon"].cpu().numpy().squeeze()
    reconstructed_vdf = 10 ** reconstructed_vdf
    reconstructed_vdf=np.array(reconstructed_vdf,dtype=np.double)
    final_vdf=reconstructed_vdf
    sparsify(final_vdf,sparsity)
    return cid, np.array(final_vdf, dtype=np.float32)



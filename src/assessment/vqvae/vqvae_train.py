import sys
from vqvae_tools import * 
import numpy as np
from tqdm import tqdm
import sys
import os
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from torchvision import datasets, transforms
from torch.multiprocessing import Process
import torch.multiprocessing as mp
from torch.multiprocessing import Process
from torch.utils.tensorboard import SummaryWriter

use_tensorboard=False
writer = SummaryWriter(log_dir="./log")


def plot_vdfs_tb(a, b, vdf_vmin=1e-16):
    from matplotlib import rcParams
    from scipy import ndimage
    import shutil
    rcParams['text.usetex']= True if shutil.which('latex') else False
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm
    nx, ny, nz = np.shape(a)
    fig, ax = plt.subplots(2, 3)


    slicer2d = np.s_[:, :, nz // 2]
    slicer1d = np.s_[:, ny // 2, nz // 2]
    im1 = ax[0, 0].imshow(a[slicer2d])
    im2 = ax[0, 1].imshow(b[slicer2d])
    ax[1, 0].semilogy(b[slicer1d], label="Reconstructed")
    ax[1, 0].semilogy(a[slicer1d], label="Original")
    ax2 = ax[1, 0].twinx()
    ax2.plot(b[slicer1d] - a[slicer1d], label="Difference recon-orig", color="k")
    yabs_max = abs(max(ax2.get_ylim(), key=abs))
    ax2.set_ylim(ymin=-yabs_max, ymax=yabs_max)
    im4 = ax[1, 1].imshow(np.abs(a[slicer2d] - b[slicer2d]))
    plt.colorbar(im1)
    plt.colorbar(im2)
    plt.colorbar(im4)
    ax[0, 0].set_title("Original VDF")
    ax[0, 1].set_title("Reconstructed VDF")
    ax[1, 0].set_title("Profile")
    ax[1, 1].set_title("Absolute Diff")
    ax[1, 0].legend()
    ax2.legend()

    grad_a = np.stack(np.gradient(a), axis=-1)
    grad_b = np.stack(np.gradient(b), axis=-1)
    diff_grads = np.linalg.norm(grad_a - grad_b, axis=-1)

    im5 = ax[0, 2].imshow(diff_grads[slicer2d], cmap="batlow")
    ax[0, 2].set_title("norm(diff of gradient vectors)")
    plt.colorbar(im5)

    lapl_a = ndimage.gaussian_laplace(a, 1)
    lapl_b = ndimage.gaussian_laplace(b, 1)

    diff_lapls = lapl_a - lapl_b

    im6 = ax[1, 2].imshow(diff_grads[slicer2d], cmap="seismic")
    ax[1, 2].set_title("diff of (gaussian[1]) laplacians")
    plt.colorbar(im6)
    return fig


def main():
    mp.set_start_method('spawn')

    dist.init_process_group(backend='nccl')
    rank = int(os.environ['LOCAL_RANK'])
    global_rank = int(os.environ['RANK'])
    torch.cuda.set_device(rank)

    filename=sys.argv[1]
    sparsity=float(sys.argv[2])
    device ='cuda'

    # Initialize model
    use_ema = True # Use exponential moving average
    model_args = {
        "in_channels":1,
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
    model = VQVAE(**model_args).to(device)
    model = DDP(model)

    batch_size =2
    epochs = 100
    workers = 4
    f = pt.vlsvfile.VlsvReader(filename)
    size = f.get_velocity_mesh_size()
    WID = f.get_WID()
    box=int(WID*size[0])
    cids=f.read(mesh="SpatialGrid",name="CellID", tag="VARIABLE")
    VDF_Data = Lazy_Vlasiator_DataSet(cids,filename,device,box,sparsity)
    # train_sampler = DistributedSampler(VDF_Data, rank=rank,device_ids=[rank])
    train_sampler = DistributedSampler(VDF_Data)
    train_loader = DataLoader(
        dataset=VDF_Data,
        batch_size=batch_size,
        sampler=train_sampler,
        shuffle=False,
        num_workers=workers,
        pin_memory=True
    )

    # Multiplier for commitment loss. See Equation (3) in "Neural Discrete Representation Learning"
    beta = 0.25

    # Initialize optimizer
    train_params = [params for params in model.parameters()]
    lr = 3e-4
    optimizer = optim.Adam(train_params, lr=lr)
    criterion = nn.MSELoss()
    # Train model
    eval_every = 1
    best_train_loss = float("inf")
    model.train()



    counter=0
    for epoch in range(epochs):
        train_sampler.set_epoch(epoch)
        total_train_loss = 0
        total_recon_error = 0
        n_train = 0
        for batch_idx, train_tensors in enumerate(train_loader):
            optimizer.zero_grad()
            imgs = train_tensors.to(rank)
            out = model(imgs)
            recon_error = criterion(out["x_recon"], imgs)
            total_recon_error += recon_error.item()
            loss = recon_error + beta * out["commitment_loss"]
            if not use_ema:
                loss += out["dictionary_loss"]

            total_train_loss += loss.item()
            loss.backward()
            optimizer.step()
            n_train += 1
            
            if global_rank and use_tensorboard==0:
                temp=imgs.cpu().detach().numpy()
                reconstructions = out["x_recon"].cpu().detach().numpy()
                fig=plot_vdfs_tb(temp[0,0,:,:,:], reconstructions[0,0,:,:,:] , vdf_vmin=1e-16)
                writer.add_figure('State',fig,counter)
                counter+=1

            if ((batch_idx + 1) % eval_every) == 0 and global_rank == 0:
                print(f"epoch: {epoch}\nbatch_idx: {batch_idx + 1}", flush=True)
                avg_train_loss = total_train_loss / n_train
                if avg_train_loss < best_train_loss:
                    best_train_loss = avg_train_loss

                print(f"best_train_loss: {best_train_loss}")
                print(f"recon_error: {total_recon_error / n_train}\n")
                total_train_loss = 0
                total_recon_error = 0
                n_train = 0
            if (epoch%10==0 and global_rank==0):
                torch.save(model.state_dict(), f"model_state_{epoch}.ptch")

    if global_rank==0:
        torch.save(model.state_dict(), "model_state_final.ptch")

if __name__ == "__main__":
    main()

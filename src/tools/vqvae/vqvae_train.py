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



def main():
    mp.set_start_method('spawn')


    dist.init_process_group(backend='nccl')
    rank = int(os.environ['LOCAL_RANK'])
    global_rank = int(os.environ['RANK'])
    torch.cuda.set_device(rank)

    filename=sys.argv[1]
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
    model = DDP(model, device_ids=[rank])

    batch_size = 128
    epochs = 2
    workers = 32
    cids = np.arange(1, 2000)
    #VDF_Data = Vlasiator_DataSet(cids, filename, device)
    # VDF_Data = MMapped_Vlasiator_DataSet(cids,filename,device,box=25)
    VDF_Data = Lazy_Vlasiator_DataSet(cids,filename,device,box=25)
    # train_sampler = DistributedSampler(VDF_Data, rank=rank,device_ids=[rank])
    train_sampler = DistributedSampler(VDF_Data, rank=rank)
    train_loader = DataLoader(
        dataset=VDF_Data,
        batch_size=batch_size,
        sampler=train_sampler,
        num_workers=workers,
        pin_memory=False
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



    for epoch in range(epochs):
        train_sampler.set_epoch(epoch)
        total_train_loss = 0
        total_recon_error = 0
        n_train = 0
        for batch_idx, train_tensors in enumerate(train_loader):
            optimizer.zero_grad()
            imgs = train_tensors[0].unsqueeze(0).to(rank)
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
            if (epoch%16==0 and global_rank==0):
                torch.save(model.state_dict(), f"model_state_{epoch}.ptch")

    if global_rank==0:
        torch.save(model.state_dict(), "model_state_final.ptch")

if __name__ == "__main__":
    main()

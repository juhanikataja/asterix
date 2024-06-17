import sys
from vqvae_tools import * 
import numpy as np
import torch
from torch.utils.data import DataLoader

filename = sys.argv[1] 
model_checkpoint = sys.argv[2]  

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
model = VQVAE(**model_args).to(device)
state_dict = torch.load(model_checkpoint)
model.eval()

cids = np.arange(1, 30)  
VDF_Data = Vlasiator_DataSet(cids, filename, device)
inference_loader = DataLoader(
    dataset=VDF_Data,
    batch_size=1,  
    num_workers=0,
    pin_memory=False
)

predictions = []
with torch.no_grad():
    for batch_idx, inference_tensors in enumerate(inference_loader):
        imgs = inference_tensors[0].unsqueeze(0).to(device)
        out = model(imgs)
        reconstructions = out["x_recon"].cpu().numpy()
        predictions.append(reconstructions)
        print(f"Inference done for batch {batch_idx + 1}")


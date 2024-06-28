
def main():
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm
    from matplotlib.colors import LogNorm

    size=50
    vdf_file=sys.argv[1]
    recon_file=sys.argv[2]
    vdf=np.fromfile(vdf_file,dtype=np.double)
    recon_vdf=np.fromfile(recon_file,dtype=np.double)
    vdf[vdf<1e-15]=0
    recon_vdf[recon_vdf<1e-15]=0
    vdf=np.reshape(vdf,(size,size,size))
    recon_vdf=np.reshape(recon_vdf,(size,size,size))
    fig, (ax1, ax2,ax3) = plt.subplots(1, 3)
    im1=ax1.imshow(vdf[:,:,size//2],norm=LogNorm())
    im2=ax2.imshow(recon_vdf[:,:,size//2],norm=LogNorm())
    ax3.plot(vdf[:,size//2,size//2],label='Original')
    ax3.plot(recon_vdf[:,size//2,size//2],label='tinyAI2')
    plt.colorbar(im1)
    plt.colorbar(im2)
    ax3.legend()
    ax3.set_yscale('log')
    # ax1.imshow(vdf[:,:,100],norm=LogNorm(1e-16,1e-5))
    plt.tight_layout()
    # ax2.imshow(recon_vdf[:,:,100],norm=LogNorm(1e-16,1e-5))


    ax1.set_title("Original VDF")
    ax2.set_title("Reconstructed VDF")
    plt.savefig("temp.png")
    

import os,sys
if ( len(sys.argv)<3 ):
    print(f"Usage: ./{sys.argv[0]} <vdf> <reconstructed_vdf>")
    sys.exit()
main()

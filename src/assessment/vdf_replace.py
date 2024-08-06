import sys, os
import numpy as np
from tqdm import tqdm
import xml.etree.ElementTree as ET
import struct
import pytools as pt
import numpy as np
import vdf_extract
import mlp_compress
from mpi4py import MPI
import shutil
import struct
import compression_methods as cm


def clone_file(vlsvReader, dst):
    src = vlsvReader.file_name
    print(f"Duplicating Reader File from {src} to {dst}")
    return shutil.copy2(src, dst)


def generate_tag(tag, arraysize, datasize, datatype, mesh, name, vectorsize, content):
    tag_template = (
        '<{tag} arraysize="{arraysize}" datasize="{datasize}" datatype="{datatype}" '
        'mesh="{mesh}" name="{name}" vectorsize="{vectorsize}">{content}</{tag}>\n'
    )
    return tag_template.format(
        tag=tag,
        arraysize=arraysize,
        datasize=datasize,
        datatype=datatype,
        mesh=mesh,
        name=name,
        vectorsize=vectorsize,
        content=content,
    )


# Taken from https://github.com/fmihpc/analysator/blob/60f7ca86a2f66e7798c83799301ca0e6d194f8b9/pyVlsv/vlsvwriter.py#L419
def xml_footer_indent(elem, level=0):
    i = "\n" + level * "   "
    if len(elem):
        if not elem.text or not elem.text.strip():
            elem.text = i + "   "
        if not elem.tail or not elem.tail.strip():
            elem.tail = i
        for elem in elem:
            xml_footer_indent(elem, level + 1)
        if not elem.tail or not elem.tail.strip():
            elem.tail = i
    else:
        if level and (not elem.tail or not elem.tail.strip()):
            elem.tail = i


def write_and_update_xml(fptr, xml, cellid, blocks_and_values, bpc):
    """
    fptr: file pointer
    xml: xml footer
    cellid: list of cellids to write
    blocks_and_values: list of blocks and values [block1..., block1_values...,]
    bpc: blocks per cell which is constant for all cells as we do not sparsify here
    """
    cells_with_blocks = np.array([cellid])
    number_of_blocks = len(blocks_and_values)
    blocks_per_cell = np.array([cellid])
    blocks_per_cell[:] = bpc
    bytes_written = 0

    tag = generate_tag(
        "CELLSWITHBLOCKS",
        cells_with_blocks.size,
        8,
        "uint",
        "SpatialGrid",
        "proton",
        1,
        fptr.tell(),
    )
    xml.append(ET.fromstring(tag))
    data = np.atleast_1d(cells_with_blocks)
    data.tofile(fptr)
    bytes_written += data.nbytes

    tag = generate_tag(
        "BLOCKSPERCELL",
        blocks_per_cell.size,
        4,
        "uint",
        "SpatialGrid",
        "proton",
        1,
        fptr.tell(),
    )
    xml.append(ET.fromstring(tag))
    data = np.atleast_1d(blocks_per_cell).astype(np.uint32)
    data.tofile(fptr)
    bytes_written += data.nbytes

    data = np.atleast_1d(blocks_and_values[0][:]).astype(np.uint32)
    a, b = np.shape(data)
    tag = generate_tag(
        "BLOCKIDS",
        a * b,
        4,
        "uint",
        "SpatialGrid",
        "proton",
        1,
        fptr.tell(),
    )
    xml.append(ET.fromstring(tag))
    print(f"Writing {np.shape(data)} ,min={np.min(data)}, max= {np.max(data)}")
    data.tofile(fptr)
    bytes_written += data.nbytes

    data = np.atleast_1d(blocks_and_values[1][:])
    a, b, c = np.shape(data)
    tag = generate_tag(
        "BLOCKVARIABLE",
        a * b,
        4,
        "float",
        "SpatialGrid",
        "proton",
        c,
        fptr.tell(),
    )
    xml.append(ET.fromstring(tag))
    print(f"Writing block data {np.shape(data)} ,min={np.min(data)}, max= {np.max(data)}")
    data.tofile(fptr)
    print(data.dtype)
    bytes_written += data.nbytes

    # Update footer xml tag
    xml_footer_indent(xml)
    footer_loc = fptr.tell()
    fptr.write(ET.tostring(xml))
    return bytes_written, footer_loc


def add_reconstructed_velocity_space(dst, cellid, blocks_and_values, bpc):
    """
    dst: destination file
    cellid: list of cellids
    blocks_and_values: list of blocks and values [block1..., block1_values...,]
    bpc: blocks per cell which is constant for all cells as we do not sparsify here
    """
    f = open(dst, "r+b")
    f.seek(8)  # system endianess
    (offset,) = struct.unpack("Q", f.read(8))  # VLSV footer
    f.seek(offset)
    footer = f.read()
    footer_str = footer.decode("utf-8")
    footer_str = footer_str.replace("BLOCK", "_LOCK")
    xml = ET.fromstring(footer_str)
    f.seek(offset)  # go back to write position (before the VLSV tag)
    bytes_written, footer_loc = write_and_update_xml(f, xml, cellid, blocks_and_values, bpc)
    f.seek(8)
    # Update footer location
    np.array(footer_loc, dtype=np.uint64).tofile(f)
    return bytes_written


def add_reconstructed_velocity_space_mpi(dst, cellid, blocks_and_values, bpc):
    """
    dst: destination file
    cellid: list of cellids
    blocks_and_values: list of blocks and values [block1..., block1_values...,]
    bpc: blocks per cell which is constant for all cells as we do not sparsify here
    """
    world_comm = MPI.COMM_WORLD
    world_size = world_comm.Get_size()
    my_rank = world_comm.Get_rank()


    #Collectively open and write our data
    buffer = np.zeros(8, dtype=np.byte)
    f = MPI.File.Open(world_comm, dst,  MPI.MODE_RDWR )
    fsize = f.Get_size()

    #Footer offset
    f.Read_at(8,buffer)
    (footer_offset,) = struct.unpack("Q",buffer)  # VLSV footer

    
    #Footer
    buffer=np.zeros(fsize-footer_offset,dtype=np.byte)
    f.Read_at(footer_offset,buffer)
    footer_str=buffer.tobytes().decode('utf-8')
    footer_str = footer_str.replace("BLOCK", "_LOCK")
    xml = ET.fromstring(footer_str)

    #CELLIDS
    total_cids=world_comm.gather(len(cellid), root=0)
    total_cids=world_comm.bcast(total_cids,root=0)
    curr_offset=footer_offset
    if (my_rank==0):

        tag = generate_tag(
            "CELLSWITHBLOCKS",
            np.uint64(np.sum(total_cids)),
            8,
            "uint",
            "SpatialGrid",
            "proton",
            1,
            footer_offset,
        )
        xml.append(ET.fromstring(tag))

    #Offsets
    byte_offset=8*np.sum(total_cids[0:my_rank]);
    my_offset=int(curr_offset+byte_offset)
    data=np.atleast_1d(cellid).astype(np.uint64)
    f.Write_at_all(my_offset,data)
    bytes_written=8*np.sum(total_cids)
    curr_offset+=bytes_written

   
    #Blocks per Cell
    blocks_per_cell = np.array([cellid])
    blocks_per_cell[:] = bpc
    if(my_rank==0):
        tag = generate_tag(
        "BLOCKSPERCELL",
        np.uint64(np.sum(total_cids)),
        4,
        "uint",
        "SpatialGrid",
        "proton",
        1,
        np.uint64(curr_offset),
        )
        xml.append(ET.fromstring(tag))

    byte_offset=4*np.sum(total_cids[0:my_rank]);
    my_offset=int(curr_offset+byte_offset)
    data=np.atleast_1d(blocks_per_cell).astype(np.uint32)
    f.Write_at_all(my_offset,data)
    bytes_written=4*np.sum(total_cids)
    curr_offset+=bytes_written

    
    #BlockIDs
    if(my_rank==0):
        tag = generate_tag(
            "BLOCKIDS",
            np.uint64(bpc*np.sum(total_cids)),
            4,
            "uint",
            "SpatialGrid",
            "proton",
            1,
            np.uint64(curr_offset),
        )
        xml.append(ET.fromstring(tag))    

    byte_offset=4*np.sum(total_cids[0:my_rank])*bpc;
    my_offset=int(curr_offset+byte_offset)
    data=np.atleast_1d(blocks_and_values[0]).astype(np.uint32)
    f.Write_at_all(my_offset,data)
    bytes_written=4*np.sum(total_cids)*bpc
    curr_offset+=bytes_written

    if (my_rank==0):
        tag = generate_tag(
            "BLOCKVARIABLE",
            np.uint64(bpc*np.sum(total_cids)),
            4,
            "float",
            "SpatialGrid",
            "proton",
            64,
            np.uint64(curr_offset),
        )
        xml.append(ET.fromstring(tag))        

    byte_offset=4*np.sum(total_cids[0:my_rank])*bpc*64;
    my_offset=int(curr_offset+byte_offset)
    data=np.atleast_1d(blocks_and_values[1]).astype(np.float32)
    f.Write_at_all(my_offset,data)
    bytes_written=4*np.sum(total_cids)*bpc*64
    curr_offset+=bytes_written
    f.Close()
    world_comm.barrier()

    if (my_rank==0):
        curr_offset=np.uint64(curr_offset)
        xml_footer_indent(xml)
        xml_data=ET.tostring(xml)
        f = open(dst, "r+b")
        f.seek(curr_offset)
        f.write(xml_data)
        f.seek(8)
        np.array(curr_offset, dtype=np.uint64).tofile(f)
        f.close()

    return


def reconstruct_vdf(f, cid,sparsity ,reconstruction_method):
    """
    f: VlsvReader Object
    len : boxed limits of vdfs that get reconstructed
    cid: the cellid to reconstruct
    reconstruction_method: function that performs the reconstruction
    """
    print(f"Extracting CellID {cid}")
    _, reconstructed = reconstruction_method(f, cid,sparsity)
    extents = f.get_velocity_mesh_extent()
    size = f.get_velocity_mesh_size()
    dv = f.get_velocity_mesh_dv()
    assert dv[0] == dv[1] == dv[2]
    dv = dv[0]
    WID = f.get_WID()

    blocks = np.arange(0, np.prod(size), dtype=np.int32)
    reconstructed = np.array(reconstructed, dtype=np.float32)
    block_data = np.zeros((np.prod(size), np.power(WID, 3)), dtype=np.float32)
    for blockid in blocks:
        block_coords = f.get_velocity_block_coordinates(blockid)
        rx = int(np.floor((block_coords[0] - extents[0]) / dv))
        ry = int(np.floor((block_coords[1] - extents[1]) / dv))
        rz = int(np.floor((block_coords[2] - extents[2]) / dv))
        for bx in range(0, WID):
            for by in range(0, WID):
                for bz in range(0, WID):
                    localid = bz * WID**2 + by * WID + bx
                    block_data[blockid, localid] = reconstructed[rz + bz, ry + by, rx + bx]

    return blocks, block_data


def reconstruct_vdf_debug(f, cid,sparsity ,reconstruction_method):
    import matplotlib.pyplot as plt
    import tools
    import vdf_extract
    """
    f: VlsvReader Object
    len : boxed limits of vdfs that get reconstructed
    cid: the cellid to reconstruct
    reconstruction_method: function that performs the reconstruction
    """
    print(f"Extracting CellID {cid}")
    _, reconstructed = reconstruction_method(f, cid,sparsity)
    _, original_vdf,_ = vdf_extract.extract(f, cid,sparsity,False)
    print(reconstructed.shape)
    print(original_vdf.shape)
    image_name=str(cid)+"_"+reconstruction_method.__name__+".png"
    tools.plot_vdfs(original_vdf, reconstructed, sparsity,True,image_name)
    extents = f.get_velocity_mesh_extent()
    size = f.get_velocity_mesh_size()
    dv = f.get_velocity_mesh_dv()
    assert dv[0] == dv[1] == dv[2]
    dv = dv[0]
    WID = f.get_WID()

    blocks = np.arange(0, np.prod(size), dtype=np.int32)
    reconstructed = np.array(reconstructed, dtype=np.float32)
    block_data = np.zeros((np.prod(size), np.power(WID, 3)), dtype=np.float32)
    for blockid in blocks:
        block_coords = f.get_velocity_block_coordinates(blockid)
        rx = int(np.floor((block_coords[0] - extents[0]) / dv))
        ry = int(np.floor((block_coords[1] - extents[1]) / dv))
        rz = int(np.floor((block_coords[2] - extents[2]) / dv))
        for bx in range(0, WID):
            for by in range(0, WID):
                for bz in range(0, WID):
                    localid = bz * WID**2 + by * WID + bx
                    block_data[blockid, localid] = reconstructed[rz + bz, ry + by, rx + bx]

    return blocks, block_data

def reconstruct_vdfs_mpi(filename, sparsity, reconstruction_method, output_file_name):
    """
    filename: file name to reconstuct
    len : boxed limits of vdfs that get reconstructed
    sparsity: sparsity to be applied to the reconstructed VDF. Block are not removed!
    """
    world_comm = MPI.COMM_WORLD
    world_size = world_comm.Get_size()
    my_rank = world_comm.Get_rank()
    f = pt.vlsvfile.VlsvReader(filename)
    WID = f.get_WID()
    size = f.get_velocity_mesh_size()
    total_cids = None
    num_cids = None
    if my_rank == 0:
        # cids = np.array(np.arange(1, 1 + np.prod(f.get_spatial_mesh_size())), dtype=int)
        cids=f.read(mesh="SpatialGrid",name="CellID", tag="VARIABLE")
        num_cids = cids.size
        assert num_cids % world_size == 0
        assert num_cids >= world_size
        total_cids = np.array_split(cids, world_size)
    world_comm.barrier()
    local_cids = world_comm.scatter(total_cids, root=0)
    print(f"Rank {my_rank} has {local_cids.size} cids!")
    world_comm.barrier()

    local_blocks = []
    local_block_data = []
    local_reconstructed_cids = []

    cnt = 0
    for cid in local_cids:
        a, b = reconstruct_vdf(f, cid,sparsity, reconstruction_method)
        # a, b = reconstruct_vdf_debug(f, cid,sparsity, reconstruction_method)
        local_reconstructed_cids.append(cid)
        local_blocks.append(a)
        local_block_data.append(b)
        cnt += 1
        # if cnt >= 1:
          # break

    world_comm.barrier()
    global_reconstructed_cids = world_comm.gather(local_reconstructed_cids, root=0)
    if (my_rank==0):
        output_file = clone_file(f, output_file_name)

    world_comm.barrier()
    add_reconstructed_velocity_space_mpi(
        output_file_name,
        local_reconstructed_cids,
        [local_blocks, local_block_data],
        np.prod(size),
    )

    world_comm.barrier()
    return

def main(file,sparsity):
    assert sparsity>=0
    f = pt.vlsvfile.VlsvReader(file)
    basename=os.path.basename(file)
    methods={
        cm.reconstruct_cid_fourier_mlp: "output_fourier_mlp_"+basename, 
        cm.reconstruct_cid_mlp: "output_mlp_"+basename, 
        cm.reconstruct_cid_zfp: "output_zfp_"+basename, 
        cm.reconstruct_cid_sph: "output_sph_"+basename, 
        cm.reconstruct_cid_cnn: "output_cnn_"+basename, 
        cm.reconstruct_cid_gmm: "output_gmm_"+basename, 
        cm.reconstruct_cid_dwt: "output_dwt_"+basename, 
        cm.reconstruct_cid_dct: "output_dct_"+basename, 
        cm.reconstruct_cid_pca: "output_pca_"+basename, 
        cm.reconstruct_cid_oct: "output_oct_"+basename, 
    }

    for method,output_file_name in methods.items():
        reconstruct_vdfs_mpi(file, sparsity, method,output_file_name)    

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("ERROR: Wrong usage!")
        print(f"USAGE: ./{sys.argv[0]} <vlsv file> <sparsity>")
        sys.exit()
        
    file = sys.argv[1]
    ext=os.path.splitext(file)[-1]
    if not os.path.isfile(file) or not (ext == ".vlsv"):
        print(f"ERROR: {file} is not a VLSV file !")
        sys.exit()
    sparsity = np.float64(sys.argv[2])
    main(file,sparsity)


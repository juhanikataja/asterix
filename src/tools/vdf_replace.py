import sys, os
from tqdm import tqdm
import xml.etree.ElementTree as ET
import struct
import pytools as pt
import numpy as np
import vdf_extract
import mlp_compress
import tools

if len(sys.argv) != 2:
    print(len(sys.argv))
    print("ERROR: Wrong usage!")
    print(f"USAGE: ./{sys.argv[0]} <vlsv file>")
    sys.exit()


def reconstruct_cid(f, cid,len):
    max_indexes, vdf = vdf_extract.extract(f, cid, len)
    nx, ny, nz = np.shape(vdf)
    assert nx == ny == nz
    sparsity = 1.0e-16
    if f.check_variable("MinValue"):
        sparsity = f.read_variable("proton" + "/EffectiveSparsityThreshold", cid)
    reconstructed_vdf = np.reshape(
        mlp_compress.compress_mlp_from_vec(vdf.flatten(), 12, 20, 2, 50, nx, sparsity),
        (nx, ny, nz),
    )
    mesh = f.get_velocity_mesh_size()
    final_vdf = np.zeros((int(4 * mesh[0]), int(4 * mesh[1]), int(4 * mesh[2])))
    final_vdf[
        max_indexes[0] - len : max_indexes[0] + len,
        max_indexes[1] - len : max_indexes[1] + len,
        max_indexes[2] - len : max_indexes[2] + len,
    ] = reconstructed_vdf
    _, vdf2 = vdf_extract.extract(f, cid, -1)
    tools.plot_vdfs(
        vdf2[
            max_indexes[0] - len : max_indexes[0] + len,
            max_indexes[1] - len : max_indexes[1] + len,
            max_indexes[2] - len : max_indexes[2] + len,
        ],
        final_vdf[
            max_indexes[0] - len : max_indexes[0] + len,
            max_indexes[1] - len : max_indexes[1] + len,
            max_indexes[2] - len : max_indexes[2] + len,
        ],
    )
    return cid, np.array(final_vdf, dtype=np.float32)


def clone_file(vlsvReader, dst):
    import shutil

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


def write_and_update_xml(fptr, xml, cellid, blocks_and_values):

    # Add back the reconstructed VDFs
    cells_with_blocks = np.array([cellid])
    number_of_blocks = len(blocks_and_values)
    blocks_per_cell = np.array([15625])
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
    data = np.atleast_1d(blocks_per_cell)
    data.tofile(fptr)
    bytes_written += data.nbytes

    tag = generate_tag(
        "BLOCKIDS",
        len(blocks_and_values[0]),
        4,
        "uint",
        "SpatialGrid",
        "proton",
        1,
        fptr.tell(),
    )
    xml.append(ET.fromstring(tag))
    data = np.atleast_1d(blocks_and_values[0])
    print(f"Writing {np.shape(data)} ,min={np.min(data)}, max= {np.max(data)}")
    data.tofile(fptr)
    bytes_written += data.nbytes

    tag = generate_tag(
        "BLOCKVARIABLE",
        len(blocks_and_values[0]),
        4,
        "float",
        "SpatialGrid",
        "proton",
        64,
        fptr.tell(),
    )
    xml.append(ET.fromstring(tag))
    data = np.atleast_1d(blocks_and_values[1])
    print(f"Writing block data {np.shape(data)} ,min={np.min(data)}, max= {np.max(data)}")
    data.tofile(fptr)
    bytes_written += data.nbytes

    # Update footer xml tag
    xml_footer_indent(xml)
    footer_loc = fptr.tell()
    fptr.write(ET.tostring(xml))
    return bytes_written, footer_loc


def add_reconstructed_velocity_space(dst, cellid, blocks_and_values):
    import struct

    f = open(dst, "r+b")
    f.seek(8)  # system endianess
    (offset,) = struct.unpack("Q", f.read(8))  # VLSV footer
    f.seek(offset)
    footer = f.read()
    footer_str = footer.decode("utf-8")
    footer_str = footer_str.replace("BLOCK", "_LOCK")
    xml = ET.fromstring(footer_str)
    f.seek(offset)  # go back to write position (before the VLSV tag)
    bytes_written, footer_loc = write_and_update_xml(f, xml, cellid, blocks_and_values)
    f.seek(8)
    # Update footer location
    np.array(footer_loc, dtype=np.uint64).tofile(f)
    return bytes_written


def reconstruct_vdfs(file):
    f = pt.vlsvfile.VlsvReader(file)
    cids = np.array(np.arange(1, 1 + np.prod(f.get_spatial_mesh_size())), dtype=int)
    print("Cell IDs to replace =", cids)
    for cid in cids:
        print(f"Extracting CellID {cid}")
        len=25
        _, reconstructed = reconstruct_cid(f, cid,len)
        # _,reconstructed=vdf_extract.extract(f,cid,-1)
        # reconstructed=np.array(reconstructed,dtype=np.float32)

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
                        block_data[blockid, localid] = reconstructed[
                            rz + bz, ry + by, rx + bx
                        ]

        output_file = clone_file(f, "output.vlsv")
        bytes_written = add_reconstructed_velocity_space(
            output_file, 1, [blocks, block_data]
        )
        beautyfied = "{:.4f}".format(bytes_written / (1024 * 1024 * 1024))
        print(f"Wrote  {beautyfied} GB to {output_file}!")


file = sys.argv[1]
reconstruct_vdfs(file)

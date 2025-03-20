import operator
import time
from datetime import datetime
from itertools import product
from typing import Optional
from util import get_process_slice
import math

import dspaces as ds
import numpy as np
from mpi4py import MPI

from config import DEBUG, element_dtype, n_dims, procs, shape

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
num_procs = comm.Get_size()

if DEBUG:
    assert len(shape) == len(procs), (
        f"Expected len(dims) {n_dims}, but got {len(shape)}"
    )
    assert len(shape) == len(procs), (
        f"Expected len(procs) {n_dims}, but got {len(shape)}"
    )


# data = np.random.rand(*shape).astype(element_dtype)
data = np.empty(shape, dtype=element_dtype)
if DEBUG:
    assert data.shape == shape, f"Expected shape {shape}, but produced {data.shape}"
    assert data.dtype == element_dtype, (
        f"Expected dtype {element_dtype}, but produced {data.dtype}"
    )
    if rank == 0:
        print(data)


if DEBUG and rank == 0:
    # print(f"num_procs: {num_procs}")
    assert num_procs == math.prod(procs), f"Expected {math.prod(procs)} processes, but got {num_procs}"

client = ds.DSClient()

if DEBUG:
    assert client, "Failed to create DataSpaces client"

# Mandatory to define for non-zero ranks to receive the broadcasted values
current_unix_time: Optional[int] = None
data_var_name :Optional[str] = None

if rank == 0:
    current_unix_time = int(time.time())
    data_var_name = f"data_{str(datetime.now())}"


current_unix_time = comm.bcast(current_unix_time, root=0)
data_var_name = comm.bcast(data_var_name, root=0)

if DEBUG:
    assert current_unix_time, "Broadcast of current_unix_time failed"
    assert data_var_name, "Broadcast of data_var_name failed"

slice, offset = get_process_slice(process_num=rank, processes_per_dim=procs, array_shape=shape)
current_process_data = data[slice]

if DEBUG:
    print(f"Rank {rank}: Shape of slice: {current_process_data.shape}")

client.Put(
    current_process_data,
    data_var_name,
    version=current_unix_time,
    offset=offset,
)

if DEBUG:
    comm.barrier()
    time.sleep(2)
    if rank == 0:
        print()
        fetched_data = client.Get(
            data_var_name,
            version=current_unix_time,
            lb=(0,)*n_dims,
            ub=tuple(map(lambda x: operator.sub(x, 1), shape)),
            timeout=10,
        )
        print(f"fetched_data: {fetched_data}")


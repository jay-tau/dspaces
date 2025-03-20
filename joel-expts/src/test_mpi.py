import sys
import time
from datetime import datetime

import dspaces as ds
import numpy as np
from mpi4py import MPI

if len(sys.argv) != 2:
    print("Usage: mpiexec -n n_procs python test_mpi.py array_size_exponent")

ARRAY_SIZE = int(2 ** int(sys.argv[1]))

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
num_procs = comm.Get_size()


chunk_size = int(ARRAY_SIZE / num_procs)

if rank == 0:
    print(f"Array size: {ARRAY_SIZE}")
    print(f"num_procs: {num_procs}")
    print(f"chunk_size: {chunk_size}")
    print()

assert ARRAY_SIZE % num_procs == 0, (
    f"Rank count must evenly divide array size\nArray size: {ARRAY_SIZE}, size: {num_procs}"
)

client = ds.DSClient()  # Initialize DataSpaces library - defaults to using COMM_WORLD

data = np.arange(ARRAY_SIZE)

current_unix_time = None
data_var_name = None

if rank == 0:
    current_unix_time = int(time.time())
    data_var_name = f"data_{str(datetime.now())}"

current_unix_time = comm.bcast(current_unix_time, root=0)
data_var_name = comm.bcast(data_var_name, root=0)

slice_start = rank * chunk_size
slice_end = slice_start + chunk_size

print(f"Rank {rank}:\t data[{slice_start}:{slice_end}]")


time.sleep(2)

client.Put(
    data[slice_start:slice_end],
    data_var_name,
    version=current_unix_time,
    offset=(slice_start,),
)

print(f"Rank {rank} written")


comm.Barrier()

if rank == 0:
    # print(data)
    print()
    # print(f"{data_var_name} written.")

    fetched_data = client.Get(
        data_var_name,
        version=current_unix_time,
        lb=(0,),
        ub=(ARRAY_SIZE - 1,),
        timeout=10000,
    )
    print(f"fetched_data: {fetched_data}")
    pass

print(data_var_name, current_unix_time)

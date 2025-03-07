from mpi4py import MPI
import dspaces as ds
import numpy as np
import time

# Initialize MPI and DataSpaces
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()
client = ds.DSClient()

# Generate data
array_size = 100
local_size = array_size // size

# Benchmark parameters
iterations = 5

# Define array dimensions for testing - adjusted for better division
dims = [(100,), (10, 10), (4, 4, 4)]  # 1D, 2D, 3D arrays
patterns = ['contiguous', 'strided', 'random']
timings = {f"{len(dim)}D_{p}": [] for dim in dims for p in patterns}

for dim in dims:
    ndim = len(dim)
    total_size = np.prod(dim)
    # Calculate local size for first dimension only
    local_dim = list(dim)
    local_dim[0] = dim[0] // size
    
    for pattern in patterns:
        for i in range(iterations):
            if pattern == 'contiguous':
                data = np.random.rand(*local_dim)
                offset = tuple(0 if j > 0 else rank * local_dim[0] for j in range(ndim))
            
            elif pattern == 'strided':
                stride_dim = list(local_dim)
                stride_dim[0] = stride_dim[0] // 2
                data = np.random.rand(*stride_dim)
                offset = tuple(0 if j > 0 else rank * local_dim[0] * 2 for j in range(ndim))
            
            else:  # random
                data = np.random.rand(*local_dim)
                max_offset = dim[0] - local_dim[0]
                rand_pos = np.random.randint(0, max_offset + 1) if max_offset > 0 else 0
                offset = tuple(0 if j > 0 else rand_pos for j in range(ndim))

            start_time = time.time()
            client.Put(data, f"{ndim}D_{pattern}_data", version=i, offset=offset)
            end_time = time.time()
            timings[f"{ndim}D_{pattern}"].append(end_time - start_time)

if rank == 0:
    for dim in dims:
        ndim = len(dim)
        for pattern in patterns:
            key = f"{ndim}D_{pattern}"
            avg_time = np.mean(timings[key])
            data_size = np.prod(dim) * 8  # size in bytes
            bandwidth = (data_size * iterations) / (np.sum(timings[key]) * 1024 * 1024)  # MB/s
            print(f"Writer {ndim}D {pattern} pattern - Avg time: {avg_time:.4f} s, "
                  f"Bandwidth: {bandwidth:.2f} MB/s")
    client.KillServer()

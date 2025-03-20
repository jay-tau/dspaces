import numpy as np


def get_process_slice(process_num, processes_per_dim, array_shape):
    # Convert inputs to numpy arrays
    processes_per_dim = np.asarray(processes_per_dim)
    array_shape = np.asarray(array_shape)

    # Convert flat process number to multidimensional indices using NumPy
    indices = np.array(np.unravel_index(process_num, tuple(processes_per_dim)))

    # Calculate chunk sizes for each dimension (vectorized)
    chunk_sizes = array_shape // processes_per_dim
    remainders = array_shape % processes_per_dim

    # Calculate start and end indices for each dimension (vectorized)
    is_remainder_process = indices < remainders

    # Use NumPy's where() to efficiently calculate starts and ends
    starts = np.where(is_remainder_process,
                      indices * (chunk_sizes + 1),
                      indices * chunk_sizes + remainders)

    ends = np.where(is_remainder_process,
                    starts + chunk_sizes + 1,
                    starts + chunk_sizes)

    # Create slice objects
    slices = tuple(slice(int(start), int(end)) for start, end in zip(starts, ends))

    # Convert starts array to tuple of ints for the offset
    offset = tuple(int(start) for start in starts)

    return slices, offset

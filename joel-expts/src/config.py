import numpy as np
import numpy.typing as npt

DEBUG: bool = True

n_dims: int = 2
shape: tuple[int, ...] = (4,3)
procs: tuple[int, ...] = (2,3)
element_dtype: npt.DTypeLike = np.int64
data: npt.NDArray[npt.DTypeLike] = None

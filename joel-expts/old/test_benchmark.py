import pytest
import dspaces as ds
import numpy as np

class BenchmarkConfig:
    """Configuration parameters for DataSpaces benchmarks"""
    # Test dimensions: 1D (100), 2D (10x10), 3D (4x4x4)
    DIMS = [(100,), (10, 10), (4, 4, 4)]
    # Number of versions to maintain for read tests
    MAX_VERSIONS = 15

@pytest.fixture(scope="session")
def ds_client():
    """Create a DataSpaces client instance for the entire test session"""
    try:
        client = ds.DSClient(rank=0)
        yield client
    except Exception as e:
        print(f"Failed to initialize client: {str(e)}")
        raise

class BenchmarkHelper:
    """Helper methods for data generation and bounds calculation"""
    
    @staticmethod
    def create_data(dim, stride=False):
        """Generate random test data array
        
        Args:
            dim: Tuple of dimensions
            stride: If True, reduce each dimension by half
        Returns:
            numpy.ndarray: Random data array
        """
        if stride:
            dim = [d//2 for d in dim]
        return np.random.rand(*dim)
    
    @staticmethod
    def create_bounds(dim, pattern='contiguous', version=0):
        """Calculate lower and upper bounds for data access
        
        Args:
            dim: Tuple of dimensions
            pattern: Access pattern ('contiguous', 'strided', or 'random')
            version: Version number (used for random pattern seed)
        Returns:
            tuple: (lower_bounds, upper_bounds)
        """
        ndim = len(dim)
        
        if pattern == 'contiguous':
            # Full data range
            lb = tuple(0 for _ in range(ndim))
            ub = tuple(d-1 for d in dim)
        
        elif pattern == 'strided':
            # Half size in each dimension
            lb = tuple(0 for _ in range(ndim))
            ub = tuple((d//2)-1 for d in dim)
        
        else:  # random
            # Random starting point with half-size window
            np.random.seed(version)
            max_pos = [d - d//2 for d in dim]
            starts = [np.random.randint(0, max_p + 1) for max_p in max_pos]
            lb = tuple(start for start in starts)
            ub = tuple(start + (d//2)-1 for start, d in zip(starts, dim))
        
        return lb, ub

class TestWriteBenchmark:
    """Benchmark DataSpaces write operations"""
    
    @pytest.mark.parametrize("dim", BenchmarkConfig.DIMS)
    def test_contiguous_write(self, benchmark, ds_client, dim):
        """Benchmark contiguous write pattern"""
        version = [0]
        def run_write():
            data = BenchmarkHelper.create_data(dim)
            offset = tuple(0 for _ in range(len(dim)))
            ds_client.Put(data, f"{len(dim)}D_contiguous", version=version[0], offset=offset)
            version[0] += 1
        benchmark(run_write)

    @pytest.mark.parametrize("dim", BenchmarkConfig.DIMS)
    def test_strided_write(self, benchmark, ds_client, dim):
        """Benchmark strided write pattern"""
        version = [0]
        def run_write():
            data = BenchmarkHelper.create_data(dim, stride=True)
            offset = tuple(0 for _ in range(len(dim)))
            ds_client.Put(data, f"{len(dim)}D_strided", version=version[0], offset=offset)
            version[0] += 1
        benchmark(run_write)

    @pytest.mark.parametrize("dim", BenchmarkConfig.DIMS)
    def test_random_write(self, benchmark, ds_client, dim):
        """Benchmark random write pattern"""
        version = [0]
        def run_write():
            data = BenchmarkHelper.create_data(dim)
            lb, _ = BenchmarkHelper.create_bounds(dim, 'random', version[0])
            ds_client.Put(data, f"{len(dim)}D_random", version=version[0], offset=lb)
            version[0] += 1
        benchmark(run_write)

class TestReadBenchmark:
    """Benchmark DataSpaces read operations"""
    
    @pytest.mark.parametrize("dim", BenchmarkConfig.DIMS)
    def test_contiguous_read(self, benchmark, ds_client, dim):
        """Benchmark contiguous read pattern"""
        # Prepare test data
        self._populate_test_data(ds_client, dim, 'contiguous')
        
        # Set up read test
        version = [0]
        lb, ub = BenchmarkHelper.create_bounds(dim, 'contiguous')
        
        def run_read():
            result = ds_client.Get(f"{len(dim)}D_contiguous", version=version[0], 
                                 lb=lb, ub=ub, timeout=-1)
            version[0] = (version[0] + 1) % BenchmarkConfig.MAX_VERSIONS
            return result
        
        benchmark(run_read)

    @pytest.mark.parametrize("dim", BenchmarkConfig.DIMS)
    def test_strided_read(self, benchmark, ds_client, dim):
        """Benchmark strided read pattern"""
        # Prepare test data
        self._populate_test_data(ds_client, dim, 'strided')
        
        # Set up read test
        version = [0]
        lb, ub = BenchmarkHelper.create_bounds(dim, 'strided')
        
        def run_read():
            result = ds_client.Get(f"{len(dim)}D_strided", version=version[0], 
                                 lb=lb, ub=ub, timeout=-1)
            version[0] = (version[0] + 1) % BenchmarkConfig.MAX_VERSIONS
            return result
        
        benchmark(run_read)

    @pytest.mark.parametrize("dim", BenchmarkConfig.DIMS)
    def test_random_read(self, benchmark, ds_client, dim):
        """Benchmark random read pattern"""
        # Prepare test data
        self._populate_test_data(ds_client, dim, 'random')
        
        # Set up read test
        version = [0]
        
        def run_read():
            lb, ub = BenchmarkHelper.create_bounds(dim, 'random', version[0])
            result = ds_client.Get(f"{len(dim)}D_random", version=version[0], 
                                 lb=lb, ub=ub, timeout=-1)
            version[0] = (version[0] + 1) % BenchmarkConfig.MAX_VERSIONS
            return result
        
        benchmark(run_read)
    
    def _populate_test_data(self, ds_client, dim, pattern):
        """Helper method to populate test data for read benchmarks
        
        Args:
            ds_client: DataSpaces client
            dim: Tuple of dimensions
            pattern: Access pattern type
        """
        for i in range(BenchmarkConfig.MAX_VERSIONS):
            data = BenchmarkHelper.create_data(dim, stride=(pattern == 'strided'))
            if pattern == 'random':
                lb, _ = BenchmarkHelper.create_bounds(dim, pattern, i)
                offset = lb
            else:
                offset = tuple(0 for _ in range(len(dim)))
            ds_client.Put(data, f"{len(dim)}D_{pattern}", version=i, offset=offset)

from typing import Callable, Optional

import numba
from numba import cuda

# From https://github.com/googlecolab/colabtools/issues/5081
from numba import config
# No longer necessary for numba-cuda>=0.16
# config.CUDA_ENABLE_PYNVJITLINK = 1
config.CUDA_LOW_OCCUPANCY_WARNINGS = 0

from .tensor import Tensor
from .tensor_data import (
    MAX_DIMS,
    Shape,
    Storage,
    Strides,
    TensorData,
    broadcast_index,
    index_to_position,
    shape_broadcast,
    to_index,
)
from .tensor_ops import MapProto, TensorOps

# This code will CUDA compile fast versions your tensor_data functions.
# If you get an error, read the docs for NUMBA as to what is allowed
# in these functions.

to_index = cuda.jit(device=True)(to_index)
index_to_position = cuda.jit(device=True)(index_to_position)
broadcast_index = cuda.jit(device=True)(broadcast_index)

THREADS_PER_BLOCK = 32


class CudaOps(TensorOps):
    cuda = True

    @staticmethod
    def map(fn: Callable[[float], float]) -> MapProto:
        "See `tensor_ops.py`"
        f = tensor_map(cuda.jit(device=True)(fn))

        def ret(a: Tensor, out: Optional[Tensor] = None) -> Tensor:
            if out is None:
                out = a.zeros(a.shape)

            # Instantiate and run the cuda kernel.
            threadsperblock = THREADS_PER_BLOCK
            blockspergrid = (out.size + THREADS_PER_BLOCK - 1) // THREADS_PER_BLOCK
            f[blockspergrid, threadsperblock](*out.tuple(), out.size, *a.tuple())  # type: ignore
            return out

        return ret

    @staticmethod
    def zip(fn: Callable[[float, float], float]) -> Callable[[Tensor, Tensor], Tensor]:
        f = tensor_zip(cuda.jit(device=True)(fn))

        def ret(a: Tensor, b: Tensor) -> Tensor:
            c_shape = shape_broadcast(a.shape, b.shape)
            out = a.zeros(c_shape)
            threadsperblock = THREADS_PER_BLOCK
            blockspergrid = (out.size + (threadsperblock - 1)) // threadsperblock
            f[blockspergrid, threadsperblock](  # type: ignore
                *out.tuple(), out.size, *a.tuple(), *b.tuple()
            )
            return out

        return ret

    @staticmethod
    def reduce(
        fn: Callable[[float, float], float], start: float = 0.0
    ) -> Callable[[Tensor, int], Tensor]:
        f = tensor_reduce(cuda.jit(device=True)(fn))

        def ret(a: Tensor, dim: int) -> Tensor:
            out_shape = list(a.shape)
            # Based off this, it seems like we should be parallelizing
            # over the reduce dimension, and if that dimension has
            # cardinality > 1024 we will end up with cardinality > 1
            # in the output
            out_shape[dim] = (a.shape[dim] - 1) // 1024 + 1
            out_a = a.zeros(tuple(out_shape))

            threadsperblock = 1024
            # Ok, so we have a lot of blocks, seemingly one block
            # per element in the dimensions other than the reduce dimension
            # So, it seems like each block will be dedicated to one
            # column over the reduce dimension, with each thread
            # handling individual values in that column
            blockspergrid = out_a.size
            f[blockspergrid, threadsperblock](  # type: ignore
                *out_a.tuple(), out_a.size, *a.tuple(), dim, start
            )

            return out_a

        return ret

    @staticmethod
    def matrix_multiply(a: Tensor, b: Tensor) -> Tensor:
        # Make these always be a 3 dimensional multiply
        both_2d = 0
        if len(a.shape) == 2:
            a = a.contiguous().view(1, a.shape[0], a.shape[1])
            both_2d += 1
        if len(b.shape) == 2:
            b = b.contiguous().view(1, b.shape[0], b.shape[1])
            both_2d += 1
        both_2d = both_2d == 2

        ls = list(shape_broadcast(a.shape[:-2], b.shape[:-2]))
        ls.append(a.shape[-2])
        ls.append(b.shape[-1])
        assert a.shape[-1] == b.shape[-2]
        out = a.zeros(tuple(ls))

        # One block per batch, extra rows, extra col
        blockspergrid = (
            (out.shape[1] + (THREADS_PER_BLOCK - 1)) // THREADS_PER_BLOCK,
            (out.shape[2] + (THREADS_PER_BLOCK - 1)) // THREADS_PER_BLOCK,
            out.shape[0],
        )
        threadsperblock = (THREADS_PER_BLOCK, THREADS_PER_BLOCK, 1)

        print("blockspergrid", blockspergrid)
        print("threadsperblock", threadsperblock)
        print("a")
        print(a)
        print("b")
        print(b)
        print("out")
        print(out)

        tensor_matrix_multiply[blockspergrid, threadsperblock](
            *out.tuple(), out.size, *a.tuple(), *b.tuple()
        )

        # Undo 3d if we added it.
        if both_2d:
            out = out.view(out.shape[1], out.shape[2])
        
        print("out post")
        print(out)

        return out


# Implement


def tensor_map(
    fn: Callable[[float], float]
) -> Callable[[Storage, Shape, Strides, Storage, Shape, Strides], None]:
    """
    CUDA higher-order tensor map function. ::

      fn_map = tensor_map(fn)
      fn_map(out, ... )

    Args:
        fn: function mappings floats-to-floats to apply.

    Returns:
        Tensor map function.
    """

    def _map(
        out: Storage,
        out_shape: Shape,
        out_strides: Strides,
        out_size: int,
        in_storage: Storage,
        in_shape: Shape,
        in_strides: Strides,
    ) -> None:

        out_index = cuda.local.array(MAX_DIMS, numba.int32)
        in_index = cuda.local.array(MAX_DIMS, numba.int32)
        i = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
        
        if i < out_size:
            # Get index from ordinal
            to_index(i, out_shape, out_index)
            # Convert index to position in storage (taking into account strides)
            out_position = index_to_position(out_index, out_strides)

            # Get broadcast in  index from out index
            broadcast_index(out_index, out_shape, in_shape, in_index)
            # Convert index to position in storage (taking into account strides)
            in_position = index_to_position(in_index, in_strides)

            # Apply function
            out[out_position] = fn(in_storage[in_position])

    return cuda.jit()(_map)  # type: ignore


def tensor_zip(
    fn: Callable[[float, float], float]
) -> Callable[
    [Storage, Shape, Strides, Storage, Shape, Strides, Storage, Shape, Strides], None
]:
    """
    CUDA higher-order tensor zipWith (or map2) function ::

      fn_zip = tensor_zip(fn)
      fn_zip(out, ...)

    Args:
        fn: function mappings two floats to float to apply.

    Returns:
        Tensor zip function.
    """

    def _zip(
        out: Storage,
        out_shape: Shape,
        out_strides: Strides,
        out_size: int,
        a_storage: Storage,
        a_shape: Shape,
        a_strides: Strides,
        b_storage: Storage,
        b_shape: Shape,
        b_strides: Strides,
    ) -> None:

        out_index = cuda.local.array(MAX_DIMS, numba.int32)
        a_index = cuda.local.array(MAX_DIMS, numba.int32)
        b_index = cuda.local.array(MAX_DIMS, numba.int32)
        i = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x

        if i < out_size:
            # Get index from ordinal
            to_index(i, out_shape, out_index)
            # Convert index to position in storage (taking into account strides)
            out_position = index_to_position(out_index, out_strides)

            # Get broadcast a index from out index
            broadcast_index(out_index, out_shape, a_shape, a_index)
            # Convert index to position in storage (taking into account strides)
            a_position = index_to_position(a_index, a_strides)

            # Get broadcast b index from out index
            broadcast_index(out_index, out_shape, b_shape, b_index)
            # Convert index to position in storage (taking into account strides)
            b_position = index_to_position(b_index, b_strides)

            # Apply function
            out[out_position] = fn(a_storage[a_position], b_storage[b_position])

    return cuda.jit()(_zip)  # type: ignore


def _sum_practice(out: Storage, a: Storage, size: int) -> None:
    """
    This is a practice sum kernel to prepare for reduce.

    Given an array of length $n$ and out of size $n // \text{blockDIM}$
    it should sum up each blockDim values into an out cell.

    $[a_1, a_2, ..., a_{100}]$

    |

    $[a_1 +...+ a_{31}, a_{32} + ... + a_{64}, ... ,]$

    Note: Each block must do the sum using shared memory!

    Args:
        out (Storage): storage for `out` tensor.
        a (Storage): storage for `a` tensor.
        size (int):  length of a.

    """
    BLOCK_DIM = 32

    cache = cuda.shared.array(BLOCK_DIM, numba.float64)
    i = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
    pos = cuda.threadIdx.x

    if i < size:
        cache[pos] = a[i]
    else:
        cache[pos] = 0
    
    cuda.syncthreads()

    if pos % 2 == 0:
        cache[pos] += cache[pos + 1]

    cuda.syncthreads()

    if pos % 4 == 0:
        cache[pos] += cache[pos + 2]
    
    cuda.syncthreads()

    if pos % 8 == 0:
        cache[pos] += cache[pos + 4]
    
    cuda.syncthreads()

    if pos % 16 == 0:
        cache[pos] += cache[pos + 8]
    
    cuda.syncthreads()

    if pos == 0:
        cache[pos] += cache[pos + 16]
        out[cuda.blockIdx.x] = cache[pos]


jit_sum_practice = cuda.jit()(_sum_practice)


def sum_practice(a: Tensor) -> TensorData:
    (size,) = a.shape
    threadsperblock = THREADS_PER_BLOCK
    blockspergrid = (size // THREADS_PER_BLOCK) + 1
    out = TensorData([0.0 for i in range(2)], (2,))
    out.to_cuda_()
    jit_sum_practice[blockspergrid, threadsperblock](
        out.tuple()[0], a._tensor._storage, size
    )
    return out


def tensor_reduce(
    fn: Callable[[float, float], float]
) -> Callable[[Storage, Shape, Strides, Storage, Shape, Strides, int], None]:
    """
    CUDA higher-order tensor reduce function.

    Args:
        fn: reduction function maps two floats to float.

    Returns:
        Tensor reduce function.

    """

    def _reduce(
        out: Storage,
        out_shape: Shape,
        out_strides: Strides,
        out_size: int,
        a_storage: Storage,
        a_shape: Shape,
        a_strides: Strides,
        reduce_dim: int,
        reduce_value: float,
    ) -> None:
        BLOCK_DIM = 1024
        cache = cuda.shared.array(BLOCK_DIM, numba.float64)
        out_index = cuda.local.array(MAX_DIMS, numba.int32)
        out_pos = cuda.blockIdx.x
        out_pos_wrap = out_pos % out_size
        out_pos_num = out_pos // out_size
        pos = cuda.threadIdx.x

        out_reduce_stride = out_strides[reduce_dim]
        a_reduce_stride = a_strides[reduce_dim]

        reduce_size = a_shape[reduce_dim]

        # Get index from ordinal
        to_index(out_pos, out_shape, out_index)
        # Convert index to position in storage (taking into account strides)
        out_position = index_to_position(out_index, out_strides)

        out_index[reduce_dim] *= BLOCK_DIM
        a_position = index_to_position(out_index, a_strides)

        if pos < reduce_size:
            cache[pos] = a_storage[a_position + a_reduce_stride * pos]
        else:
            cache[pos] = reduce_value

        cuda.syncthreads()

        if pos % 2 == 0:
            cache[pos] = fn(cache[pos], cache[pos + 1])

        cuda.syncthreads()

        if pos % 4 == 0:
            cache[pos] = fn(cache[pos], cache[pos + 2])

        cuda.syncthreads()

        if pos % 8 == 0:
            cache[pos] = fn(cache[pos], cache[pos + 4])

        cuda.syncthreads()

        if pos % 16 == 0:
            cache[pos] = fn(cache[pos], cache[pos + 8])

        cuda.syncthreads()

        if pos % 32 == 0:
            cache[pos] = fn(cache[pos], cache[pos + 16])

        cuda.syncthreads()

        if pos % 64 == 0:
            cache[pos] = fn(cache[pos], cache[pos + 32])

        cuda.syncthreads()

        if pos % 128 == 0:
            cache[pos] = fn(cache[pos], cache[pos + 64])

        cuda.syncthreads()

        if pos % 256 == 0:
            cache[pos] = fn(cache[pos], cache[pos + 128])

        cuda.syncthreads()

        if pos % 512 == 0:
            cache[pos] = fn(cache[pos], cache[pos + 256])

        cuda.syncthreads()

        if pos == 0:
            cache[pos] = fn(cache[pos], cache[pos + 512])
            out[out_position] = cache[pos]

    return cuda.jit()(_reduce)  # type: ignore


def _mm_practice(out: Storage, a: Storage, b: Storage, size: int) -> None:
    """
    This is a practice square MM kernel to prepare for matmul.

    Given a storage `out` and two storage `a` and `b`. Where we know
    both are shape [size, size] with strides [size, 1].

    Size is always < 32.

    Requirements:

    * All data must be first moved to shared memory.
    * Only read each cell in `a` and `b` once.
    * Only write to global memory once per kernel.

    Compute

    ```
     for i:
         for j:
              for k:
                  out[i, j] += a[i, k] * b[k, j]
    ```

    Args:
        out (Storage): storage for `out` tensor.
        a (Storage): storage for `a` tensor.
        b (Storage): storage for `b` tensor.
        size (int): size of the square
    """
    BLOCK_DIM = 32
    CACHE_SIZE = 1024 # 32 * 32

    cache_a = cuda.shared.array(CACHE_SIZE, numba.float64)
    cache_b = cuda.shared.array(CACHE_SIZE, numba.float64)

    x_stride = size
    y_stride = 1

    # Get our x,y thread position in the matrix
    x, y = cuda.grid(2)
    # Strides are [size, 1], so we calculate our index in the storage
    out_idx = x * x_stride + y * y_stride

    # Shouldn't be too bad, the matrix is square and < 32 elements wide
    if x < size and y < size:
        # Move both matrices into shared memory
        cache_a[out_idx] = a[out_idx]
        cache_b[out_idx] = b[out_idx]
    
    cuda.syncthreads()

    if x < size and y < size:
        # Starting indices
        a_idx = x * x_stride + 0 * y_stride
        b_idx = 0 * x_stride + y * y_stride

        val = 0

        for n in range(size):
            val += a[a_idx] * b[b_idx]

            # Move to next index!
            a_idx += y_stride
            b_idx += x_stride
        
        out[out_idx] = val



jit_mm_practice = cuda.jit()(_mm_practice)


def mm_practice(a: Tensor, b: Tensor) -> TensorData:
    (size, _) = a.shape
    threadsperblock = (THREADS_PER_BLOCK, THREADS_PER_BLOCK)
    blockspergrid = 1
    out = TensorData([0.0 for i in range(size * size)], (size, size))
    out.to_cuda_()
    jit_mm_practice[blockspergrid, threadsperblock](
        out.tuple()[0], a._tensor._storage, b._tensor._storage, size
    )
    return out


def _tensor_matrix_multiply(
    out: Storage,
    out_shape: Shape,
    out_strides: Strides,
    out_size: int,
    a_storage: Storage,
    a_shape: Shape,
    a_strides: Strides,
    b_storage: Storage,
    b_shape: Shape,
    b_strides: Strides,
) -> None:
    """
    CUDA tensor matrix multiply function.

    Requirements:

    * All data must be first moved to shared memory.
    * Only read each cell in `a` and `b` once.
    * Only write to global memory once per kernel.

    Should work for any tensor shapes that broadcast as long as ::

    ```python
    assert a_shape[-1] == b_shape[-2]
    ```
    Returns:
        None : Fills in `out`
    """
    a_batch_stride = a_strides[0] if a_shape[0] > 1 else 0
    b_batch_stride = b_strides[0] if b_shape[0] > 1 else 0
    # Batch dimension - fixed
    batch = cuda.blockIdx.z

    BLOCK_DIM = 32
    a_shared = cuda.shared.array((BLOCK_DIM, BLOCK_DIM), numba.float64)
    b_shared = cuda.shared.array((BLOCK_DIM, BLOCK_DIM), numba.float64)

    # The final position c[i, j]
    i = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
    j = cuda.blockIdx.y * cuda.blockDim.y + cuda.threadIdx.y

    # The local position in the block.
    pi = cuda.threadIdx.x
    pj = cuda.threadIdx.y

    # Code Plan:
    # 1) Move across shared dimension by block dim.
    #    a) Copy into shared memory for a matrix.
    #    b) Copy into shared memory for b matrix
    #    c) Compute the dot produce for position c[i, j]

    fuse_dim_size = a_shape[-1]
    a_fuse_stride = a_strides[-1]
    b_fuse_stride = b_strides[-2]

    # It seems like this function is only implemented to work in 3d?
    # So I think I can take some shortcuts/make some assumptions
    out_pos = i * out_strides[1] + j * out_strides[2] + batch * out_strides[0]

    # We'll be starting at zero y/x respectively
    a_pos = i * a_strides[1] + 0 * a_strides[2] + batch * a_strides[0]
    b_pos = 0 * b_strides[1] + j * b_strides[2] + batch * b_strides[0]

    val = 0

    for m in range(fuse_dim_size // BLOCK_DIM + 1):
        a_shared[pi][pj] = a_storage[a_pos] + pj * a_strides[2]
        b_shared[pi][pj] = b_storage[b_pos] + pi * b_strides[1]
    
        cuda.syncthreads()
    
        for n in range(min(BLOCK_DIM, fuse_dim_size - m * BLOCK_DIM)):
            # val += a_shared[pi][n] * b_shared[n][pj]
            val += 0.1
    
        a_pos += BLOCK_DIM * a_strides[2]
        b_pos += BLOCK_DIM * a_strides[1]
    
        cuda.syncthreads()
    
    out[out_pos] = val


tensor_matrix_multiply = cuda.jit(_tensor_matrix_multiply)

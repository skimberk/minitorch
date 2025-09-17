from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from numba import njit, prange

from .tensor_data import (
    MAX_DIMS,
    broadcast_index,
    index_to_position,
    shape_broadcast,
    to_index,
)
from .tensor_ops import MapProto, TensorOps

if TYPE_CHECKING:
    from typing import Callable, Optional

    from .tensor import Tensor
    from .tensor_data import Index, Shape, Storage, Strides

# TIP: Use `NUMBA_DISABLE_JIT=1 pytest tests/ -m task3_1` to run these tests without JIT.

# This code will JIT compile fast versions your tensor_data functions.
# If you get an error, read the docs for NUMBA as to what is allowed
# in these functions.
to_index = njit(inline="always")(to_index)
index_to_position = njit(inline="always")(index_to_position)
broadcast_index = njit(inline="always")(broadcast_index)


class FastOps(TensorOps):
    @staticmethod
    def map(fn: Callable[[float], float]) -> MapProto:
        "See `tensor_ops.py`"

        # This line JIT compiles your tensor_map
        f = tensor_map(njit()(fn))

        def ret(a: Tensor, out: Optional[Tensor] = None) -> Tensor:
            if out is None:
                out = a.zeros(a.shape)
            f(*out.tuple(), *a.tuple())
            return out

        return ret

    @staticmethod
    def zip(fn: Callable[[float, float], float]) -> Callable[[Tensor, Tensor], Tensor]:
        "See `tensor_ops.py`"

        f = tensor_zip(njit()(fn))

        def ret(a: Tensor, b: Tensor) -> Tensor:
            c_shape = shape_broadcast(a.shape, b.shape)
            out = a.zeros(c_shape)
            f(*out.tuple(), *a.tuple(), *b.tuple())
            return out

        return ret

    @staticmethod
    def reduce(
        fn: Callable[[float, float], float], start: float = 0.0
    ) -> Callable[[Tensor, int], Tensor]:
        "See `tensor_ops.py`"
        f = tensor_reduce(njit()(fn))

        def ret(a: Tensor, dim: int) -> Tensor:
            out_shape = list(a.shape)
            out_shape[dim] = 1

            # Other values when not sum.
            out = a.zeros(tuple(out_shape))
            out._tensor._storage[:] = start

            f(*out.tuple(), *a.tuple(), dim)
            return out

        return ret

    @staticmethod
    def matrix_multiply(a: Tensor, b: Tensor) -> Tensor:
        """
        Batched tensor matrix multiply ::

            for n:
              for i:
                for j:
                  for k:
                    out[n, i, j] += a[n, i, k] * b[n, k, j]

        Where n indicates an optional broadcasted batched dimension.

        Should work for tensor shapes of 3 dims ::

            assert a.shape[-1] == b.shape[-2]

        Args:
            a : tensor data a
            b : tensor data b

        Returns:
            New tensor data
        """

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

        tensor_matrix_multiply(*out.tuple(), *a.tuple(), *b.tuple())

        # Undo 3d if we added it.
        if both_2d:
            out = out.view(out.shape[1], out.shape[2])
        return out


# Implementations


def tensor_map(
    fn: Callable[[float], float]
) -> Callable[[Storage, Shape, Strides, Storage, Shape, Strides], None]:
    """
    NUMBA low_level tensor_map function. See `tensor_ops.py` for description.

    Optimizations:

    * Main loop in parallel
    * All indices use numpy buffers
    * When `out` and `in` are stride-aligned, avoid indexing

    Args:
        fn: function mappings floats-to-floats to apply.

    Returns:
        Tensor map function.
    """

    def _map(
        out: Storage,
        out_shape: Shape,
        out_strides: Strides,
        in_storage: Storage,
        in_shape: Shape,
        in_strides: Strides,
    ) -> None:
        if np.array_equal(out_shape, in_shape) and np.array_equal(out_strides, in_strides):
            # Both strides and shapes are equal, we don't need to worry about
            # indexing or broadcasting!
            for i in prange(np.prod(out_shape)):
                # Apply function
                out[i] = fn(in_storage[i])
        else:
            for i in prange(np.prod(out_shape)):
                # Create a numpy array to store out index
                # important that we initialize it as int64! defaults to float
                out_index = np.empty(len(out_shape), dtype=np.int64)
                # Get index from ordinal
                to_index(i, out_shape, out_index)
                # Convert index to position in storage (taking into account strides)
                out_position = index_to_position(out_index, out_strides)

                # Create a numpy array to store in index
                # important that we initialize it as int64! defaults to float
                in_index = np.empty(len(in_shape), dtype=np.int64)
                # Get broadcast in  index from out index
                broadcast_index(out_index, out_shape, in_shape, in_index)
                # Convert index to position in storage (taking into account strides)
                in_position = index_to_position(in_index, in_strides)

                # Apply function
                out[out_position] = fn(in_storage[in_position])

    return njit(parallel=True)(_map)  # type: ignore


def tensor_zip(
    fn: Callable[[float, float], float]
) -> Callable[
    [Storage, Shape, Strides, Storage, Shape, Strides, Storage, Shape, Strides], None
]:
    """
    NUMBA higher-order tensor zip function. See `tensor_ops.py` for description.


    Optimizations:

    * Main loop in parallel
    * All indices use numpy buffers
    * When `out`, `a`, `b` are stride-aligned, avoid indexing

    Args:
        fn: function maps two floats to float to apply.

    Returns:
        Tensor zip function.
    """

    def _zip(
        out: Storage,
        out_shape: Shape,
        out_strides: Strides,
        a_storage: Storage,
        a_shape: Shape,
        a_strides: Strides,
        b_storage: Storage,
        b_shape: Shape,
        b_strides: Strides,
    ) -> None:
        if np.array_equal(out_shape, a_shape) and np.array_equal(out_shape, b_shape) and np.array_equal(out_strides, a_strides) and np.array_equal(out_strides, b_strides):
            # Both strides and shapes are equal, we don't need to worry about
            # indexing or broadcasting!
            for i in prange(np.prod(out_shape)):
                # Apply function
                out[i] = fn(a_storage[i], b_storage[i])
        else:
            # This is pretty much the same as tensor_map except we have
            # two ins: a and b
            for i in prange(np.prod(out_shape)):
                # Create a numpy array to store out index
                # important that we initialize it as int64! defaults to float
                out_index = np.empty(len(out_shape), dtype=np.int64)
                # Get index from ordinal
                to_index(i, out_shape, out_index)
                # Convert index to position in storage (taking into account strides)
                out_position = index_to_position(out_index, out_strides)

                # Create a numpy array to store a index
                # important that we initialize it as int64! defaults to float
                a_index = np.empty(len(a_shape), dtype=np.int64)
                # Get broadcast a  index from out index
                broadcast_index(out_index, out_shape, a_shape, a_index)
                # Convert index to position in storage (taking into account strides)
                a_position = index_to_position(a_index, a_strides)

                # Create a numpy array to store b index
                # important that we initialize it as int64! defaults to float
                b_index = np.empty(len(b_shape), dtype=np.int64)
                # Get broadcast b  index from out index
                broadcast_index(out_index, out_shape, b_shape, b_index)
                # Convert index to position in storage (taking into account strides)
                b_position = index_to_position(b_index, b_strides)

                # Apply function
                out[out_position] = fn(a_storage[a_position], b_storage[b_position])

    return njit(parallel=True)(_zip)  # type: ignore


def tensor_reduce(
    fn: Callable[[float, float], float]
) -> Callable[[Storage, Shape, Strides, Storage, Shape, Strides, int], None]:
    """
    NUMBA higher-order tensor reduce function. See `tensor_ops.py` for description.

    Optimizations:

    * Main loop in parallel
    * All indices use numpy buffers
    * Inner-loop should not call any functions or write non-local variables

    Args:
        fn: reduction function mapping two floats to float.

    Returns:
        Tensor reduce function
    """

    def _reduce(
        out: Storage,
        out_shape: Shape,
        out_strides: Strides,
        a_storage: Storage,
        a_shape: Shape,
        a_strides: Strides,
        reduce_dim: int,
    ) -> None:
        # Modified from code in tensor_ops.py as that one wasn't parallelizing correctly
        for i in prange(np.prod(out_shape)):
            # Create a numpy array to store out index
            # important that we initialize it as int64! defaults to float
            out_index = np.empty(len(out_shape), dtype=np.int64)
            # Get index from ordinal
            to_index(i, out_shape, out_index)
            # Convert index to position in storage (taking into account strides)
            out_position = index_to_position(out_index, out_strides)

            a_index = np.copy(out_index)
            a_position = index_to_position(a_index, a_strides)
            reduce_stride = a_strides[reduce_dim]
            reduced_value = a_storage[a_position]
            # Iterate through the dimension we're reducing
            # Think of a three dimensional cube we're reducing along one of its
            # dimensions
            # We're essentially taking a 2d slice and reducing down, getting a
            # reduced rectangle
            # In this loop, for each index in the resulting rectangle, we iterate
            # through that third, reduced dimension and do the reduction
            for j in range(a_shape[reduce_dim] - 1):
                a_position = a_position + reduce_stride
                reduced_value = fn(a_storage[a_position], reduced_value)

            out[out_position] = reduced_value

    return njit(parallel=True)(_reduce)  # type: ignore


def _tensor_matrix_multiply(
    out: Storage,
    out_shape: Shape,
    out_strides: Strides,
    a_storage: Storage,
    a_shape: Shape,
    a_strides: Strides,
    b_storage: Storage,
    b_shape: Shape,
    b_strides: Strides,
) -> None:
    """
    NUMBA tensor matrix multiply function.

    Should work for any tensor shapes that broadcast as long as

    ```
    assert a_shape[-1] == b_shape[-2]
    ```

    Optimizations:

    * Outer loop in parallel
    * No index buffers or function calls
    * Inner loop should have no global writes, 1 multiply.


    Args:
        out (Storage): storage for `out` tensor
        out_shape (Shape): shape for `out` tensor
        out_strides (Strides): strides for `out` tensor
        a_storage (Storage): storage for `a` tensor
        a_shape (Shape): shape for `a` tensor
        a_strides (Strides): strides for `a` tensor
        b_storage (Storage): storage for `b` tensor
        b_shape (Shape): shape for `b` tensor
        b_strides (Strides): strides for `b` tensor

    Returns:
        None : Fills in `out`
    """
    # Not totally sure what these are for?
    a_batch_stride = a_strides[0] if a_shape[0] > 1 else 0
    b_batch_stride = b_strides[0] if b_shape[0] > 1 else 0

    # I *think* this should be similar to reduce? Wherein we can start with our
    # index in the out matrix, then iterate through what we need to calculate it
    # using strides to reduce calls

    # According to comment that
    # assert a_shape[-1] == b_shape[-2]
    # it seems like we're "fusing" (my own made up terminology) the last dimension
    # of "a" with the second to last dimension of "b"
    # So let's say they have the following shapes:
    # a: A*B*C*D
    # b: A*B*D*E
    # then
    # a*b: A*B*C*E
    # So we want all dimensions except for the last two to be broadcastable
    # and the last dimension of "a" to be equal to the last dimension of "b"
    # I'm visualizing the multiplication in >2 dimensions by visualizing a cube
    # composed of a bunch of stacked matrices, and so we're really just doing a bunch
    # of stacked matrix multiplications

    assert a_shape[-1] == b_shape[-2]
    fuse_dim_size = a_shape[-1]
    a_fuse_stride = a_strides[-1]
    b_fuse_stride = b_strides[-2]

    for i in prange(np.prod(out_shape)):
        # Create a numpy array to store out index
        # important that we initialize it as int64, otherwise defaults to float
        out_index = np.empty(len(out_shape), dtype=np.int64)
        # Get index from ordinal
        to_index(i, out_shape, out_index)
        # Convert index to position in storage (taking into account strides)
        out_position = index_to_position(out_index, out_strides)

        a_index = np.empty(len(a_shape), dtype=np.int64)
        # Broadcast
        broadcast_index(out_index, out_shape, a_shape, a_index)
        a_index[-1] = 0
        # Convert index to position in storage (taking into account strides)
        a_position = index_to_position(a_index, a_strides)

        b_index = np.empty(len(b_shape), dtype=np.int64)
        # Broadcast
        broadcast_index(out_index, out_shape, b_shape, b_index)
        b_index[-2] = 0
        # Convert index to position in storage (taking into account strides)
        b_position = index_to_position(b_index, b_strides)

        val = 0
        for j in range(fuse_dim_size):
            val += a_storage[a_position] * b_storage[b_position]
            # Originally I'd been doing a_storage[a_position + j * a_fuse_stride]
            # but that meant I was doing three multiplications in the inner loop
            # hence this kindof janky (maybe faster?) solution
            a_position += a_fuse_stride
            b_position += b_fuse_stride
        
        out[out_position] = val


tensor_matrix_multiply = njit(parallel=True, fastmath=True)(_tensor_matrix_multiply)

from typing import Tuple

from . import operators
from .autodiff import Context
from .fast_ops import FastOps
from .tensor import Tensor
from .tensor_functions import Function, rand, tensor


def tile(input: Tensor, kernel: Tuple[int, int]) -> Tuple[Tensor, int, int]:
    """
    Reshape an image tensor for 2D pooling

    Args:
        input: batch x channel x height x width
        kernel: height x width of pooling

    Returns:
        Tensor of size batch x channel x new_height x new_width x (kernel_height * kernel_width) as well as the new_height and new_width value.
    """

    batch, channel, height, width = input.shape
    kh, kw = kernel
    assert height % kh == 0
    assert width % kw == 0

    new_width = width // kw # we know it divides cleanly bc of above assert
    new_height = height // kh # same

    input = input.contiguous().view(batch, channel, new_height, kh, new_width, kw)
    # Permute and make memory contiguous (so we can use view on it again)
    input = input.permute(0, 1, 2, 4, 3, 5).contiguous()
    # Resize to our final desired shape
    input = input.view(batch, channel, new_height, new_width, kh * kw).contiguous()

    return input, new_height, new_width

def avgpool2d(input: Tensor, kernel: Tuple[int, int]) -> Tensor:
    """
    Tiled average pooling 2D

    Args:
        input : batch x channel x height x width
        kernel : height x width of pooling

    Returns:
        Pooled tensor
    """
    batch, channel, height, width = input.shape

    tiled, new_height, new_width = tile(input, kernel)

    summed = tiled.sum(4)
    # Remove last dimension (has size 1)
    summed = summed.view(batch, channel, new_height, new_width)
    divided = summed / (kernel[0] * kernel[1])
    
    return divided


max_reduce = FastOps.reduce(operators.max, -1e9)


def argmax(input: Tensor, dim: int) -> Tensor:
    """
    Compute the argmax as a 1-hot tensor.

    Args:
        input : input tensor
        dim : dimension to apply argmax


    Returns:
        :class:`Tensor` : tensor with 1 on highest cell in dim, 0 otherwise

    """
    out = max_reduce(input, dim)
    return out == input


class Max(Function):
    @staticmethod
    def forward(ctx: Context, input: Tensor, dim: Tensor) -> Tensor:
        "Forward of max should be max reduction"
        dim_int = int(dim.item())
        ctx.save_for_backward(dim_int)
        return max_reduce(input, dim_int)

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tuple[Tensor, float]:
        "Backward of max should be argmax (see above)"
        dim, = ctx.saved_values
        return argmax(grad_output, dim), 0.0


def max(input: Tensor, dim: int) -> Tensor:
    return Max.apply(input, input._ensure_tensor(dim))


def softmax(input: Tensor, dim: int) -> Tensor:
    r"""
    Compute the softmax as a tensor.



    $z_i = \frac{e^{x_i}}{\sum_i e^{x_i}}$

    Args:
        input : input tensor
        dim : dimension to apply softmax

    Returns:
        softmax tensor
    """
    
    # Exponentiate
    input_exp = input.exp()
    # Sum e^x over our target dimension
    summed = input_exp.sum(dim)
    # Broadcasting will take care of everything else!
    return input_exp / summed


def logsoftmax(input: Tensor, dim: int) -> Tensor:
    r"""
    Compute the log of the softmax as a tensor.

    $z_i = x_i - \log \sum_i e^{x_i}$

    See https://en.wikipedia.org/wiki/LogSumExp#log-sum-exp_trick_for_log-domain_calculations

    Args:
        input : input tensor
        dim : dimension to apply log-softmax

    Returns:
         log of softmax tensor
    """
    # max_t = Max.apply(input, input._ensure_tensor(dim))

    # This wasn't working for some reason?
    # # Intermediate calculation of the sum of logs
    # inter_t = input - max_t
    # inter_t = inter_t.exp()
    # inter_t = inter_t.sum(dim)
    # inter_t = inter_t.log()

    # return input - inter_t - max_t

    # Intermediate calculation of the sum of logs
    inter_t = input.exp()
    inter_t = inter_t.sum(dim)
    inter_t = inter_t.log()

    return input - inter_t


def maxpool2d(input: Tensor, kernel: Tuple[int, int]) -> Tensor:
    """
    Tiled max pooling 2D

    Args:
        input: batch x channel x height x width
        kernel: height x width of pooling

    Returns:
        Tensor : pooled tensor
    """
    batch, channel, height, width = input.shape
    
    tiled, new_height, new_width = tile(input, kernel)
    maxed = max(tiled, 4)
    # Remove last dimension (has size 1)
    maxed = maxed.view(batch, channel, new_height, new_width)

    return maxed


def dropout(input: Tensor, rate: float, ignore: bool = False) -> Tensor:
    """
    Dropout positions based on random noise.

    Args:
        input : input tensor
        rate : probability [0, 1) of dropping out each position
        ignore : skip dropout, i.e. do nothing at all

    Returns:
        tensor with random positions dropped out
    """
    if ignore:
        return input
    
    random_t = rand(input.shape, backend=input.backend)
    keep_t = random_t > rate
    return input * keep_t


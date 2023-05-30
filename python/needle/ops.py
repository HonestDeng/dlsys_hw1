"""Operator implementations."""

from numbers import Number
from typing import Optional, List
from .autograd import NDArray
from .autograd import Op, Tensor, Value, TensorOp
from .autograd import TensorTuple, TensorTupleOp

# NOTE: we will import numpy as the array_api
# as the backend for our computations, this line will change in later homeworks
import numpy as array_api


class EWiseAdd(TensorOp):
    def compute(self, a: NDArray, b: NDArray):
        return a + b

    def gradient(self, out_grad: Tensor, node: Tensor):
        return out_grad, out_grad


def add(a, b):
    return EWiseAdd()(a, b)


class AddScalar(TensorOp):
    def __init__(self, scalar):
        self.scalar = scalar

    def compute(self, a: NDArray):
        return a + self.scalar

    def gradient(self, out_grad: Tensor, node: Tensor):
        return out_grad


def add_scalar(a, scalar):
    return AddScalar(scalar)(a)


class EWiseMul(TensorOp):
    def compute(self, a: NDArray, b: NDArray):
        return a * b

    def gradient(self, out_grad: Tensor, node: Tensor):
        lhs, rhs = node.inputs
        return out_grad * rhs, out_grad * lhs


def multiply(a, b):
    return EWiseMul()(a, b)


class MulScalar(TensorOp):
    def __init__(self, scalar):
        self.scalar = scalar

    def compute(self, a: NDArray):
        return a * self.scalar

    def gradient(self, out_grad: Tensor, node: Tensor):
        return (out_grad * self.scalar,)


def mul_scalar(a, scalar):
    return MulScalar(scalar)(a)


class PowerScalar(TensorOp):
    """Op raise a tensor to an (integer) power."""

    def __init__(self, scalar: int):
        self.scalar = scalar

    def compute(self, a: NDArray) -> NDArray:
        ### BEGIN YOUR SOLUTION
        # raise NotImplementedError()
        return array_api.power(a, self.scalar)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        raise NotImplementedError()
        ### END YOUR SOLUTION


def power_scalar(a, scalar):
    return PowerScalar(scalar)(a)


class EWiseDiv(TensorOp):
    """Op to element-wise divide two nodes."""

    def compute(self, a, b):
        ### BEGIN YOUR SOLUTION
        # raise NotImplementedError()
        return a / b
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        # raise NotImplementedError()i
        lhs, rhs = node.inputs
        left_grad = divide(Tensor([1]).broadcast_to(rhs.shape), rhs)
        right_grad = negate(lhs) * power_scalar(rhs, -2)
        return left_grad * out_grad, right_grad * out_grad
        ### END YOUR SOLUTION


def divide(a, b):
    return EWiseDiv()(a, b)


class DivScalar(TensorOp):
    def __init__(self, scalar):
        self.scalar = scalar

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        # raise NotImplementedError()
        return a / self.scalar
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        # raise NotImplementedError()
        lhs = node.inputs[0]
        return Tensor(1 / self.scalar).broadcast_to(lhs.shape) * out_grad
        ### END YOUR SOLUTION


def divide_scalar(a, scalar):
    return DivScalar(scalar)(a)


class Transpose(TensorOp):
    def __init__(self, axes: Optional[tuple] = None):
        self.axes = axes
        if self.axes == None:
            self.axes = [-1, -2]

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        # raise NotImplementedError()
        change_to = [i for i in range(len(a.shape))]
        change_to[self.axes[0]], change_to[self.axes[1]] = change_to[self.axes[1]], change_to[self.axes[0]]
        return array_api.transpose(a, change_to)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return transpose(out_grad, self.axes)
        ### END YOUR SOLUTION


def transpose(a, axes=None):
    return Transpose(axes)(a)


class Reshape(TensorOp):
    def __init__(self, shape):
        self.shape = shape

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return array_api.reshape(a, self.shape)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return reshape(out_grad, shape=node.inputs[0].shape)
        ### END YOUR SOLUTION


def reshape(a, shape):
    return Reshape(shape)(a)


class BroadcastTo(TensorOp):
    def __init__(self, shape):
        self.shape = shape  # 想要广播到目的形状

    def compute(self, a):
        return array_api.broadcast_to(a, self.shape)

    def gradient(self, out_grad, node):
        # out_grad是经过维度拓展之后的张量的梯度，也就是b_exp的梯度b_exp_grad
        # node是张量b_exp，node.inputs[0]就是b。
        lhs = node.inputs[0]  # 算子的输入，也就是b=[3]
        origin_shape = lhs.shape  # 原本的形状，也就是b.shape=(1,)
        target_shape = self.shape  # 想要变换到的形状，也就是b_exp.shape=(1,2)
        expanded_axes = []  # 记录哪一个维度被拓展了
        for i in range(-1, -len(target_shape)-1, -1):  # 从尾部开始遍历
            if i < -len(origin_shape):
                # origin_shape的长度可能会比target_shape短，
                # 比如origin_shape=(1,)，target_shape=(1,2)。
                expanded_axes.append(i+len(target_shape))
                continue
            if target_shape[i] != origin_shape[i]:
                # 如果目标形状与原本的形状不相同
                # 那就说明这个维度经过了拓展，需要记录到expanded_axes中
                expanded_axes.append(i + len(target_shape))
        # out_grad进行sum运算，运算的轴axes是b_exp相对于b经过拓展的维度
        res = summation(out_grad, tuple(expanded_axes))
        # 因为res的形状可能与lhs(也就是b)不相同，所以这里需要reshape到b原本的维度上。
        res = reshape(res, origin_shape)
        return res


def broadcast_to(a, shape):
    return BroadcastTo(shape)(a)


class Summation(TensorOp):
    def __init__(self, axes: Optional[tuple] = None):
        self.axes = axes

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        # raise NotImplementedError()
        return array_api.sum(a, axis=self.axes)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        to_shape = list(node.inputs[0].shape)
        if self.axes == None:
            to_shape = [1] * len(to_shape)
        else:
            for i in self.axes:
                to_shape[i] = 1
        return broadcast_to(out_grad.reshape(to_shape), node.inputs[0].shape)
        ### END YOUR SOLUTION


def summation(a, axes=None):
    return Summation(axes)(a)


class MatMul(TensorOp):
    def compute(self, a, b):
        ### BEGIN YOUR SOLUTION
        # raise NotImplementedError()
        return array_api.matmul(a, b)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        # raise NotImplementedError()
        lhs, rhs = node.inputs[0], node.inputs[1]
        left_grad = matmul(out_grad, transpose(rhs))
        right_grad = matmul(transpose(lhs), out_grad)
        for i in range(len(left_grad.shape) - len(lhs.shape)):
            left_grad = summation(left_grad, axes=0)
        for i in range(len(right_grad.shape) - len(rhs.shape)):
            right_grad = summation(right_grad, axes=0)

        return left_grad, right_grad
        ### END YOUR SOLUTION


def matmul(a, b):
    return MatMul()(a, b)


class Negate(TensorOp):
    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        # raise NotImplementedError()
        return -a
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return negate(out_grad)
        ### END YOUR SOLUTION


def negate(a):
    return Negate()(a)


class Log(TensorOp):
    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return array_api.log(a)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        return multiply(out_grad, divide(broadcast_to(Tensor(1), node.inputs[0].shape), node.inputs[0]))


def log(a):
    return Log()(a)


class Exp(TensorOp):
    def compute(self, a):
        return array_api.exp(a)

    def gradient(self, out_grad, node):
        return multiply(out_grad, exp(node.inputs[0]))


def exp(a):
    return Exp()(a)


# TODO
class ReLU(TensorOp):
    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        raise NotImplementedError()
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        raise NotImplementedError()
        ### END YOUR SOLUTION


def relu(a):
    return ReLU()(a)

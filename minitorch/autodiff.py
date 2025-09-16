import queue

from dataclasses import dataclass
from typing import Any, Iterable, List, Tuple

from typing_extensions import Protocol

# ## Task 1.1
# Central Difference calculation


def central_difference(f: Any, *vals: Any, arg: int = 0, epsilon: float = 1e-6) -> Any:
    r"""
    Computes an approximation to the derivative of `f` with respect to one arg.

    See :doc:`derivative` or https://en.wikipedia.org/wiki/Finite_difference for more details.

    Args:
        f : arbitrary function from n-scalar args to one value
        *vals : n-float values $x_0 \ldots x_{n-1}$
        arg : the number $i$ of the arg to compute the derivative
        epsilon : a small constant

    Returns:
        An approximation of $f'_i(x_0, \ldots, x_{n-1})$
    """
    vals_under = list(vals)
    vals_under[arg] -= epsilon

    vals_over = list(vals)
    vals_over[arg] += epsilon
    
    return (f(*vals_over) - f(*vals_under)) / (2.0 * epsilon)


variable_count = 1


class Variable(Protocol):
    def accumulate_derivative(self, x: Any) -> None:
        pass

    @property
    def unique_id(self) -> int:
        pass

    def is_leaf(self) -> bool:
        pass

    def is_constant(self) -> bool:
        pass

    @property
    def parents(self) -> Iterable["Variable"]:
        pass

    def chain_rule(self, d_output: Any) -> Iterable[Tuple["Variable", Any]]:
        pass


def topological_sort(variable: Variable) -> Iterable[Variable]:
    """
    Computes the topological order of the computation graph.

    Args:
        variable: The right-most variable

    Returns:
        Non-constant Variables in topological order starting from the right.
    """

    in_degree = {variable.unique_id: 0}
    variable_queue = queue.Queue()
    variable_queue.put(variable)
    visited = set()

    while not variable_queue.empty():
        v = variable_queue.get()
        if v.unique_id in visited or v.is_constant():
            # If v is a constant, it has no history/parents!
            continue
        visited.add(v.unique_id)

        for p in v.parents:
            if p.unique_id in in_degree:
                in_degree[p.unique_id] += 1
            else:
                in_degree[p.unique_id] = 1

            variable_queue.put(p)
    
    sorted_variables = []
    variable_queue = queue.Queue()
    variable_queue.put(variable)

    while not variable_queue.empty():
        v = variable_queue.get()
        if v.is_constant():
            # If v is a constant, it has no history/parents!
            continue

        sorted_variables.append(v)
        for p in v.parents:
            if p.unique_id not in in_degree:
                raise RuntimeError("p should be in in_degree!")
            in_degree[p.unique_id] -= 1
            if in_degree[p.unique_id] == 0:
                variable_queue.put(p)
            elif in_degree[p.unique_id] < 0:
                raise RuntimeError("in_degree should never be less than zero!")
            

    return sorted_variables


def backpropagate(variable: Variable, deriv: Any) -> None:
    """
    Runs backpropagation on the computation graph in order to
    compute derivatives for the leave nodes.

    Args:
        variable: The right-most variable
        deriv  : Its derivative that we want to propagate backward to the leaves.

    No return. Should write to its results to the derivative values of each leaf through `accumulate_derivative`.
    """
    ordered = topological_sort(variable)
    derivatives = {variable.unique_id: deriv}

    for v in ordered:
        d = derivatives[v.unique_id]
        if v.is_leaf():
            v.accumulate_derivative(d)
        else:
            out_vs_ds = v.chain_rule(d)
            for target_v, out_d in out_vs_ds:
                if target_v.unique_id in derivatives:
                    derivatives[target_v.unique_id] += out_d
                else:
                    derivatives[target_v.unique_id] = out_d



@dataclass
class Context:
    """
    Context class is used by `Function` to store information during the forward pass.
    """

    no_grad: bool = False
    saved_values: Tuple[Any, ...] = ()

    def save_for_backward(self, *values: Any) -> None:
        "Store the given `values` if they need to be used during backpropagation."
        if self.no_grad:
            return
        self.saved_values = values

    @property
    def saved_tensors(self) -> Tuple[Any, ...]:
        return self.saved_values

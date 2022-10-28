from abc import ABC, abstractmethod

import firedrake as fd
import numpy as np
import pyadjoint
import torch

from .util import to_numpy, to_firedrake

class SolveFunction(torch.autograd.Function):
    """Executes the solve function of a FiredrakeModule"""

    @staticmethod
    def forward(ctx, solver, *args):
        """Computes the output of the FEA model and saves a corresponding gradient tape

        Input:
            fenics_solver (FEniCSSolver): FEniCSSolver to be executed during the forward pass
            args (tuple): tensor representation of the input to fenics_solver.forward

        Output:
            tensor representation of the output from fenics_solver.solve
        """
        # Check that the number of inputs arguments is correct
        n_args = len(args)
        expected_nargs = len(solver.firedrake_input_templates())
        if n_args != expected_nargs:
            raise ValueError(f'Wrong number of arguments to {solver}.'
                             f' Expected {expected_nargs} got {n_args}.')

        # Check that each input argument has correct dimensions
        for i, (arg, template) in enumerate(zip(args, solver.numpy_input_templates())):
            if arg.shape != template.shape:
                raise ValueError(f'Expected input shape {template.shape} for input'
                                 f' {i} but got {arg.shape}.')

        # Check that the inputs are of double precision
        for i, arg in enumerate(args):
            if (isinstance(arg, np.ndarray) and arg.dtype != np.float64) or \
               (torch.is_tensor(arg) and arg.dtype != torch.float64):
                raise TypeError(f'All inputs must be type {torch.float64},'
                                f' but got {arg.dtype} for input {i}.')

        # Convert input tensors to corresponding FEniCS variables
        firedrake_inputs = []
        for inp, template in zip(args, solver.firedrake_input_templates()):
            if torch.is_tensor(inp):
                inp = inp.detach().numpy()
            firedrake_inputs.append(to_firedrake(inp, template))

        # Create tape associated with this forward pass
        tape = pyadjoint.Tape()
        pyadjoint.set_working_tape(tape)

        # Execute forward pass
        firedrake_outputs = solver.solve(*firedrake_inputs)

        # If single output
        if not isinstance(firedrake_outputs, tuple):
            firedrake_outputs = (firedrake_outputs,)

        # Save variables to be used for backward pass
        ctx.tape = tape
        ctx.firedrake_inputs = firedrake_inputs
        ctx.firedrake_outputs = firedrake_outputs

        # Return tensor representation of outputs
        return tuple(torch.from_numpy(to_numpy(firedrake_output)) for firedrake_output in firedrake_outputs)

    @staticmethod
    def backward(ctx, *grad_outputs):
        """Computes the gradients of the output with respect to the input

        Input:
            ctx: Context used for storing information from the forward pass
            grad_output: gradient of the output from successive operations
        """
        # Convert gradient of output to a Firedrake variable
        adj_values = []
        for grad_output, firedrake_output in zip(grad_outputs, ctx.firedrake_outputs):
            adj_values.append(to_firedrake(grad_output.numpy(), firedrake_output))

        # Check which gradients need to be computed
        controls = list(map(pyadjoint.Control,
                            (c for g, c in zip(ctx.needs_input_grad[1:], ctx.firedrake_inputs) if g)))

        # Compute and accumulate gradient for each output with respect to each input
        accumulated_grads = [None] * len(controls)
        for firedrake_output, adj_value in zip(ctx.firedrake_outputs, adj_values):
            firedrake_grads = pyadjoint.compute_gradient(firedrake_output, controls,
                                                           tape=ctx.tape, adj_value=adj_value)

            # Convert gradients to tensor representation
            numpy_grads = [g if g is None else torch.from_numpy(to_numpy(g)) for g in firedrake_grads]
            for i, (acc_g, g) in enumerate(zip(accumulated_grads, numpy_grads)):
                if g is None:
                    continue
                if acc_g is None:
                    accumulated_grads[i] = g
                else:
                    accumulated_grads[i] += g

        # Insert None for not computed gradients
        acc_grad_iter = iter(accumulated_grads)
        return tuple(None if not g else next(acc_grad_iter) for g in ctx.needs_input_grad)


class FiredrakeModule(ABC, torch.nn.Module):
    """Solves a PDE problem with Firedrake"""

    def __init__(self):
        super().__init__()
        self._firedrake_input_templates = None
        self._numpy_input_templates = None

    @abstractmethod
    def input_templates(self):
        """Returns templates of the input to FiredrakeModule.solve

        Not intended to be called by the user. Instead uses FiredrakeModule.firedrake_input_templates

        Output:
            Firedrake variable or tuple of Firedrake variables
        """
        pass

    @abstractmethod
    def solve(self, *args):
        """Solve PDE defined in Firedrake

        Input:
            args (tuple): Firedrake variables of same type as specified by FiredrakeModule.input_templates

        Output:
            outputs (tuple): results from solving the PDE
        """
        pass

    def firedrake_input_templates(self):
        """Returns tuple of Firedrake variables corresponding to input templates to FiredrakeModule.solve"""
        if self._firedrake_input_templates is None:
            templates = self.input_templates()
            if not isinstance(templates, tuple):
                templates = (templates,)
            self._firedrake_input_templates = templates
        return self._firedrake_input_templates

    def numpy_input_templates(self):
        """Returns tuple of numpy representations of the input templates to FiredrakeModule.solve"""
        if self._numpy_input_templates is None:
            self._numpy_input_templates = [to_numpy(temp) for temp in self.firedrake_input_templates()]
        return self._numpy_input_templates

    def forward(self, *args):
        """ Executes solve through SolveFunction for multiple inputs

        Input:
            args (tuple): List of tensor representations of the input to the FiredrakeModel.
                          Each element in the tuple should be on the format
                          N x M_1 x M_2 x ... where N is the batch size and
                          M_1 x M_2 ... are the dimensions of the input argument
        Output:
            output: Tensor representations of the output from the FiredrakeModel on the format
                    N x P_1 x P_2 x ... where N is the batch size and P_1 x P_2 x ...
                    are the dimensions of the output
        """
        # Check that the number of inputs is the same for each input argument
        if len(args) != 0:
            n = args[0].shape[0]
            for arg in args[1:]:
                if arg.shape[0] != n:
                    raise ValueError('Number of inputs must be the same for each input argument.')

        # Run the model on each set of inputs
        outs = [SolveFunction.apply(self, *inp) for inp in zip(*args)]

        # Rearrange by output index and stack over number of input sets
        outs = tuple(torch.stack(out) for out in zip(*outs))

        if len(outs) == 1:
            return outs[0]
        else:
            return outs

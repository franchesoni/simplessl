print("Importing standard...")
import math
from typing import Tuple, Union, Optional, Iterable, Dict, Any
from typing_extensions import TypeAlias
from typing import Tuple
import collections

print("Importing external...")
import torch
import torch.nn.functional as F
import torch.optim
try:
    from torch.optim.optimizer import ParamsT
except ImportError:
    ParamsT : TypeAlias = Union[Iterable[torch.Tensor], Iterable[Dict[str, Any]]]




def centralize_gradient(x):
    """credit - https://github.com/Yonghongwei/Gradient-Centralization"""

    size = x.dim()

    if size > 1:
        x.data.add_(-x.mean(dim=tuple(range(1, size)), keepdim=True))


def normalize_gradient(x, eps=1e-8):
    x.data.div_(x.norm() + eps)


class SingFree(torch.optim.Optimizer):
    r"""
    Schedule-Free AdamW (but with SING!)
    As the name suggests, no scheduler is needed with this optimizer. 
    To add warmup, rather than using a learning rate schedule you can just
    set the warmup_steps parameter.
    
    This optimizer requires that .train() and .eval() be called before the
    beginning of training and evaluation respectively. The optimizer should
    also be placed in eval mode when saving checkpoints.
    
    Arguments:
        params (iterable): 
            Iterable of parameters to optimize or dicts defining 
            parameter groups.
        lr (float): 
            Learning rate parameter (default 0.0025)
        betas (Tuple[float, float], optional): coefficients used for computing
            running averages of gradient and its square (default: (0.9, 0.999)).
        eps (float): 
            Term added to the denominator outside of the root operation to 
            improve numerical stability. (default: 1e-8).
        weight_decay (float): 
            Weight decay, i.e. a L2 penalty (default: 0).
        warmup_steps (int): Enables a linear learning rate warmup (default 0).
        r (float): Use polynomial weighting in the average 
            with power r (default 0).
        weight_lr_power (float): During warmup, the weights in the average will
            be equal to lr raised to this power. Set to 0 for no weighting
            (default 2.0).
        foreach (bool): Use a foreach-backed implementation of the optimizer.
            Should be significantly faster, but will have higher peak memory
            usage (default True if supported in your PyTorch version).
    """
    def __init__(self,
                 params: ParamsT,
                 lr: Union[float, torch.Tensor] = 0.0025,
                 betas: Tuple[float, float] = (0.9, 0.999),
                 eps: float = 1e-8,
                 weight_decay: float = 0,
                 warmup_steps: int = 0,
                 r: float = 0.0,
                 weight_lr_power: float = 2.0,
                 foreach: Optional[bool] = hasattr(torch, "_foreach_mul_"),
                 # sing params
                 softplus: bool = True,
                 beta_softplus: int = 50,
                 grad_central: bool = True,
                 grad_norm: bool = True,
                 lookahead_active: bool = True,
                 la_mergetime: int = 5,
                 la_alpha: float = 0.5,
                 ):

        defaults = dict(lr=lr, 
                        betas=betas, 
                        eps=eps,
                        r=r,
                        k=0,
                        warmup_steps=warmup_steps,
                        train_mode=True,
                        weight_sum=0.0,
                        lr_max=-1.0,
                        weight_lr_power=weight_lr_power,
                        weight_decay=weight_decay,
                        foreach=foreach,
                        # sing params
                        grad_central=grad_central,
                        grad_norm=grad_norm,
                        softplus=softplus,
                        beta_softplus=beta_softplus,
                        lookahead_active=lookahead_active,
                        la_mergetime=la_mergetime,
                        la_alpha=la_alpha,
                        la_step=0,
                        )
        super().__init__(params, defaults)

    def __setstate__(self, state):
        super().__setstate__(state)
        for group in self.param_groups:
            group.setdefault('la_step', 0)
    
    def eval(self):
        for group in self.param_groups:
            train_mode = group['train_mode']
            beta1, _ = group['betas']
            if train_mode:
                for p in group['params']:
                    state = self.state[p]
                    if 'z' in state:
                        # Set p.data to x
                        p.data.lerp_(end=state['z'], weight=1-1/beta1)
                group['train_mode'] = False

    def train(self):
        for group in self.param_groups:
            train_mode = group['train_mode']
            beta1, _ = group['betas']
            if not train_mode:
                for p in group['params']:
                    state = self.state[p]
                    if 'z' in state:
                        # Set p.data to y
                        p.data.lerp_(end=state['z'], weight=1-beta1)
                group['train_mode'] = True

    def step(self, closure=None):
        """Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """

        loss = None
        if closure is not None:
            loss = closure()
        
        for group in self.param_groups:
            eps = group['eps']
            beta1, beta2 = group['betas']
            decay = group['weight_decay']
            k = group['k']
            r = group['r']
            warmup_steps = group['warmup_steps']
            weight_lr_power = group['weight_lr_power']
            
            if k < warmup_steps:
              sched = (k+1) / warmup_steps
            else:
              sched = 1.0
            
            bias_correction2 = 1 - beta2 ** (k+1)
            lr = group['lr']*sched*math.sqrt(bias_correction2)
            
            lr_max = group['lr_max'] = max(lr, group['lr_max'])
            
            weight = ((k+1)**r) * (lr_max**weight_lr_power)
            weight_sum = group['weight_sum'] = group['weight_sum'] + weight

            try:
                ckp1 = weight/weight_sum
            except ZeroDivisionError:
                ckp1 = 0

            if not group['train_mode']:
                raise Exception("Not in train mode!")

            active_p = [p for p in group['params'] if p.grad is not None]
            
            for p in active_p:
                if 'z' not in self.state[p]:
                    self.state[p]['z'] = torch.clone(p.data)
                    self.state[p]['exp_avg_sq'] = torch.zeros_like(p.data)
                    if group["lookahead_active"]:
                        self.state[p]["lookahead_params"] = torch.zeros_like(p)
                        self.state[p]["lookahead_params"].copy_(p)

                    if group["grad_central"]:
                        centralize_gradient(p.grad)
                    
                    if group["grad_norm"]:
                        normalize_gradient(p.grad, eps)

                    y = p.data # Notation to match theory
                    grad = p.grad.data

                    state = self.state[p]

                    z = state['z']
                    exp_avg_sq = state['exp_avg_sq']

                    exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1-beta2)
                    denom = exp_avg_sq.sqrt().add_(eps)

                    # Reuse grad buffer for memory efficiency
                    grad_normalized = grad.div_(denom)

                    # Weight decay calculated at y
                    if decay != 0:
                        grad_normalized.add_(y, alpha=decay)

                    # These operations update y in-place,
                    # without computing x explicitly.
                    y.lerp_(end=z, weight=ckp1)
                    y.add_(grad_normalized, alpha=lr*(beta1*(1-ckp1)-1))

                    # z step
                    z.sub_(grad_normalized, alpha=lr)

            group['k'] = k+1
        return loss




class SING(torch.optim.Optimizer):
    def __init__(
        self,
        params,
        lr: float = 5e-3,
        betas: Tuple[float, float] = (0.9, 0.999),
        weight_decay: float = 0,
        eps: float = 1e-8,
        softplus: bool = True,
        beta_softplus: int = 50,
        grad_central: bool = True,
        grad_norm: bool = True,
        lookahead_active: bool = True,
        la_mergetime: int = 5,
        la_alpha: float = 0.5,
    ):
        defaults = dict(
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
            grad_central=grad_central,
            grad_norm=grad_norm,
            softplus=softplus,
            beta_softplus=beta_softplus,
            lookahead_active=lookahead_active,
            la_mergetime=la_mergetime,
            la_alpha=la_alpha,
            la_step=0,
        )
        super().__init__(params, defaults)

    def __setstate__(self, state):
        super().__setstate__(state)
        for group in self.param_groups:
            group.setdefault("la_step", 0)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None

        if closure is not None and isinstance(closure, collections.Callable):
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            weight_decay = group["weight_decay"]
            eps = group["eps"]
            lr = group["lr"]
            beta1, beta2 = group["betas"]

            for p in group["params"]:
                if p.grad is None:
                    continue

                if p.grad.is_sparse:
                    raise NotImplementedError()

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state["step"] = torch.tensor(0.0)

                    state["exp_avg"] = torch.zeros_like(
                        p, memory_format=torch.preserve_format
                    )
                    state["exp_avg_sq"] = torch.zeros_like(
                        p, memory_format=torch.preserve_format
                    )

                    if group["lookahead_active"]:
                        state["lookahead_params"] = torch.zeros_like(p)
                        state["lookahead_params"].copy_(p)

                # Gradient centralization
                if group["grad_central"]:
                    centralize_gradient(p.grad)

                # Gradient normalization
                if group["grad_norm"]:
                    normalize_gradient(p.grad, eps)

                state["step"] += 1

                exp_avg = state["exp_avg"]
                exp_avg_sq = state["exp_avg_sq"]

                # Adam update
                exp_avg.mul_(beta1).add_(p.grad, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(p.grad, p.grad, value=1 - beta2)

                bias_correction1 = 1 - torch.pow(beta1, state["step"])
                bias_correction2 = 1 - torch.pow(beta2, state["step"])

                # Weight decay (decoupled like AdamW)
                # Only apply weight decay to weights: https://arxiv.org/pdf/1812.01187.pdf
                if weight_decay and p.dim() > 1:
                    p.data.mul_(1 - lr * weight_decay)

                # Computing the denominator (Adam)
                denom = exp_avg_sq.sqrt() / bias_correction2.sqrt()

                # SAdam - https://arxiv.org/abs/1908.00700
                if group["softplus"]:
                    denom = F.softplus(denom, beta=group["beta_softplus"])
                else:
                    denom.add_(eps)

                # Update the parameter
                p.addcdiv_(exp_avg, denom, value=-lr / bias_correction1)

        # LookAhead - https://arxiv.org/abs/1907.08610
        for group in self.param_groups:
            if not group["lookahead_active"]:
                continue

            group["la_step"] += 1
            la_alpha = group["la_alpha"]

            if group["la_step"] >= group["la_mergetime"]:
                group["la_step"] = 0

                for p in group["params"]:
                    if p.grad is None:
                        continue

                    state = self.state[p]

                    p.data.mul_(la_alpha).add_(
                        state["lookahead_params"], alpha=1 - la_alpha
                    )
                    state["lookahead_params"].copy_(p)

        return loss

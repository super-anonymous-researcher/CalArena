from typing import Optional, Dict, Any, List, Tuple, Callable

import sklearn
import torch
import torch.nn.functional as F

from calibration_package.distributions import Distribution, CategoricalDistribution, ContinuousDistribution
from calibration_package.metrics import Metric


class Objective:
    def __call__(self, y_true: Distribution, y_pred: torch.Tensor):
        raise NotImplementedError()

    def compute_with_grad_and_hess(self, y_true: Distribution, y_pred: torch.Tensor) -> Tuple[
        torch.Tensor, torch.Tensor, torch.Tensor]:
        # default pytorch implementation that can be overridden for fake gradients/hessians
        # todo: should the Hessian be a matrix or do we just want the diagonal?
        raise NotImplementedError()  # todo: implement


class MetricBasedObjective(Objective):
    def __init__(self, metric: Metric, create_pred_distribution: Callable[[torch.Tensor], Distribution]):
        # could also call it link_function instead of create_pred_distribution
        self.metric = metric
        self.create_pred_distribution = create_pred_distribution

    def __call__(self, y_true: Distribution, y_pred: torch.Tensor):
        return self.metric.compute(y_true=y_true, y_pred=self.create_pred_distribution(y_pred))
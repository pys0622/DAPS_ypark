from abc import ABC, abstractmethod
from piq import psnr, ssim, LPIPS
import prettytable
import torch
import torch.nn as nn
import wandb
import numpy as np
import warnings


class Evaluator:
    """
        Evaluation module for computing evaluation metrics.
    """

    def __init__(self, eval_fn_list, device = "cuda", eval_batch_size = 8):
        """
            Initializes the evaluator with the ground truth and measurement.

            Parameters:
                eval_fn_list (tuple): List of evaluation functions to use.
        """
        super().__init__()
        self.eval_fn = {
            eval_fn.name: eval_fn
            for eval_fn in eval_fn_list
        }
        self.main_eval_fn_name = eval_fn_list[0].name
        self.device = torch.device(device)
        self.eval_batch_size = int(eval_batch_size)

    def get_main_eval_fn(self):
        """
            return the first eval_fn by default
        """
        return self.eval_fn[self.main_eval_fn_name]

    def __call__(self, gt, measurement, x, reduction='mean'):
        """
            Computes evaluation metrics for the given input.

            Parameters:
                x (torch.Tensor): Input tensor.
                reduction (str): Reduction method ('mean' or 'none').

            Returns:
                dict: Dictionary of evaluation results.
        """
        results = {}
        for eval_fn_name, eval_fn in self.eval_fn.items():
            results[eval_fn_name] = eval_fn(gt, measurement, x, reduction)
        return results
    @staticmethod
    def to_list( x):
        return x.cpu().detach().tolist()

    @staticmethod
    def _normalize_metric_output(value, batch_size):
        """
        Convert metric output to one scalar per input sample.
        Supported examples:
            [B]
            [B, 1]
            [B, 1, 1, 1]
            [B, C, H, W]
        Non-batch dimensions are averaged.
        """

        if not torch.is_tensor(value):
            value = torch.as_tensor(value)
        if value.ndim == 0:
            if batch_size != 1:
                raise ValueError(
                    "Metric returned a scalar for a batch larger than one "
                    "while reduction='none' was requested."
                )
            return value.reshape(1)
        if value.shape[0] != batch_size:
            if value.numel() == batch_size:
                return value.reshape(batch_size)
            raise ValueError(
                "Metric output does not preserve the batch dimension. "
                f"Expected first dimension {batch_size}, "
                f"but received shape {tuple(value.shape)}."
            )
        return value.reshape(batch_size, -1).mean(dim=1)
    
    def report(self, gt, measurement, x, eval_batch_size = None, device = None):
        '''x: [N, B, C, H, W] or [B, C, H, W]'''
        if len(x.shape) == 4:
            x = x.unsqueeze(0)
        if x.ndim != 5:
            raise ValueError(
                "x must have shape [B, C, H, W] or "
                "[N, B, C, H, W]. "
                f"Received {tuple(x.shape)}."
            )
        num_runs, num_samples = x.shape[:2]
        if gt.shape[0] != num_samples:
            raise ValueError(
                "Ground-truth sample count does not match reconstruction "
                f"sample count: gt={gt.shape[0]}, x={num_samples}."
            )

        if measurement.shape[0] != num_samples:
            raise ValueError(
                "Measurement sample count does not match reconstruction "
                f"sample count: measurement={measurement.shape[0]}, "
                f"x={num_samples}."
            )    
        batch_size = (
            self.eval_batch_size
            if eval_batch_size is None
            else int(eval_batch_size)

        )

        if batch_size <= 0:
            raise ValueError("eval_batch_size must be greater than zero.")

        eval_device = (
            self.device
            if device is None
            else torch.device(device)
        )
        result_dicts = {}

        # eval function
        with torch.inference_mode():
            for metric_name, metric_fn in self.eval_fn.items():
                values_per_run = []
                for run_idx in range(num_runs):
                    values_per_batch = []
                    for start in range(0, num_samples, batch_size):
                        end = min(start + batch_size, num_samples)
                        current_gt = gt[start:end].to(
                            eval_device,
                            non_blocking=True,
                        )

                        current_measurement = measurement[start:end].to(
                            eval_device,
                            non_blocking=True,
                        )

                        current_sample = x[run_idx, start:end].to(
                            eval_device,
                            non_blocking=True,
                        )

                        value = metric_fn(
                            current_gt,
                            current_measurement,
                            current_sample,
                            reduction="none",
                        )

                        value = self._normalize_metric_output(
                            value,
                            batch_size=end - start,
                        )

                        values_per_batch.append(
                            value.detach().cpu()
                        )

                        del current_gt
                        del current_measurement
                        del current_sample
                        del value

                    # [B]

                    run_value = torch.cat(
                        values_per_batch,
                        dim=0,
                    )

                    values_per_run.append(run_value)
                # [N, B]

                value = torch.stack(
                    values_per_run,
                    dim=0,
                )

                mean_value = value.mean(dim=0)
                if num_runs > 1:
                    std_value = value.std(
                        dim=0,
                        unbiased=True,
                    )
                else:
                    std_value = torch.zeros_like(mean_value)
                result_dicts[metric_name] = {
                    # [B, N]
                    "sample": value.permute(1, 0).tolist(),
                    # Statistics over runs for each image.
                    "mean": mean_value.tolist(),
                    "std": std_value.tolist(),
                    "max": value.max(dim=0).values.tolist(),
                    "min": value.min(dim=0).values.tolist(),
                }
        return result_dicts

    def display(self, result_dicts):
        table = Table('results')
        average, std = {}, {}
        for key in result_dicts.keys():
            value = ['{:.3f}'.format(v) for v in result_dicts[key][get_eval_fn_cmp(key)]]
            table.add_column(key, value)
            average[key] = '{:.3f}'.format(np.mean(result_dicts[key][get_eval_fn_cmp(key)]))
            std[key] = '{:.3f}'.format(np.std(result_dicts[key][get_eval_fn_cmp(key)]))
        # for average
        table.add_row(['' for _ in result_dicts.keys()])
        table.add_row(['mean' for _ in result_dicts.keys()])
        table.add_row(average.values())
        table.add_row(['' for _ in result_dicts.keys()])
        table.add_row(['std' for _ in result_dicts.keys()])
        table.add_row(std.values())

        return table.get_string()

    def log_wandb(self, result_dicts):
        if not result_dicts:
            return

        first_metric_name = next(iter(result_dicts))
        first_comparison_key = get_eval_fn_cmp(first_metric_name)
        num_samples = len(
            result_dicts[first_metric_name][first_comparison_key]
        )

        for sample_idx in range(num_samples):
            log_dict = {
                key: result_dicts[key][get_eval_fn_cmp(key)][sample_idx]
                for key in result_dicts
            }
            wandb.log(log_dict)
        
        aggregate_log = {
            f"{key}_all": np.mean(
                result_dicts[key][get_eval_fn_cmp(key)]
            )
            for key in result_dicts
        }

        wandb.log(aggregate_log)


class Table(object):
    def __init__(self, title=None, field_names=None):
        """
            title:          str
            field_names:    list of field names
        """
        self.table = prettytable.PrettyTable(title=title, field_names=field_names)

    def add_rows(self, rows):
        """
            rows: list of tuples
        """
        self.table.add_rows(rows)

    def add_row(self, row):
        self.table.add_row(row)

    def add_column(self, fieldname, column):
        self.table.add_column(fieldname=fieldname, column=column)

    def get_string(self):
        """
            a markdown format table
        """
        _junc = self.table.junction_char
        if _junc != "|":
            self.table.junction_char = "|"
        markdown = [row for row in self.table.get_string().split("\n")[1:-1]]
        self.table.junction_char = _junc
        return "\n" + "\n".join(markdown)

    def get_latex_string(self):
        # TODO: to be done in future
        pass


__EVAL_FN__ = {}
__EVAL_FN_CMP__ = {}


def register_eval_fn(name: str):
    def wrapper(cls):
        if __EVAL_FN__.get(name, None):
            if __EVAL_FN__[name] != cls:
                warnings.warn(f"Name {name} is already registered!", UserWarning)
        __EVAL_FN__[name] = cls
        __EVAL_FN_CMP__[name] = cls.cmp
        cls.name = name
        return cls

    return wrapper


def get_eval_fn(name: str, **kwargs):
    if __EVAL_FN__.get(name, None) is None:
        raise NameError(f"Name {name} is not defined.")
    return __EVAL_FN__[name](**kwargs)


def get_eval_fn_cmp(name: str):
    return __EVAL_FN_CMP__[name]


class EvalFn(ABC):
    @staticmethod
    def norm(x):
        return (x * 0.5 + 0.5).clip(0, 1)

    @abstractmethod
    def __call__(self, gt, measurement, sample, reduction='none'):
        pass


@register_eval_fn('psnr')
class PeakSignalNoiseRatio(EvalFn):
    cmp = 'max'  # the higher, the better

    def __call__(self, gt, measurement, sample, reduction='none'):
        return psnr(self.norm(gt), self.norm(sample), data_range=1.0, reduction=reduction)


@register_eval_fn('ssim')
class StructuralSimilarityIndexMeasure(EvalFn):
    cmp = 'max'  # the higher, the better

    def __call__(self, gt, measurement, sample, reduction='none'):
        return ssim(self.norm(gt), self.norm(sample), data_range=1.0, reduction=reduction)


@register_eval_fn('lpips')
class LearnedPerceptualImagePatchSimilarity(EvalFn):
    cmp = 'min'  # the higher, the better

    def __init__(self, device="cuda", batch_size=8):
        self.device = torch.device(device)
        self.batch_size = int(batch_size)
        
        if self.batch_size <= 0:
            raise ValueError("LPIPS batch_size must be greater than zero.")
        self.lpips_fn = LPIPS(replace_pooling=True, reduction='none').to(self.device)
        self.lpips_fn.eval()

    def evaluate_in_batch(self, gt, pred):
        # batch_size = self.batch_size
        results = []
        with torch.inference_mode():
            for start in range(0, gt.shape[0], self.batch_size):
                end = min(start+self.batch_size, gt.shape[0])
                current_gt = self.norm(
                    gt[start:end]
                ).to(
                    self.device,
                    non_blocking=True,
                )

                current_pred = self.norm(
                    pred[start:end]
                ).to(
                    self.device,
                    non_blocking=True,
                )
                result = self.lpips_fn(
                    current_gt,
                    current_pred,
                )
                results.append(result.detach().cpu())
                del current_gt
                del current_pred
                del result
        return torch.cat(results, dim=0)

    def __call__(self, gt, measurement, sample, reduction='none'):
        res = self.evaluate_in_batch(gt, sample)
        if reduction == 'mean':
            res = res.mean()
        return res
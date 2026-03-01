# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file exceam in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import math
import inspect
from typing import Optional
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR


class LayerLambdaLR(LambdaLR):
    def __init__(self, optimizer: Optimizer, lr_lambda, last_epoch: int = -1):
        use_llrd = False
        n_params = 0

        if not isinstance(lr_lambda, (list, tuple)) and callable(lr_lambda):
            sig = inspect.signature(lr_lambda)
            n_params = len(sig.parameters)
            if n_params >= 3:
                use_llrd = True

        if use_llrd:
            fn = lr_lambda
            group_lambdas = []
            for group in optimizer.param_groups:
                llrd = float(group.get("llrd", 1.0))
                llrd_scale = float(group.get("llrd_scale", 0.0))
                llrd_mode = str(group.get("llrd_mode", "exp"))
                llrd_scale_max = float(group.get("llrd_scale_max", 1.0))

                def group_lambda(
                    step,
                    fn=fn,
                    n_params=n_params,
                    llrd=llrd,
                    llrd_scale=llrd_scale,
                    llrd_mode=llrd_mode,
                    llrd_scale_max=llrd_scale_max,
                ):
                    if n_params >= 5:
                        return fn(step, llrd, llrd_scale, llrd_mode, llrd_scale_max)
                    if n_params == 4:
                        return fn(step, llrd, llrd_scale, llrd_mode)
                    return fn(step, llrd, llrd_scale)

                group_lambdas.append(group_lambda)

            lr_lambda = group_lambdas

        super().__init__(optimizer, lr_lambda, last_epoch)


def _llrd_factor(llrd: float, llrd_scale: float, llrd_mode: str, llrd_scale_max: float) -> float:
    mode = str(llrd_mode).lower()
    if mode == "linear":
        # llrd_scale is "distance from deepest layer" (deep=0, shallow=max).
        denom = max(1.0, float(llrd_scale_max))
        depth_ratio = 1.0 - (float(llrd_scale) / denom)
        depth_ratio = min(max(depth_ratio, 0.0), 1.0)
        factor = float(llrd) + (1.0 - float(llrd)) * depth_ratio
    else:
        # Legacy behavior.
        factor = float(max(llrd, 1e-12)) ** float(llrd_scale)
    return max(0.0, float(factor))


def get_llrd_cosine_schedule(
    optimizer: Optimizer,
    num_warmup_steps: int,
    num_training_steps: int,
    num_cycles: float = 0.5,
    last_epoch: int = -1,
    llrd_end: Optional[float] = None,
    llrd_mode: str = "exp",
) -> LayerLambdaLR:
    if num_training_steps <= 0:
        raise ValueError("num_training_steps must be > 0")

    num_warmup_steps = int(num_warmup_steps)
    num_training_steps = int(num_training_steps)

    def lr_lambda(
        current_step: int,
        llrd: float,
        llrd_scale: float,
        llrd_mode_group: str = llrd_mode,
        llrd_scale_max: float = 1.0,
    ) -> float:
        if current_step < 0:
            return 0.0

        if llrd_end is not None:
            progress = float(current_step) / float(max(1, num_training_steps))
            cosine = 0.5 * (1.0 - math.cos(math.pi * progress))
            llrd = llrd + (llrd_end - llrd) * cosine

        factor = _llrd_factor(llrd, llrd_scale, llrd_mode_group, llrd_scale_max)

        # global warmup, same for all layers
        if current_step < num_warmup_steps and num_warmup_steps > 0:
            base = float(current_step) / float(max(1, num_warmup_steps))
            return factor * base

        if current_step >= num_training_steps:
            return 0.0

        if num_training_steps == num_warmup_steps:
            progress = 1.0
        else:
            progress = float(current_step - num_warmup_steps) / float(
                max(1, num_training_steps - num_warmup_steps)
            )
        progress = min(max(progress, 0.0), 1.0)

        cosine = 0.5 * (
            1.0 + math.cos(math.pi * 2.0 * num_cycles * progress)
        )

        return factor * cosine

    return LayerLambdaLR(optimizer, lr_lambda, last_epoch=last_epoch)


def get_llrd_cosine_schedule_with_warmup(
    optimizer: Optimizer,
    num_warmup_steps: int,
    num_training_steps: int,
    num_cycles: float = 0.5,
    last_epoch: int = -1,
    llrd_mode: str = "exp",
) -> LayerLambdaLR:
    if num_training_steps <= 0:
        raise ValueError("num_training_steps must be > 0")

    num_warmup_steps = int(num_warmup_steps)
    num_training_steps = int(num_training_steps)

    def lr_lambda(
        current_step: int,
        llrd: float,
        llrd_scale: float,
        llrd_mode_group: str = llrd_mode,
        llrd_scale_max: float = 1.0,
    ) -> float:
        if current_step < 0:
            return 0.0

        factor = _llrd_factor(llrd, llrd_scale, llrd_mode_group, llrd_scale_max)

        if num_warmup_steps > 0:
            raw_warmup = int(round(num_warmup_steps * min(factor, 1.0)))
            raw_warmup = max(0, min(raw_warmup, num_warmup_steps))

            # invert: larger factor -> shorter warmup_layer, smaller factor -> longer warmup_layer
            warmup_layer = num_warmup_steps - raw_warmup
            warmup_layer = max(1, min(warmup_layer, num_training_steps))

            if current_step < warmup_layer:
                base = float(current_step) / float(max(1, warmup_layer))
                return factor * base
        else:
            warmup_layer = 0

        if current_step >= num_training_steps:
            return 0.0

        decay_den = max(1, num_training_steps - warmup_layer)
        progress = float(current_step - warmup_layer) / float(decay_den)
        progress = min(max(progress, 0.0), 1.0)

        cosine = 0.5 * (
            1.0 + math.cos(math.pi * 2.0 * num_cycles * progress)
        )

        return factor * cosine

    return LayerLambdaLR(optimizer, lr_lambda, last_epoch=last_epoch)

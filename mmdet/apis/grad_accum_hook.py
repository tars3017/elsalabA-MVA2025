from mmcv.runner import OptimizerHook, HOOKS

@HOOKS.register_module()
class GradientAccumulationOptimizerHook(OptimizerHook):
    def __init__(self, cumulative_iters=1, **kwargs):
        super().__init__(**kwargs)
        self.cumulative_iters = cumulative_iters
        self._inner_count = 0

    def after_train_iter(self, runner):
        self._inner_count += 1
        loss = runner.outputs['loss']
        loss = loss / self.cumulative_iters
        loss.backward()

        if self._inner_count % self.cumulative_iters == 0:
            if self.grad_clip is not None:
                self.clip_grads(runner.model.parameters())
            runner.optimizer.step()
            runner.optimizer.zero_grad()

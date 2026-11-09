"""Executor and policy trainability boundaries."""



def set_phase(engine, phase):
    warmup = phase == "warmup"
    engine.model.requires_grad_(warmup)
    engine.executor.set_execution_trainable(warmup)
    engine.policy.requires_grad_(not warmup)
    # All backbone dropout stays off. LoRA/projection gradients still flow in eval mode.
    engine.executor.eval()
    engine.model.train(warmup)
    engine.policy.train(not warmup)

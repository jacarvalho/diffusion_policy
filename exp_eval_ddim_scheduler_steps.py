"""
Usage:
python eval.py --checkpoint data/image/pusht/diffusion_policy_cnn/train_0/checkpoints/latest.ckpt -o data/pusht_eval_output
"""
import re
import sys

from experiment_launcher import single_experiment_yaml, run_experiment
from omegaconf import open_dict

# use line-buffering for both stdout and stderr
sys.stdout = open(sys.stdout.fileno(), mode='w', buffering=1)
sys.stderr = open(sys.stderr.fileno(), mode='w', buffering=1)

import os
import pathlib
import click
import hydra
import torch
import dill
import wandb
import json
from diffusion_policy.workspace.base_workspace import BaseWorkspace


@single_experiment_yaml
def experiment(
    checkpoint: str = 'data_train_eval/experiments/low_dim/lift_ph/diffusion_policy_cnn/train_0/checkpoints/epoch=0450-test_mean_score=1.000.ckpt',
    num_inference_steps: int = 10,

    device: str = 'cuda:0',

    #######################################
    # MANDATORY
    seed: int = 0,
    results_dir: str = 'logs',

    #######################################
    **kwargs
):
    # get directory of checkpoint
    checkpoint_dir = os.path.dirname(checkpoint)
    output_dir = re.sub('/checkpoints', '', checkpoint_dir)
    output_dir = re.sub('experiments', 'evaluations', output_dir)
    output_dir = os.path.join(output_dir, f'eval-ddim-{num_inference_steps}')

    if os.path.exists(output_dir):
        click.confirm(f"Output path {output_dir} already exists! Overwrite?", abort=True)
    pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # load checkpoint
    payload = torch.load(open(checkpoint, 'rb'), pickle_module=dill)
    cfg = payload['cfg']

    # Replace DDPM with DDIM scheduler for inference
    cfg.policy.noise_scheduler._target_ = f'diffusers.schedulers.scheduling_ddim.DDIMScheduler'
    assert 0 < num_inference_steps <= cfg.policy.noise_scheduler.num_train_timesteps
    cfg.policy.num_inference_steps = num_inference_steps
    del cfg.policy.noise_scheduler.variance_type  # not valid for DDIM

    cls = hydra.utils.get_class(cfg._target_)
    workspace = cls(cfg, output_dir=output_dir)
    workspace: BaseWorkspace
    workspace.load_payload(payload, exclude_keys=None, include_keys=None)
    
    # get policy from workspace
    policy = workspace.model
    if cfg.training.use_ema:
        policy = workspace.ema_model
    
    device = torch.device(device)
    policy.to(device)
    policy.eval()
    
    # run eval
    env_runner = hydra.utils.instantiate(
        cfg.task.env_runner,
        output_dir=output_dir)
    runner_log = env_runner.run(policy)
    
    # dump log to json
    json_log = dict()
    for key, value in runner_log.items():
        if isinstance(value, wandb.sdk.data_types.video.Video):
            json_log[key] = value._path
        else:
            json_log[key] = value
    out_path = os.path.join(output_dir, 'eval_log.json')
    json.dump(json_log, open(out_path, 'w'), indent=2, sort_keys=True)


if __name__ == '__main__':
    # Leave unchanged
    run_experiment(experiment)

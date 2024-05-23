import datetime
import glob
from itertools import product

from experiment_launcher import Launcher, is_local

LOCAL = is_local()
USE_CUDA = True

N_EXPS_IN_PARALLEL = 1

N_CORES = 8
MEMORY_SINGLE_JOB = 36000
MEMORY_PER_CORE = N_EXPS_IN_PARALLEL * MEMORY_SINGLE_JOB // N_CORES
PARTITION = 'amd2,amd'  # 'amd', 'rtx'
# GRES = 'gpu:1' if USE_CUDA else None  # gpu:rtx2080:1, gpu:rtx3080:1
GRES = 'gpu:rtx3090:1' if USE_CUDA else None  # gpu:rtx2080:1, gpu:rtx3080:1
CONDA_ENV = 'robodiff'  # None

BASE_OUTPUT_DIR = f'logs/{datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}'


launcher = Launcher(
    exp_name='exp_eval_ddim_scheduler_steps',
    exp_file='exp_eval_ddim_scheduler_steps',
    n_seeds=1,
    n_exps_in_parallel=N_EXPS_IN_PARALLEL,
    n_cores=N_CORES,
    memory_per_core=MEMORY_PER_CORE,
    days=2,
    hours=23,
    minutes=59,
    seconds=0,
    partition=PARTITION,
    gres=GRES,
    conda_env=CONDA_ENV,
    use_timestamp=False,
    compact_dirs=True,
    base_dir=BASE_OUTPUT_DIR
)

############################################################################
# Create and launch experiments

#########################
# get all checkpoint paths
# take the best checkpoint and not the latest
ckpt_paths = glob.glob(f'./data_experiments/**/epoch*.ckpt', recursive=True)


def get_checkpoints(tags_l):
    res = []
    for ckpt in ckpt_paths:
        for tags in tags_l:
            flag = 1
            for tag in tags:
                if tag not in ckpt:
                    flag = 0
                    break
            if flag:
                res.append(ckpt)
    return res


tasks = [
    'lift_ph',
    'lift_mh',
    'can_ph',
    'can_mh',
    'square_ph'
    'square_mh',
    'transport_ph',
    'transport_mh',
    'tool_hang_ph',
]

# Robomimic LOW-DIM - Diffusion Policy Transformer
checkpoints_robomimic_low_dim_diffusion_policy_transformer_l = get_checkpoints(
    [(task, 'low_dim', 'diffusion_policy_transformer') for task in tasks]
)

# Robomimic IMAGE - Diffusion Policy Transformer
checkpoints_robomimic_image_diffusion_policy_transformer_l = get_checkpoints(
    [(task, 'image', 'diffusion_policy_transformer') for task in tasks]
)


checkpoints_l = [
    *checkpoints_robomimic_low_dim_diffusion_policy_transformer_l,
    *checkpoints_robomimic_image_diffusion_policy_transformer_l,
]

num_inference_steps_l = [
    50,
    20,
    10,
    5,
    2
]

for i, (checkpoint, num_inference_steps) in enumerate(product(checkpoints_l, num_inference_steps_l)):
    launcher.add_experiment(
        exp__=i,
        checkpoint=checkpoint,
        num_inference_steps=num_inference_steps,
        base_output_dir=BASE_OUTPUT_DIR,
    )
launcher.run(LOCAL)


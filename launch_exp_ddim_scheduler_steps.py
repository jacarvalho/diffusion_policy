import datetime
from itertools import product

from experiment_launcher import Launcher, is_local

LOCAL = is_local()
USE_CUDA = True

N_EXPS_IN_PARALLEL = 1

N_CORES = 8
MEMORY_SINGLE_JOB = 32000
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
# Robomimic LOW-DIM
checkpoints_robomimic_low_dim_diffusion_policy_cnn_l = [
    # Lift
    f'data_experiments/experiments/low_dim/lift_ph/diffusion_policy_cnn/train_0/checkpoints/epoch=0450-test_mean_score=1.000.ckpt',
    f'data_experiments/experiments/low_dim/lift_ph/diffusion_policy_cnn/train_1/checkpoints/epoch=0400-test_mean_score=1.000.ckpt',
    f'data_experiments/experiments/low_dim/lift_ph/diffusion_policy_cnn/train_2/checkpoints/epoch=0300-test_mean_score=1.000.ckpt',

    f'data_experiments/experiments/low_dim/lift_mh/diffusion_policy_cnn/train_0/checkpoints/epoch=0250-test_mean_score=1.000.ckpt',
    f'data_experiments/experiments/low_dim/lift_mh/diffusion_policy_cnn/train_1/checkpoints/epoch=0250-test_mean_score=1.000.ckpt',
    f'data_experiments/experiments/low_dim/lift_mh/diffusion_policy_cnn/train_2/checkpoints/epoch=0250-test_mean_score=1.000.ckpt',

    # Square
    f'data_experiments/experiments/low_dim/square_ph/diffusion_policy_cnn/train_0/checkpoints/epoch=1750-test_mean_score=1.000.ckpt',
    f'data_experiments/experiments/low_dim/square_ph/diffusion_policy_cnn/train_1/checkpoints/epoch=1600-test_mean_score=1.000.ckpt',
    f'data_experiments/experiments/low_dim/square_ph/diffusion_policy_cnn/train_2/checkpoints/epoch=0750-test_mean_score=1.000.ckpt',

    f'data_experiments/experiments/low_dim/square_mh/diffusion_policy_cnn/train_0/checkpoints/epoch=4600-test_mean_score=1.000.ckpt',
    f'data_experiments/experiments/low_dim/square_mh/diffusion_policy_cnn/train_1/checkpoints/epoch=2650-test_mean_score=0.955.ckpt',
    f'data_experiments/experiments/low_dim/square_mh/diffusion_policy_cnn/train_2/checkpoints/epoch=1250-test_mean_score=1.000.ckpt',

    # Can
    f'data_experiments/experiments/low_dim/can_ph/diffusion_policy_cnn/train_0/checkpoints/epoch=0350-test_mean_score=1.000.ckpt',
    f'data_experiments/experiments/low_dim/can_ph/diffusion_policy_cnn/train_1/checkpoints/epoch=0750-test_mean_score=1.000.ckpt',
    f'data_experiments/experiments/low_dim/can_ph/diffusion_policy_cnn/train_2/checkpoints/epoch=0300-test_mean_score=1.000.ckpt',

    f'data_experiments/experiments/low_dim/can_mh/diffusion_policy_cnn/train_0/checkpoints/epoch=3200-test_mean_score=1.000.ckpt',
    f'data_experiments/experiments/low_dim/can_mh/diffusion_policy_cnn/train_1/checkpoints/epoch=0550-test_mean_score=1.000.ckpt',
    f'data_experiments/experiments/low_dim/can_mh/diffusion_policy_cnn/train_2/checkpoints/epoch=0450-test_mean_score=1.000.ckpt',

    # Transport
    f'data_experiments/experiments/low_dim/transport_ph/diffusion_policy_cnn/train_0/checkpoints/epoch=2100-test_mean_score=0.955.ckpt',
    f'data_experiments/experiments/low_dim/transport_ph/diffusion_policy_cnn/train_1/checkpoints/epoch=2350-test_mean_score=0.909.ckpt',
    f'data_experiments/experiments/low_dim/transport_ph/diffusion_policy_cnn/train_2/checkpoints/epoch=2800-test_mean_score=1.000.ckpt',

    f'data_experiments/experiments/low_dim/transport_mh/diffusion_policy_cnn/train_0/checkpoints/epoch=4100-test_mean_score=0.727.ckpt',
    f'data_experiments/experiments/low_dim/transport_mh/diffusion_policy_cnn/train_1/checkpoints/epoch=4700-test_mean_score=0.773.ckpt',
    f'data_experiments/experiments/low_dim/transport_mh/diffusion_policy_cnn/train_2/checkpoints/epoch=3800-test_mean_score=0.773.ckpt',

    # ToolHang
    f'data_experiments/experiments/low_dim/tool_hang_ph/diffusion_policy_cnn/train_0/checkpoints/epoch=0850-test_mean_score=0.818.ckpt',
    f'data_experiments/experiments/low_dim/tool_hang_ph/diffusion_policy_cnn/train_1/checkpoints/epoch=2000-test_mean_score=0.864.ckpt',
    f'data_experiments/experiments/low_dim/tool_hang_ph/diffusion_policy_cnn/train_2/checkpoints/epoch=3750-test_mean_score=0.864.ckpt',
]

# Robomimic IMAGE
checkpoints_robomimic_image_diffusion_policy_cnn_l = [
    # Lift
    f'data_experiments/experiments/image/lift_ph/diffusion_policy_cnn/train_0/checkpoints/epoch=0300-test_mean_score=1.000.ckpt',
    f'data_experiments/experiments/image/lift_ph/diffusion_policy_cnn/train_1/checkpoints/epoch=0250-test_mean_score=1.000.ckpt',
    f'data_experiments/experiments/image/lift_ph/diffusion_policy_cnn/train_2/checkpoints/epoch=0250-test_mean_score=1.000.ckpt',

    f'data_experiments/experiments/image/lift_mh/diffusion_policy_cnn/train_0/checkpoints/epoch=0250-test_mean_score=1.000.ckpt',
    f'data_experiments/experiments/image/lift_mh/diffusion_policy_cnn/train_1/checkpoints/epoch=0250-test_mean_score=1.000.ckpt',
    f'data_experiments/experiments/image/lift_mh/diffusion_policy_cnn/train_2/checkpoints/epoch=0300-test_mean_score=1.000.ckpt',

    # Square
    f'data_experiments/experiments/image/square_ph/diffusion_policy_cnn/train_0/checkpoints/epoch=1250-test_mean_score=1.000.ckpt',
    f'data_experiments/experiments/image/square_ph/diffusion_policy_cnn/train_1/checkpoints/epoch=2050-test_mean_score=0.955.ckpt',
    f'data_experiments/experiments/image/square_ph/diffusion_policy_cnn/train_2/checkpoints/epoch=2600-test_mean_score=1.000.ckpt',

    f'data_experiments/experiments/image/square_mh/diffusion_policy_cnn/train_0/checkpoints/epoch=0050-test_mean_score=1.000.ckpt',
    f'data_experiments/experiments/image/square_mh/diffusion_policy_cnn/train_1/checkpoints/epoch=1600-test_mean_score=0.955.ckpt',
    f'data_experiments/experiments/image/square_mh/diffusion_policy_cnn/train_2/checkpoints/epoch=1950-test_mean_score=1.000.ckpt',

    # Can
    f'data_experiments/experiments/image/can_ph/diffusion_policy_cnn/train_0/checkpoints/epoch=0350-test_mean_score=1.000.ckpt',
    f'data_experiments/experiments/image/can_ph/diffusion_policy_cnn/train_1/checkpoints/epoch=0400-test_mean_score=1.000.ckpt',
    f'data_experiments/experiments/image/can_ph/diffusion_policy_cnn/train_2/checkpoints/epoch=1150-test_mean_score=1.000.ckpt',

    f'data_experiments/experiments/image/can_mh/diffusion_policy_cnn/train_0/checkpoints/epoch=1200-test_mean_score=1.000.ckpt',
    f'data_experiments/experiments/image/can_mh/diffusion_policy_cnn/train_1/checkpoints/epoch=2550-test_mean_score=1.000.ckpt',
    f'data_experiments/experiments/image/can_mh/diffusion_policy_cnn/train_2/checkpoints/epoch=0600-test_mean_score=1.000.ckpt',

    # Transport
    f'data_experiments/experiments/image/transport_ph/diffusion_policy_cnn/train_0/checkpoints/epoch=2750-test_mean_score=1.000.ckpt',
    f'data_experiments/experiments/image/transport_ph/diffusion_policy_cnn/train_1/checkpoints/epoch=2050-test_mean_score=0.955.ckpt',
    f'data_experiments/experiments/image/transport_ph/diffusion_policy_cnn/train_2/checkpoints/epoch=2600-test_mean_score=1.000.ckpt',

    f'data_experiments/experiments/image/transport_mh/diffusion_policy_cnn/train_0/checkpoints/epoch=2850-test_mean_score=0.909.ckpt',
    f'data_experiments/experiments/image/transport_mh/diffusion_policy_cnn/train_1/checkpoints/epoch=2350-test_mean_score=0.864.ckpt',
    f'data_experiments/experiments/image/transport_mh/diffusion_policy_cnn/train_2/checkpoints/epoch=2500-test_mean_score=0.864.ckpt',

    # ToolHang
    f'data_experiments/experiments/image/tool_hang_ph/diffusion_policy_cnn/train_0/checkpoints/epoch=2150-test_mean_score=0.955.ckpt',
    f'data_experiments/experiments/image/tool_hang_ph/diffusion_policy_cnn/train_1/checkpoints/epoch=2650-test_mean_score=1.000.ckpt',
    f'data_experiments/experiments/image/tool_hang_ph/diffusion_policy_cnn/train_2/checkpoints/epoch=1250-test_mean_score=0.909.ckpt',

]


checkpoints_l = checkpoints_robomimic_low_dim_diffusion_policy_cnn_l
# checkpoints_l = checkpoints_robomimic_low_dim_diffusion_policy_cnn_l + checkpoints_robomimic_image_diffusion_policy_cnn_l

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


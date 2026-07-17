# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# This file contains the commands to run the pixel diffusion experiment on FFHQ dataset with DAPS 1K.
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++


# ++++ Nonlinear Tasks ++++
# phase retrieval
python posterior_sample.py \
+data=test-ffhq \
+model=ffhq256ddpm \
+task=phase_retrieval \
+sampler=edm_daps \
task_group=pixel \
save_dir=results/pixel/ffhq \
num_runs=4 \
sampler.diffusion_scheduler_config.num_steps=5 \
sampler.annealing_scheduler_config.num_steps=200 \
batch_size=20 \
name=phase_retrieval \
gpu=0 \
> results/logs/pixel/ffhq/phase_retrieval.log 2>&1 &

# nonlinear deblur
python posterior_sample.py \
+data=test-ffhq \
+model=ffhq256ddpm \
+task=nonlinear_blur \
+sampler=edm_daps \
task_group=pixel \
save_dir=results/pixel/ffhq \
num_runs=1 \
sampler.diffusion_scheduler_config.num_steps=5 \
sampler.annealing_scheduler_config.num_steps=200 \
batch_size=20 \
name=nonlinear_blur \
gpu=1 \
> results/logs/pixel/ffhq/nonlinear_blur.log 2>&1 &

# high dynamic range
python posterior_sample.py \
+data=test-ffhq \
+model=ffhq256ddpm \
+task=hdr \
+sampler=edm_daps \
task_group=pixel \
save_dir=results/pixel/ffhq \
num_runs=1 \
sampler.diffusion_scheduler_config.num_steps=5 \
sampler.annealing_scheduler_config.num_steps=200 \
batch_size=20 \
name=hdr \
gpu=2 \
> results/logs/pixel/ffhq/hdr.log 2>&1 &

# ++++ Linear Tasks ++++
# down sampling
python posterior_sample.py \
+data=test-ffhq \
+model=ffhq256ddpm \
+task=down_sampling \
+sampler=edm_daps \
task_group=pixel \
save_dir=results/pixel/ffhq \
num_runs=1 \
sampler.diffusion_scheduler_config.num_steps=5 \
sampler.annealing_scheduler_config.num_steps=200 \
batch_size=20 \
name=down_sampling \
gpu=3 \
> results/logs/pixel/ffhq/down_sampling.log 2>&1 &


# Gaussian blur
python posterior_sample.py \
+data=test-ffhq \
+model=ffhq256ddpm \
+task=gaussian_blur \
+sampler=edm_daps \
task_group=pixel \
save_dir=results/pixel/ffhq \
num_runs=1 \
sampler.diffusion_scheduler_config.num_steps=5 \
sampler.annealing_scheduler_config.num_steps=200 \
batch_size=20 \
name=gaussian_blur \
gpu=4 \
> results/logs/pixel/ffhq/gaussian_blur.log 2>&1 &


# motion blur
python posterior_sample.py \
+data=test-ffhq \
+model=ffhq256ddpm \
+task=motion_blur \
+sampler=edm_daps \
task_group=pixel \
save_dir=results/pixel/ffhq \
num_runs=1 \
sampler.diffusion_scheduler_config.num_steps=5 \
sampler.annealing_scheduler_config.num_steps=200 \
batch_size=20 \
name=motion_blur \
gpu=5 \
> results/logs/pixel/ffhq/motion_blur.log 2>&1 &


# box inpainting 
python posterior_sample.py \
+data=test-ffhq \
+model=ffhq256ddpm \
+task=inpainting \
+sampler=edm_daps \
task_group=pixel \
save_dir=results/pixel/ffhq \
num_runs=1 \
sampler.diffusion_scheduler_config.num_steps=5 \
sampler.annealing_scheduler_config.num_steps=200 \
batch_size=20 \
name=inpainting \
gpu=6 \
> results/logs/pixel/ffhq/inpainting.log 2>&1 &


# random inpainting
python posterior_sample.py \
+data=test-ffhq \
+model=ffhq256ddpm \
+task=inpainting_rand \
+sampler=edm_daps \
task_group=pixel \
save_dir=results/pixel/ffhq \
num_runs=1 \
sampler.diffusion_scheduler_config.num_steps=5 \
sampler.annealing_scheduler_config.num_steps=200 \
batch_size=20 \
name=inpainting_rand \
gpu=7 \
> results/logs/pixel/ffhq/inpainting_rand.log 2>&1 &

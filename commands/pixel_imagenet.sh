# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# This file contains the commands to run the pixel diffusion experiment on ImageNet dataset with DAPS 1K.
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++


# ++++ Nonlinear Tasks ++++
# phase retrieval
python posterior_sample.py \
+data=test-imagenet \
+model=imagenet256ddpm \
+task=phase_retrieval \
+sampler=edm_daps \
task_group=pixel \
save_dir=results/pixel/imagenet \
num_runs=4 \
sampler.diffusion_scheduler_config.num_steps=5 \
sampler.annealing_scheduler_config.num_steps=200 \
batch_size=20 \
name=phase_retrieval \
gpu=0 \
> results/logs/pixel/imagenet/phase_retrieval.log 2>&1 &

# nonlinear deblur
python posterior_sample.py \
+data=test-imagenet \
+model=imagenet256ddpm \
+task=nonlinear_blur \
+sampler=edm_daps \
task_group=pixel \
save_dir=results/pixel/imagenet \
num_runs=1 \
sampler.diffusion_scheduler_config.num_steps=5 \
sampler.annealing_scheduler_config.num_steps=200 \
batch_size=20 \
name=nonlinear_blur \
gpu=1 \
> results/logs/pixel/imagenet/nonlinear_blur.log 2>&1 & 

# high dynamic range
python posterior_sample.py \
+data=test-imagenet \
+model=imagenet256ddpm \
+task=hdr \
+sampler=edm_daps \
task_group=pixel \
save_dir=results/pixel/imagenet \
num_runs=1 \
sampler.diffusion_scheduler_config.num_steps=5 \
sampler.annealing_scheduler_config.num_steps=200 \
batch_size=20 \
name=hdr \
gpu=2 \
> results/logs/pixel/imagenet/hdr.log 2>&1 & 

# ++++ Linear Tasks ++++
# down sampling
python posterior_sample.py \
+data=test-imagenet \
+model=imagenet256ddpm \
+task=down_sampling \
+sampler=edm_daps \
task_group=pixel \
save_dir=results/pixel/imagenet \
num_runs=1 \
sampler.diffusion_scheduler_config.num_steps=5 \
sampler.annealing_scheduler_config.num_steps=200 \
batch_size=20 \
name=down_sampling \
gpu=3 \
> results/logs/pixel/imagenet/down_sampling.log 2>&1 & 

# Gaussian blur
python posterior_sample.py \
+data=test-imagenet \
+model=imagenet256ddpm \
+task=gaussian_blur \
+sampler=edm_daps \
task_group=pixel \
save_dir=results/pixel/imagenet \
num_runs=1 \
sampler.diffusion_scheduler_config.num_steps=5 \
sampler.annealing_scheduler_config.num_steps=200 \
batch_size=20 \
name=gaussian_blur \
gpu=4 \
> results/logs/pixel/imagenet/gaussian_blur.log 2>&1 & 

# motion blur
python posterior_sample.py \
+data=test-imagenet \
+model=imagenet256ddpm \
+task=motion_blur \
+sampler=edm_daps \
task_group=pixel \
save_dir=results/pixel/imagenet \
num_runs=1 \
sampler.diffusion_scheduler_config.num_steps=5 \
sampler.annealing_scheduler_config.num_steps=200 \
batch_size=20 \
name=motion_blur \
gpu=5 \
> results/logs/pixel/imagenet/motion_blur.log 2>&1 & 

# box inpainting 
python posterior_sample.py \
+data=test-imagenet \
+model=imagenet256ddpm \
+task=inpainting \
+sampler=edm_daps \
task_group=pixel \
save_dir=results/pixel/imagenet \
num_runs=1 \
sampler.diffusion_scheduler_config.num_steps=5 \
sampler.annealing_scheduler_config.num_steps=200 \
batch_size=20 \
name=inpainting \
gpu=6 \
> results/logs/pixel/imagenet/inpainting.log 2>&1 & 

# random inpainting
python posterior_sample.py \
+data=test-imagenet \
+model=imagenet256ddpm \
+task=inpainting_rand \
+sampler=edm_daps \
task_group=pixel \
save_dir=results/pixel/imagenet \
num_runs=1 \
sampler.diffusion_scheduler_config.num_steps=5 \
sampler.annealing_scheduler_config.num_steps=200 \
batch_size=20 \
name=inpainting_rand \
gpu=7 \
> results/logs/pixel/imagenet/inpainting_rand.log 2>&1 & 
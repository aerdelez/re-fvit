#!/bin/bash

# Array of methods
# methods=("transformer_attribution" "rollout" "attn_gradcam" "full_lrp" "attn_last_layer" "attr_rollout")
methods=("transformer_attribution" "rollout" "attn_last_layer")

# Loop over each method and submit a separate sbatch job
for method in "${methods[@]}"; do
    for is_neg in 0 1; do
        for attack_noise in 0 8 16 24 32; do
            # Determine if the current task should use the --with-dds flag
            for use_dds in 0 1; do
              # if method is not transformer_attribution and use_dds is 1, skip
                if [[ "$method" != "transformer_attribution" && ($use_dds -eq 1 || $attack_noise -gt 0) ]]; then
                    continue
                fi
                if [ $use_dds -eq 1 ]; then
                    job_time="00:15:00"
                else
                    job_time="00:15:00"
                fi
                # Set the job name dynamically
                job_name="pert_eval_${method}_${attack_noise}_dds_${use_dds}"

                # Submit the job with sbatch
                sbatch <<EOT
#!/bin/bash
#SBATCH --partition=gpu_mig
#SBATCH --gpus=1
#SBATCH --cpus-per-task=9
#SBATCH --ntasks=1
#SBATCH --job-name=$job_name
#SBATCH --cpus-per-task=9
#SBATCH --time=$job_time
#SBATCH --output=slurm_output_${job_name}.out

module purge
module load 2023
module load Anaconda3/2023.07-2
source activate fvit

cd $HOME/fact-ai/FViT-main

if [ $use_dds -eq 1 ]; then
    echo "Running job: Perturbation eval with $method, $attack_noise and dds neg=$is_neg"
    python baselines/ViT/pertubation_eval_from_hdf5.py --method=$method --use-dds --attack_noise=$attack_noise --neg=$is_neg
    echo "Running job: Perturbation eval with $method, $attack_noise and dds neg=$is_neg"
else
    echo "Running job: Perturbation eval with $method, $attack_noise without dds neg=$is_neg"
    python baselines/ViT/pertubation_eval_from_hdf5.py --method=$method --attack_noise=$attack_noise --neg=$is_neg
    echo "Running job: Perturbation eval with $method, $attack_noise without dds neg=$is_neg"
fi
EOT
            done
        done
    done
done
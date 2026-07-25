#!/bin/bash
# Parameter search script for noise_level_a
# Runs training with noise_level_a from 0.1 to 0.9 with interval 0.1

# Base command
BASE_CMD="python script/run.py --config-dir=cfg/gym/finetune/walker2d-v2 --config-name=ft_ppo_reflow_mlp"

# Environment name
ENV_NAME="walker2d-v2"

# Base parameters (you can modify these)
BASE_PARAMS="min_std=0.10 max_std=0.24 train.ent_coef=0.03 wandb.offline_mode=True"

# Parameter range
START=0.1
END=0.9
STEP=0.1

# Create results directory
RESULTS_DIR="param_search_results"
mkdir -p "$RESULTS_DIR"

# Log file
LOG_FILE="$RESULTS_DIR/param_search_$(date +%Y%m%d_%H%M%S).log"

echo "Starting parameter search for noise_level_a from $START to $END with step $STEP"
echo "Environment: $ENV_NAME"
echo "Results will be logged to: $LOG_FILE"
echo ""

# Loop through parameter values
for a in $(seq $START $STEP $END); do
    # Format to 1 decimal place
    a_formatted=$(printf "%.1f" $a)
    
    # Format noise_level_a for folder names (replace . with _)
    noise_str=$(echo "$a_formatted" | tr '.' '_')
    
    # Build logdir path with env name and noise_level_a
    # Using REINFLOW_LOG_DIR environment variable if available, otherwise relative path
    BASE_LOG_DIR="${REINFLOW_LOG_DIR:-log}"
    LOGDIR="${BASE_LOG_DIR}/gym/finetune/${ENV_NAME}_noise_${noise_str}_ppo_reflow_mlp_ta4_td4_tdf4/\${now:%Y-%m-%d}_\${now:%H-%M-%S}_seed\${seed}"
    
    # Build wandb dir path with env name and noise_level_a
    WANDB_DIR="./wandb_offline_${ENV_NAME}_noise_${noise_str}"
    
    echo "========================================="
    echo "Running with noise_level_a=$a_formatted"
    echo "Time: $(date)"
    echo "========================================="
    
    # Run the command
    CMD="$BASE_CMD $BASE_PARAMS model.noise_level_a=$a_formatted logdir=$LOGDIR wandb.dir=$WANDB_DIR"
    
    echo "Command: $CMD" | tee -a "$LOG_FILE"
    echo "" | tee -a "$LOG_FILE"
    
    # Execute and log output
    $CMD 2>&1 | tee -a "$LOG_FILE"
    
    EXIT_CODE=${PIPESTATUS[0]}
    
    if [ $EXIT_CODE -eq 0 ]; then
        echo "✓ Successfully completed noise_level_a=$a_formatted" | tee -a "$LOG_FILE"
    else
        echo "✗ Failed with exit code $EXIT_CODE for noise_level_a=$a_formatted" | tee -a "$LOG_FILE"
    fi
    
    echo "" | tee -a "$LOG_FILE"
    echo "Waiting 5 seconds before next run..." | tee -a "$LOG_FILE"
    sleep 5
done

echo ""
echo "========================================="
echo "Parameter search completed!"
echo "Results logged to: $LOG_FILE"
echo "========================================="


#!/bin/bash
# Seed search script for ant-v2 environment
# Runs training with 10 different seeds (42-51)

# Base command
BASE_CMD="python script/run.py --config-dir=cfg/gym/finetune/ant-v2 --config-name=ft_ppo_reflow_mlp"

# Environment name
ENV_NAME="ant-v2"

# Base parameters (you can modify these)
BASE_PARAMS="min_std=0.08 max_std=0.16 train.ent_coef=0.03 wandb.offline_mode=True"

# Seed range (10 seeds: 42-51)
SEEDS=(42 43 44 45 46 47 48 49 50 51)

# Create results directory
RESULTS_DIR="seed_search_results"
mkdir -p "$RESULTS_DIR"

# Log file
LOG_FILE="$RESULTS_DIR/seed_search_$(date +%Y%m%d_%H%M%S).log"

echo "Starting seed search for ant-v2 environment"
echo "Seeds: ${SEEDS[*]}"
echo "Results will be logged to: $LOG_FILE"
echo ""

# Loop through seeds
for seed in "${SEEDS[@]}"; do
    echo "========================================="
    echo "Running with seed=$seed"
    echo "Time: $(date)"
    echo "========================================="
    
    # Build logdir path with env name and seed
    # Using REINFLOW_LOG_DIR environment variable if available, otherwise relative path
    BASE_LOG_DIR="${REINFLOW_LOG_DIR:-log}"
    LOGDIR="${BASE_LOG_DIR}/gym/finetune/${ENV_NAME}_seed_${seed}_ppo_reflow_mlp_ta4_td4_tdf4/\${now:%Y-%m-%d}_\${now:%H-%M-%S}_seed\${seed}"
    
    # Build wandb dir path with env name and seed
    WANDB_DIR="./wandb_offline_${ENV_NAME}_seed_${seed}"
    
    # Run the command
    CMD="$BASE_CMD $BASE_PARAMS seed=$seed logdir=$LOGDIR wandb.dir=$WANDB_DIR"
    
    echo "Command: $CMD" | tee -a "$LOG_FILE"
    echo "" | tee -a "$LOG_FILE"
    
    # Execute and log output
    $CMD 2>&1 | tee -a "$LOG_FILE"
    
    EXIT_CODE=${PIPESTATUS[0]}
    
    if [ $EXIT_CODE -eq 0 ]; then
        echo "✓ Successfully completed seed=$seed" | tee -a "$LOG_FILE"
    else
        echo "✗ Failed with exit code $EXIT_CODE for seed=$seed" | tee -a "$LOG_FILE"
    fi
    
    echo "" | tee -a "$LOG_FILE"
    echo "Waiting 5 seconds before next run..." | tee -a "$LOG_FILE"
    sleep 5
done

echo ""
echo "========================================="
echo "Seed search completed!"
echo "Results logged to: $LOG_FILE"
echo "========================================="


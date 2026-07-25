#!/usr/bin/env python3
"""
Parameter search script for noise_level_a
Runs training with noise_level_a from 0.1 to 0.9 with interval 0.1
"""
import subprocess
import sys
import os
from datetime import datetime
from pathlib import Path

# Base command components
BASE_CMD = [
    "python", "script/run.py",
    "--config-dir=cfg/gym/finetune/Humanoid-v3",
    "--config-name=ft_ppo_reflow_mlp"
]

# Environment name (extracted from config or set explicitly)
ENV_NAME = "Humanoid-v3"

# Base parameters (you can modify these)
BASE_PARAMS = [
    "min_std=0.08",
    "max_std=0.16",
    "train.ent_coef=0.03",
    "wandb.offline_mode=True"
]

# Parameter range
START = 0.1
END = 0.9
STEP = 0.1

# Create results directory
results_dir = Path("param_search_results")
results_dir.mkdir(exist_ok=True)

# Log file
log_file = results_dir / f"param_search_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

def log_and_print(message, log_file_handle):
    """Print to both console and log file"""
    print(message)
    log_file_handle.write(message + "\n")
    log_file_handle.flush()

def main():
    with open(log_file, 'w') as log_f:
        log_and_print("=" * 50, log_f)
        log_and_print(f"Parameter search for noise_level_a", log_f)
        log_and_print(f"Environment: {ENV_NAME}", log_f)
        log_and_print(f"Range: {START} to {END} with step {STEP}", log_f)
        log_and_print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", log_f)
        log_and_print(f"Results logged to: {log_file}", log_f)
        log_and_print("=" * 50, log_f)
        log_and_print("", log_f)
        
        # Generate parameter values
        param_values = []
        current = START
        while current <= END + STEP / 2:  # Add small epsilon to handle floating point
            param_values.append(round(current, 1))
            current += STEP
        
        results = []
        
        for a in param_values:
            log_and_print("=" * 50, log_f)
            log_and_print(f"Running with noise_level_a={a:.1f}", log_f)
            log_and_print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", log_f)
            log_and_print("=" * 50, log_f)
            
            # Format noise_level_a for folder names (replace . with _)
            noise_str = f"{a:.1f}".replace(".", "_")
            
            # Build logdir path with env name and noise_level_a
            # Using REINFLOW_LOG_DIR environment variable if available, otherwise relative path
            base_log_dir = os.getenv("REINFLOW_LOG_DIR", "log")
            # Use Hydra template syntax - escape $ and {} properly in f-string
            logdir = f"{base_log_dir}/gym/finetune/{ENV_NAME}_noise_{noise_str}_ppo_reflow_mlp_ta4_td4_tdf4/${{now:%Y-%m-%d}}_${{now:%H-%M-%S}}_seed${{seed}}"
            
            # Build wandb dir path with env name and noise_level_a
            wandb_dir = f"./wandb_offline_{ENV_NAME}_noise_{noise_str}"
            
            # Build command
            cmd = BASE_CMD + BASE_PARAMS + [
                f"model.noise_level_a={a:.1f}",
                f"logdir={logdir}",
                f"wandb.dir={wandb_dir}"
            ]
            
            log_and_print(f"Command: {' '.join(cmd)}", log_f)
            log_and_print("", log_f)
            
            # Execute command
            try:
                result = subprocess.run(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    cwd=os.getcwd()
                )
                
                # Log output
                log_f.write(result.stdout)
                log_f.flush()
                
                if result.returncode == 0:
                    log_and_print(f"✓ Successfully completed noise_level_a={a:.1f}", log_f)
                    results.append((a, "SUCCESS"))
                else:
                    log_and_print(f"✗ Failed with exit code {result.returncode} for noise_level_a={a:.1f}", log_f)
                    results.append((a, f"FAILED ({result.returncode})"))
                    
            except Exception as e:
                log_and_print(f"✗ Exception occurred for noise_level_a={a:.1f}: {e}", log_f)
                results.append((a, f"EXCEPTION: {e}"))
            
            log_and_print("", log_f)
            log_and_print("Waiting 5 seconds before next run...", log_f)
            log_and_print("", log_f)
            
            # Small delay between runs
            import time
            time.sleep(5)
        
        # Summary
        log_and_print("", log_f)
        log_and_print("=" * 50, log_f)
        log_and_print("Parameter search completed!", log_f)
        log_and_print("=" * 50, log_f)
        log_and_print("", log_f)
        log_and_print("Summary:", log_f)
        for a, status in results:
            log_and_print(f"  noise_level_a={a:.1f}: {status}", log_f)
        log_and_print("", log_f)
        log_and_print(f"Full log available at: {log_file}", log_f)

if __name__ == "__main__":
    main()


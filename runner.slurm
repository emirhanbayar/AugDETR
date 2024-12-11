#!/bin/bash -l
#SBATCH -J DINO
#SBATCH --chdir=/home/ebayar/AugDETR
#SBATCH --gres=gpu:rtx3090:1
#SBATCH --output=/home/ebayar/AugDETR-HAE/logs/DINO_%j.out
#SBATCH --error=/home/ebayar/AugDETR-HAE/logs/DINO_%j.err
#SBATCH --time=2-00:00:00
#SBATCH --partition=debug

echo "=== Starting Job ==="

# Set up environment variables with correct CUDA paths
ls -la /usr/local/

export CUDA_HOME=/usr/local/cuda
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

# Print system information
echo "=== System Information ==="
nvidia-smi
echo "CUDA_HOME: $CUDA_HOME"
echo "PATH: $PATH"
echo "LD_LIBRARY_PATH: $LD_LIBRARY_PATH"

# Setup conda
source /home/ebayar/anaconda3/etc/profile.d/conda.sh

conda activate dl4

# Print Python and pip paths
echo "=== Python Environment ==="
which conda
which pip
which python
python --version
pip --version

# Verify PyTorch installation
echo "=== PyTorch Information ==="
python -c "import torch; print('PyTorch Version:', torch.__version__); print('CUDA Available:', torch.cuda.is_available()); print('CUDA Version:', torch.version.cuda); print('GPU Device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'No GPU')"

# Set CUDA architecture flags for compilation
export TORCH_CUDA_ARCH_LIST="8.6"

# Clean previous builds if they exist
echo "=== Cleaning Previous Builds ==="
rm -rf models/dino/ops/build

# Install ninja
pip install ninja

# Build CUDA extensions with verbose output
echo "=== Building CUDA Extensions ==="
cd models/dino/ops/
pwd

# Create a temporary setup config to force use of conda compilers
cat > setup.cfg << EOL
[build_ext]
compiler=unix
EOL

# Build the CUDA extensions
python setup.py build install

python setup.py build install
cd ../../../

# Create log directory if it doesn't exist
mkdir -p /home/ebayar/AugDETR-HAE/logs/DINO

# Set additional environment variables for distributed training
export WORLD_SIZE=$SLURM_NTASKS
export RANK=$SLURM_PROCID
export LOCAL_RANK=$SLURM_LOCALID
export MASTER_PORT=$(expr 10000 + $(echo -n $SLURM_JOBID | tail -c 4))
export MASTER_ADDR=$(scontrol show hostname $SLURM_NODELIST | head -n 1)

# Function to get timestamp
timestamp() {
    date "+%Y-%m-%d %H:%M:%S"
}

# Run the main training script using srun with output redirection and tee
echo "=== Running Main Training Script ==="

# Create a unique log directory for this run
LOG_DIR=/home/ebayar/AugDETR-HAE/logs/DINO/run_${SLURM_JOB_ID}
mkdir -p $LOG_DIR

# Launch training with srun and ensure output is captured
srun bash -c "
    # Redirect all output to a combined log file
    {
        echo \"\$(timestamp) [Worker \$SLURM_PROCID] Starting training\"
        
        PYTHONUNBUFFERED=1 python -u main.py \
            --output_dir $LOG_DIR \
            -c config/DINO/DINO_4scale.py \
            --coco_path /home/ebayar/coco \
            --options dn_scalar=100 embed_init_tgt=TRUE \
            dn_label_coef=1.0 dn_bbox_coef=1.0 use_ema=False \
            dn_box_noise_scale=1.0 \
            --num_workers 4 2>&1
            
        echo \"\$(timestamp) [Worker \$SLURM_PROCID] Training completed\"
    } | tee $LOG_DIR/worker_\${SLURM_PROCID}.log
"

echo "=== Job Completed ==="
date

# Deactivate the conda environment
conda deactivate
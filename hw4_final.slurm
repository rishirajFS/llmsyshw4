#!/bin/bash -l
#SBATCH --job-name=hw4_final
#SBATCH --partition=GPU-shared
#SBATCH --nodes=1
#SBATCH --gpus=v100-32:1
#SBATCH --time=3:00:00
#SBATCH --output=/jet/home/rnagaraj/mt_output_%j.log
#SBATCH --error=/jet/home/rnagaraj/mt_error_%j.log
#SBATCH -A cis250019p

set -e  # Exit on any error so we don't silently run with stale kernels

# Initialize module system (login shell -l should handle this, but be explicit)
source /etc/profile.d/lmod.sh 2>/dev/null || true

module load cuda/12.4.0

WORKDIR=/jet/home/rnagaraj/workspace/llmsyshw4

cd $WORKDIR

# Activate venv so python3 and all deps (including minitorch) are available
source .venv/bin/activate
export PYTHONPATH=$PYTHONPATH:$(pwd)

# Ensure minitorch is installed in the active venv
python3 -m pip install -e . -q

echo "Python $(python3 --version)"
echo "nvcc: $(nvcc --version | tail -1)"
python3 -m pip show minitorch 2>/dev/null | grep -E 'Name|Version|Location' || true

# Compile CUDA kernels (will fail early if nvcc is not available)
bash compile_cuda.sh

set +e  # Allow training to continue even if one run fails

echo '--- RUNNING WITHOUT FUSED KERNELS ---'
python3 project/run_machine_translation.py --use_fused_kernel=False

echo '--- RUNNING WITH FUSED KERNELS ---'
python3 project/run_machine_translation.py --use_fused_kernel=True

export PATH="/usr/local/cuda-12.8/bin:$PATH"
export LD_LIBRARY_PATH="/usr/local/cuda-12.8/lib64:$LD_LIBRARY_PATH"
export CUDA_HOME="/usr/local/cuda-12.8/"
export CPATH="/usr/local/cuda-12.8/targets/x86_64-linux/include:$CPATH"
# export TORCH_CUDA_ARCH_LIST="80"

# Quick sanity output (optional):
# nvcc -V
# python -c "import torch; print('torch.version.cuda =', torch.version.cuda)"
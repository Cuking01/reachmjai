cmake -S . -B ./build -DCMAKE_PREFIX_PATH=~/ML/libtorch/share/cmake -DCMAKE_CUDA_ARCHITECTURES="80;86" -DCMAKE_CUDA_COMPILER=/usr/local/cuda-12.9/bin/nvcc

#include <torch/torch.h>
#include <iostream>

int main() {
    std::cout << "CUDA available: " << torch::cuda::is_available() << std::endl;
    std::cout << "PyTorch version: " << TORCH_VERSION_MAJOR << "." << TORCH_VERSION_MINOR << "." << TORCH_VERSION_PATCH << std::endl;
    return 0;
}

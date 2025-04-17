#include <torch/torch.h>
#include <iostream>

constexpr int d = 5;  // 输入/隐藏层维度

// 小网络定义
struct SmallNetImpl : torch::nn::Module {
    torch::nn::Linear fc1{nullptr}, fc2{nullptr}, fc3{nullptr};

    SmallNetImpl() {
        fc1 = register_module("fc1", torch::nn::Linear(d, d));
        fc2 = register_module("fc2", torch::nn::Linear(d, d));
        fc3 = register_module("fc3", torch::nn::Linear(d, 1));
        reset_parameters();
    }

    void reset_parameters() {
        // 权重初始化（Kaiming初始化后乘2）
        torch::nn::init::kaiming_normal_(fc1->weight, 0.0, torch::kFanIn, torch::kReLU);
        fc1->weight.set_data(fc1->weight * 2.0);  // 非原地操作
        torch::nn::init::uniform_(fc1->bias, -0.1, 0.2);
        
        torch::nn::init::kaiming_normal_(fc2->weight, 0.0, torch::kFanIn, torch::kReLU);
        fc2->weight.set_data(fc2->weight * 2.0);
        torch::nn::init::uniform_(fc2->bias, -0.1, 0.2);
        
        torch::nn::init::kaiming_normal_(fc3->weight);
        torch::nn::init::zeros_(fc3->bias);

        // 禁用所有梯度
        for (auto& param : parameters()) {
            param.set_requires_grad(false);
        }
    }

    torch::Tensor forward(torch::Tensor x) {
        x = torch::relu(fc1->forward(x));
        x = torch::relu(fc2->forward(x));
        return fc3->forward(x);
    }
};
TORCH_MODULE(SmallNet);

// 大网络定义
struct BigNetImpl : torch::nn::Module {
    torch::nn::ModuleList layers;
    int k;  // 隐藏层数量
    int n;  // 每层神经元数

    BigNetImpl(int k = 3, int n = 64) : k(k), n(n) {
        // 输入层
        layers->push_back(torch::nn::Linear(d, n));
        
        // 隐藏层
        for (int i = 0; i < k; ++i) {
            layers->push_back(torch::nn::Linear(n, n));
        }
        
        // 输出层
        layers->push_back(torch::nn::Linear(n, 1));
        
        // 注册所有子模块
        for (size_t i = 0; i < layers->size(); ++i) {
            register_module("layer" + std::to_string(i), layers[i]);
        }
    }

    torch::Tensor forward(torch::Tensor x) {
        // 前k层使用ReLU
        for (size_t i = 0; i < layers->size() - 1; ++i) {
            x = layers[i]->as<torch::nn::Linear>()->forward(x);
        }
        // 输出层不使用激活函数
        return layers[layers->size()-1]->as<torch::nn::Linear>()->forward(x);
    }
};
TORCH_MODULE(BigNet);

int main() {
    // 配置参数
    const int epochs = 1000;
    const int batch_size = 512;
    const int eval_batch_size = 1024;
    const float lr = 0.001;
    const torch::Device device(torch::kCUDA);  // 或 torch::kCPU

    // 初始化网络
    SmallNet small_net;
    BigNet big_net(4, 128);  // 4个隐藏层，每层128个神经元
    small_net->to(device);
    big_net->to(device);

    // 优化器
    torch::optim::Adam optimizer(big_net->parameters(), lr);

    // 训练循环
    for (int epoch = 1; epoch <= epochs; ++epoch) {
        // 训练阶段
        {
            // 生成新的训练数据（输入范围[-1, 1)）
            auto train_inputs = torch::empty({batch_size, d}, device)
                                  .uniform_(-1, 1)
                                  .requires_grad_(false);
            
            // 生成目标输出
            auto train_targets = small_net->forward(train_inputs).detach();

            // 前向传播
            auto output = big_net->forward(train_inputs);
            auto loss = torch::mse_loss(output, train_targets);

            // 反向传播
            optimizer.zero_grad();
            loss.backward();
            optimizer.step();

            // 验证阶段（每10个epoch）
            if (epoch % 1 == 0) {
                torch::NoGradGuard no_grad;
                // 生成新验证数据
                auto val_inputs = torch::empty({eval_batch_size, d}, device)
                                    .uniform_(-1, 1)
                                    .requires_grad_(false);
                auto val_targets = small_net->forward(val_inputs);
                
                // 计算验证损失
                auto val_output = big_net->forward(val_inputs);
                auto val_loss = torch::mse_loss(val_output, val_targets);

                std::printf("Epoch [%4d/%d]  Train Loss: %.4f  Val Loss: %.4f\n",
                           epoch, epochs,
                           loss.item<float>(),
                           val_loss.item<float>());
            }
        }

        
    }

    return 0;
}

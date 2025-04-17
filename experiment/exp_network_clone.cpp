#include <torch/torch.h>

constexpr int d = 10;      // 输入维度
constexpr int k = 4;       // 大网络的隐藏层数
constexpr int n = 128;     // 大网络每层神经元数
constexpr int batch_size = 512;
constexpr int epochs = 1000;

// 小网络定义
struct SmallNet : torch::nn::Module {
    torch::nn::Linear fc1{nullptr}, fc2{nullptr}, fc3{nullptr};

    SmallNet() {
        fc1 = register_module("fc1", torch::nn::Linear(d, d));
        fc2 = register_module("fc2", torch::nn::Linear(d, d));
        fc3 = register_module("fc3", torch::nn::Linear(d, 1));
        reset_parameters();
        for (auto& param : this->parameters()) {
        param.set_requires_grad(false);
    }

    void reset_parameters() {
        // 自定义权重初始化
        torch::nn::init::kaiming_normal_(fc1->weight, 0.0, torch::kFanIn, torch::kReLU);
        fc1->weight = fc1->weight.mul(2.0);
        torch::nn::init::uniform_(fc1->bias, -0.1, 0.2);

        torch::nn::init::kaiming_normal_(fc2->weight, 0.0, torch::kFanIn, torch::kReLU);
        fc2->weight = fc2->weight.mul(2.0);
        torch::nn::init::uniform_(fc2->bias, -0.1, 0.2);

        torch::nn::init::kaiming_normal_(fc3->weight);
        torch::nn::init::zeros_(fc3->bias);
    }

    torch::Tensor forward(torch::Tensor x) {
        x = torch::relu(fc1(x));
        x = torch::relu(fc2(x));
        x = fc3(x);  // 输出层无激活函数
        return x;
    }
};

// 大网络定义
struct LargeNet : torch::nn::Module {
    torch::nn::Sequential layers;

    LargeNet(int input_dim, int output_dim, int hidden_layers, int hidden_size) {
        // 输入层
        layers->push_back(torch::nn::Linear(input_dim, hidden_size));
        layers->push_back(torch::nn::ReLU());

        // 隐藏层
        for (int i = 0; i < hidden_layers; ++i) {
            layers->push_back(torch::nn::Linear(hidden_size, hidden_size));
            layers->push_back(torch::nn::ReLU());
        }

        // 输出层
        layers->push_back(torch::nn::Linear(hidden_size, output_dim));
        register_module("layers", layers);
    }

    torch::Tensor forward(torch::Tensor x) {
        return layers->forward(x);
    }
};

int main() {
    // 设备选择
    torch::Device device(torch::kCPU);

    // 生成训练数据
    auto small_net = std::make_shared<SmallNet>();
    auto inputs = torch::rand({batch_size, d}, device) * 2 - 1;  // [-1, 1]均匀分布
    auto targets = small_net->forward(inputs).detach();  // 确保不计算梯度

    // 初始化大网络
    auto large_net = std::make_shared<LargeNet>(d, 1, k, n);
    large_net->to(device);

    // 优化器和损失函数
    torch::optim::Adam optimizer(large_net->parameters(), 0.001);
    auto criterion = torch::nn::MSELoss();

    // 训练循环
    for (int epoch = 1; epoch <= epochs; ++epoch) {
        optimizer.zero_grad();
        auto outputs = large_net->forward(inputs);
        auto loss = criterion(outputs, targets);
        loss.backward();
        optimizer.step();

        if (epoch % 1 == 0) {
            std::cout << "Epoch [" << epoch << "/" << epochs 
                      << "], Loss: " << loss.item<float>() << std::endl;
        }
    }

    return 0;
}

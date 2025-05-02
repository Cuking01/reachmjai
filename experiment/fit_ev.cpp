#include <torch/torch.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/cuda/CUDAGuard.h>
#include <iostream>
#include <math.h>
#include <time.h>

float std_init_range(int n,int m)
{
    return sqrtf(6.0f/(n+m));
}


struct FCN : torch::nn::Module {
    torch::nn::Linear fc1{nullptr}, fc2{nullptr};

    FCN(int d1,int d2,int d3) {
        fc1 = register_module("fc1", torch::nn::Linear(d1, d2));
        fc2 = register_module("fc2", torch::nn::Linear(d2, d3));
        init_linear(fc1,d1,d2);
        init_linear(fc2,d2,d3);
    }

    static void init_linear(torch::nn::Linear&fc,int in,int out)
    {
        float w=std_init_range(in,out);
        torch::nn::init::uniform_(fc->weight,-w,w);
        //fc1->weight.set_data(fc1->weight * 2.0);
        torch::nn::init::uniform_(fc->bias, -0.1, 0.2);
    }

    torch::Tensor forward(torch::Tensor x) {
        x = torch::relu(fc1->forward(x));
        return fc2->forward(x);
    }
};

// 生成符合条件的随机采样值
torch::Tensor sample_from_range(torch::Tensor target, torch::Tensor range) {
    torch::Tensor low = target - torch::abs(range);
    torch::Tensor high = target + torch::abs(range);
    torch::Tensor rand = torch::rand_like(target);
    return low + rand * (high - low);
}

int main()
{
    int input_size=2,output_size=2;
    float lr=0.01;

    FCN target(input_size,2,output_size);
    FCN range(input_size,2,output_size);
    FCN f(input_size,10,output_size);
    int batch_size=8192;
    
    torch::nn::MSELoss mse;

    for(int i=1;i<=1000;i++)
    {
        f.zero_grad();
        for(int j=0;j<i;j++)
        {
            torch::Tensor x = torch::randn({ batch_size, input_size });
            torch::Tensor target_output = target.forward(x);
            torch::Tensor range_output = range.forward(x);
            torch::Tensor sample = sample_from_range(target_output,range_output);  // 生成采样值
            torch::Tensor y=f.forward(x);
            torch::Tensor loss=mse(y,sample);

            loss.backward();
        }

        for(auto&para:f.parameters())
        {
            para.data()=para.data()-(lr/i/batch_size)*para.grad();
        }

        torch::Tensor x = torch::randn({ batch_size, input_size });
        torch::Tensor target_output = target.forward(x);
        torch::Tensor yp = f.forward(x);
        final_loss = mse(target_output, yp);
        printf("epoch=%4d loss=%.4f\n",i,final_loss.item<float>());
        
    }
}

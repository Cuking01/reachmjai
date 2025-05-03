#include <torch/torch.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/cuda/CUDAGuard.h>
#include <iostream>
#include <math.h>
#include <time.h>
#include <algorithm>

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

    void set_requires_grad_false()
    {
        for(auto&param:parameters())
            param.set_requires_grad(false);
    }
};

// 生成符合条件的随机采样值
torch::Tensor sample_from_range(torch::Tensor target, torch::Tensor range) {
    torch::Tensor low = target - torch::abs(range);
    torch::Tensor high = target + torch::abs(range);
    torch::Tensor rand = torch::rand_like(target);
    return low + rand * (high - low);
}



struct Trainer
{
    FCN&target;
    FCN&range;
    FCN&f;

    int input_size;
    int output_size;

    Trainer(FCN&target,FCN&range,FCN&f,int input_size,int output_size):
        target(target),range(range),f(f),input_size(input_size),output_size(output_size)
    {

    }

    float get_grad(int base_batch_size,int64_t k,torch::nn::MSELoss&mse)
    {
        float loss_sum=0;
        for(int64_t i=0;i<k;i++)
        {
            torch::Tensor x = torch::randn({ base_batch_size, input_size });
            torch::Tensor target_output = target.forward(x);
            torch::Tensor range_output = range.forward(x);

            torch::Tensor sample = sample_from_range(target_output,range_output);  // 生成采样值
            torch::Tensor y=f.forward(x);
            torch::Tensor loss=mse(y,sample);

            loss_sum+=loss.item<float>();
            loss.backward();
        }

        return loss_sum/k;
    }

    void update(float lr,int64_t k)
    {
        for(auto&para:f.parameters())
        {
            para.data()=para.data()-(lr/k)*para.grad();
        }
    }

    void look_real_loss(int64_t k)
    {
        float loss_sum=0;
        for(int64_t i=0;i<k;i++)
        {
            torch::Tensor x = torch::randn({ base_batch_size, input_size });
            torch::Tensor target_output = target.forward(x);

            torch::Tensor y=f.forward(x);
            torch::Tensor loss=mse(y,target_output);

            loss_sum+=loss.item<float>();
        }

        printf("*** real_loss=%.8f ***\n",loss_sum/k);
    }

    void train_to(const int64_t epoch_limit,const float lr,const int base_batch_size)
    {
        torch::nn::MSELoss mse;

        int64_t k=1;  //累加k次梯度
        int64_t all_batch_num=0;  //总的基础batch数

        float smoothed_loss=0;
        float long_history_smoothed_loss=1;

        static constexpr float alpha=0.05;
        static constexpr float beta=0.01;

        for(int64_t epoch_id=0;epoch_id<epoch_limit;)
        {
            epoch_id++;
            f.zero_grad();
            float loss=get_grad(base_batch_size,k,mse);
            update(lr,k);

            smoothed_loss=loss*alpha+smoothed_loss*(1-alpha);
            long_history_smoothed_loss=loss*beta+long_history_smoothed_loss*(1-beta);

            all_batch_num+=k*base_batch_size;

            std::cout<<"epoch="<<epoch_id<<" k="<<k<<" all_batch_num="<<all_batch_num;
            printf(" loss=%.8f\n",smoothed_loss);

            printf(" long_history_smoothed_loss=%.8f\n",long_history_smoothed_loss);

            look_real_loss(k);

            if(smoothed_loss>long_history_smoothed_loss)
            {
                k=std::min(k*2,int64_t(1000000));
                long_history_smoothed_loss*=1.1;
            }
        }
    }

};


int main()
{
    int input_size=4,output_size=4;
    float lr=0.1;

    FCN target(input_size,4,output_size);
    FCN range(input_size,4,output_size);
    FCN f(input_size,32,output_size);
    int batch_size=100;
    
    target.set_requires_grad_false();
    range.set_requires_grad_false();

    Trainer trainer(target,range,f,input_size,output_size);

    trainer.train_to(10000,0.1,64);
}

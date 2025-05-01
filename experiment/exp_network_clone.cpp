#include <torch/torch.h>
#include <torch/cuda.h>
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



int main()
{
    int input_size=8192,output_size=8192;
    float lr=0.01;

    FCN target(input_size,8192,output_size);
    FCN f(input_size,8192,output_size);
    int batch_size=8192;
    
    torch::nn::MSELoss mse;

    target.to(torch::kCUDA);
    f.to(torch::kCUDA);

    torch::Tensor x=torch::randn({batch_size,input_size}).to(torch::kCUDA);
    torch::Tensor y1=torch::randn({input_size,output_size}).to(torch::kCUDA);
    torch::Tensor y2=torch::randn({input_size,output_size}).to(torch::kCUDA);
    
    at::cuda::CUDAStream s1=torch::cuda::getStreamFromPool(),s2=torch::cuda::getStreamFromPool();

    for(int i=0;i<10;i++)
    {
        
        torch::Tensor y=target.forward(x);
        torch::Tensor yp=f.forward(x);
        // torch::Tensor loss=mse(y,yp);

        // f.zero_grad();
        // loss.backward();

        // lr*=0.999;

        // for(auto&para:f.parameters())
        // {
        //     para.data()=para.data()-lr*para.grad();
        // }
        // if(i%1==0)
        // {
        //     printf("epoch=%4d loss=%.4f\n",i+1,loss.item<float>());
        // }
        
    }
    torch::cuda::synchronize();
    int st=clock();

    for(int i=0;i<100;i++)
    {
        
        // torch::Tensor y=target.forward(x);
        // torch::Tensor yp=f.forward(x);

        {
            at::cuda::CUDAStreamGuard guard(s1);
            y1=torch::matmul(x,x);
        }

        {
            at::cuda::CUDAStreamGuard guard(s2);
            y2=torch::matmul(x,x);
        }
        
        // torch::Tensor loss=mse(y,yp);

        // f.zero_grad();
        // loss.backward();

        // lr*=0.999;

        // for(auto&para:f.parameters())
        // {
        //     para.data()=para.data()-lr*para.grad();
        // }
        // if(i%1==0)
        // {
        //     printf("epoch=%4d loss=%.4f\n",i+1,loss.item<float>());
        // }
        puts("12312");
        
    }
    torch::cuda::synchronize();
    int end=clock();

    printf("time=%d\n",end-st);
}

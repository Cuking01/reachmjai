#include <torch/torch.h>
#include <iostream>
#include <math.h>


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
    int input_size=2,output_size=2;
    float lr=0.1;

    FCN target(input_size,2,output_size);
    FCN f(input_size,8,output_size);
    int batch_size=10;
    
    torch::nn::MSELoss mse;

    target.to(torch::kCUDA);
    f.to(torch::kCUDA);

    for(int i=0;i<100000;i++)
    {
        torch::Tensor x=torch::randn({batch_size,input_size}).to(torch::kCUDA);
        torch::Tensor y=target.forward(x);
        torch::Tensor yp=f.forward(x);
        torch::Tensor loss=mse(y,yp);

        f.zero_grad();
        loss.backward();

        lr*=0.999;

        for(auto&para:f.parameters())
        {
            para.data()=para.data()-lr*para.grad();
        }
        if(i%100==99)
        {
            printf("epoch=%4d loss=%.4f\n",i+1,loss.item<float>());
        }
        
    }
}

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

    float get_grad(int base_batch_size,int64_t k)
    {
        float loss_sum=0;
        torch::nn::MSELoss mse;
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

    void look_real_loss(int64_t k,int base_batch_size)
    {
        float loss_sum=0;
        torch::nn::MSELoss mse;
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

    void pre_train(const float lr,const int base_batch_size)
    {
        for(int i=0;i<2000;i++)
        {
            f.zero_grad();
            get_grad(base_batch_size,1);
            update(lr,1);
        }
    }



    void train_for(const int64_t epoch_limit,const float lr,const int base_batch_size)
    {
        int64_t k=1;  //累加k次梯度
        int64_t all_batch_num=0;  //总的基础batch数
        int64_t epoch_id=0;

        float smoothed_loss=0;
        float long_history_smoothed_loss=1;

        static constexpr float alpha=0.05;
        static constexpr float beta=0.01;

        pre_train(lr,base_batch_size);


        auto stick=[this,&all_batch_num,&epoch_id,base_batch_size]()
        {
            static int64_t cnt=100;
            if(all_batch_num>cnt)
            {
                std::cout<<"epoch_id="<<epoch_id<<" all_batch_num="<<all_batch_num<<"\n";
                look_real_loss(100,base_batch_size);
                cnt*=2;
            }
        };


        for(;epoch_id<epoch_limit;)
        {
            epoch_id++;
            f.zero_grad();
            float loss=get_grad(base_batch_size,k);
            update(lr,k);

            smoothed_loss=loss*alpha+smoothed_loss*(1-alpha);
            long_history_smoothed_loss=loss*beta+long_history_smoothed_loss*(1-beta);

            all_batch_num+=k*base_batch_size;

            // std::cout<<"epoch="<<epoch_id<<" k="<<k<<" all_batch_num="<<all_batch_num;
            // printf(" loss=%.8f\n",smoothed_loss);

            // printf(" long_history_smoothed_loss=%.8f\n",long_history_smoothed_loss);

            // look_real_loss(k,base_batch_size);

            stick();

            if(smoothed_loss>long_history_smoothed_loss)
            {
                k=std::min(k*2,int64_t(1000000));
                long_history_smoothed_loss*=1.1;
            }
        }
    }

    void train_simple(const int64_t epoch_limit,const float lr,const int base_batch_size)
    {
        int64_t k=1;  //累加k次梯度
        int64_t all_batch_num=0;  //总的基础batch数
        int64_t epoch_id=0;
        float smoothed_loss=1;

        static constexpr float alpha=0.05;

        pre_train(lr,base_batch_size);

        auto stick=[this,&all_batch_num,&epoch_id,base_batch_size]()
        {
            static int64_t cnt=100;
            if(all_batch_num>cnt)
            {
                std::cout<<"epoch_id="<<epoch_id<<" all_batch_num="<<all_batch_num<<"\n";
                look_real_loss(100,base_batch_size);
                cnt*=2;
            }
        };

        for(;epoch_id<epoch_limit;)
        {
            epoch_id++;
            f.zero_grad();
            float loss=get_grad(base_batch_size,k);
            update(lr,k);

            smoothed_loss=loss*alpha+smoothed_loss*(1-alpha);

            all_batch_num+=k*base_batch_size;

            // std::cout<<"epoch="<<epoch_id<<" k="<<k<<" all_batch_num="<<all_batch_num;
            // printf(" loss=%.8f\n",smoothed_loss);

            stick();

            if(epoch_id%10==0)
            {
                //k=std::min(k*2,int64_t(1000000));
                k+=1;
            }
        }
    }

};


void test_multi()
{
    int input_size=4,output_size=4;
    float lr=0.1;

    FCN target(input_size,4,output_size);
    FCN range(input_size,4,output_size);
    target.set_requires_grad_false();
    range.set_requires_grad_false();

    std::vector<FCN> f;

    for(int i=0;i<10;i++)
    {
        f.emplace_back(input_size,32,output_size);
    }


    std::vector<Trainer> trainer;

    for(int i=0;i<10;i++)
    {
        trainer.emplace_back(target,range,f[i],input_size,output_size);
        printf("start to train %d\n",i);
        trainer[i].train_simple(400,0.1,64);
    }

    torch::Tensor x=torch::randn({1<<3,input_size});

    torch::nn::MSELoss mse;
    torch::Tensor y=target.forward(x);

    for(int i=0;i<10;i++)
    {
        
        torch::Tensor yp=f[i].forward(x);

        torch::Tensor loss=mse(y,yp);
        printf("loss of f[%d]:%.8f\n",i,loss.item<float>());

        std::cout<<"f["<<i<<"]\n"<<yp-y;

    }

    torch::Tensor yp=f[0].forward(x);

    for(int i=1;i<10;i++)
    {
        yp=yp+f[i].forward(x);
    }

    yp=0.1*yp;

    torch::Tensor loss=mse(y,yp);

    printf("loss of all=%.8f\n",loss.item<float>());
    std::cout<<"f["<<"all"<<"]\n"<<yp-y;
}


void test_one()
{
    int input_size=4,output_size=4;
    float lr=0.1;

    FCN target(input_size,4,output_size);
    FCN range(input_size,4,output_size);
    FCN f(input_size,32,output_size);
    FCN g(input_size,32,output_size);
    int batch_size=100;
    
    target.set_requires_grad_false();
    range.set_requires_grad_false();

    Trainer trainer(target,range,f,input_size,output_size);

    trainer.train_for(2000,0.1,64);

    Trainer trainer2(target,range,g,input_size,output_size);
    trainer2.train_simple(10000,0.1,64);
}

int main()
{
    test_multi();
}

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


namespace check_rand
{
// 计算单个张量的方差
double calculate_tensor_variance(const torch::Tensor& tensor) {
    double variance = 0.0;
    int num_elements = tensor.numel();
    for (int i = 0; i < num_elements; ++i) {
        double element = tensor.data_ptr<float>()[i];
        variance += (element - 0.5) * (element - 0.5);
    }
    return variance / num_elements;
}

// 计算均值向量的方差
double calculate_mean_variance(const std::vector<double>& means) {
    double variance = 0.0;
    for (double mean : means) {
        variance += (mean - 0.5) * (mean - 0.5);
    }
    return variance / 10.0;
}

int main() {
    // 生成一个 10x10 的张量，仅用于确定大小
    torch::Tensor template_tensor = torch::ones({10, 10});

    // 进行十次实验
    for (int experiment = 0; experiment < 10; ++experiment) {
        std::vector<double> means;
        std::vector<double> tensor_variances;
        std::vector<double> mean_deviations;
        double total_sum = 0.0;

        // 每次实验生成 10 个张量
        for (int i = 0; i < 10; ++i) {
            torch::Tensor random_tensor = torch::rand_like(template_tensor);
            // 计算当前张量的均值
            double mean = random_tensor.mean().item<double>();
            means.push_back(mean);
            total_sum += mean;

            // 计算当前张量均值与期望值 0.5 的偏差
            double deviation = mean - 0.5;
            mean_deviations.push_back(deviation);

            // 计算当前张量的方差
            double tensor_variance = calculate_tensor_variance(random_tensor);
            tensor_variances.push_back(tensor_variance);

            std::cout << "Experiment " << experiment + 1 << ", Tensor " << i + 1 << " Mean: " << mean
                      << ", Variance: " << tensor_variance
                      << ", Deviation from 0.5: " << deviation << std::endl;
        }

        // 计算 10 个张量总的均值
        double total_mean = total_sum / 10.0;

        // 计算 10 个张量总均值与期望值 0.5 的偏差
        double total_deviation = total_mean - 0.5;

        // 计算 10 个均值的方差
        double mean_variance = calculate_mean_variance(means);

        // 输出本次实验的均值方差、10 个张量总的均值以及对应的偏差
        std::cout << "Experiment " << experiment + 1 << " Mean Variance: " << mean_variance
                  << ", Total Mean of 10 Tensors: " << total_mean
                  << ", Total Deviation from 0.5: " << total_deviation << std::endl;
    }

    return 0;
}    
};


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
        for(int i=0;i<3000;i++)
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
    int input_size=2,output_size=2;
    float lr=0.1;

    FCN target(input_size,2,output_size);
    FCN range(input_size,2,output_size);
    target.set_requires_grad_false();
    range.set_requires_grad_false();

    std::vector<FCN> f;

    for(int i=0;i<10;i++)
    {
        f.emplace_back(input_size,16,output_size);
    }


    std::vector<Trainer> trainer;

    for(int i=0;i<10;i++)
    {
        trainer.emplace_back(target,range,f[i],input_size,output_size);
        printf("start to train %d\n",i);
        trainer[i].train_simple(2000,0.1,64);
    }

    torch::Tensor x=torch::randn({1<<15,input_size});

    torch::nn::MSELoss mse;
    torch::Tensor y=target.forward(x);

    for(int i=0;i<10;i++)
    {
        
        torch::Tensor yp=f[i].forward(x);

        torch::Tensor loss=mse(y,yp);
        printf("loss of f[%d]:%.8f\n",i,loss.item<float>());

        //std::cout<<"f["<<i<<"]\n"<<yp-y;

    }

    torch::Tensor yp=f[0].forward(x);

    for(int i=1;i<10;i++)
    {
        yp=yp+f[i].forward(x);
    }

    yp=0.1*yp;

    torch::Tensor loss=mse(y,yp);

    printf("loss of all=%.8f\n",loss.item<float>());
    //std::cout<<"f["<<"all"<<"]\n"<<yp-y;
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

    trainer.train_for(800,0.1,64);

    Trainer trainer2(target,range,g,input_size,output_size);
    trainer2.train_simple(10000,0.1,64);
}

int main()
{
    //check_rand::main();
    test_multi();
}

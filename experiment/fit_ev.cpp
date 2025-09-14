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

torch::Tensor target_fun(const torch::Tensor& xys) {

    // 定义权重矩阵 W (2x2)
    torch::Tensor W = torch::tensor({{0.8, -1.0}, {-0.8, 0.7}});

    // 定义偏置 b (2x1)
    torch::Tensor b = torch::tensor({-0.1, 0.1});

    // 矩阵乘法: output = ReLU(W * input^T + b)
    // xys 是 Nx2，转置后是 2xN，W 是 2x2，结果 W * xys^T 是 2xN
    torch::Tensor tmp = torch::mm(W, xys.t()) + b.unsqueeze(1);

    // 转置回 Nx2 并应用 ReLU
    tmp = torch::relu(tmp.t());

    torch::Tensor W2 = torch::tensor({{0.3,-0.1},{0.8,-1.1}});
    torch::Tensor b2 = torch::tensor({0.1,-0.1});

    tmp=torch::mm(W2,tmp.t())+b2.unsqueeze(1);

    return tmp.t();
}

torch::Tensor range_fun(const torch::Tensor& xys) {

    // 定义权重矩阵 W (2x2)
    torch::Tensor W = torch::tensor({{0.3, -0.1}, {-0.3, 0.7}});

    // 定义偏置 b (2x1)
    torch::Tensor b = torch::tensor({0.4, 0.4});

    // 矩阵乘法: output = ReLU(W * input^T + b)
    // xys 是 Nx2，转置后是 2xN，W 是 2x2，结果 W * xys^T 是 2xN
    torch::Tensor linear_output = torch::mm(W, xys.t()) + b.unsqueeze(1);

    // 转置回 Nx2 并应用 ReLU
    torch::Tensor output = torch::relu(linear_output.t());

    return output;
}


template <typename T, typename = void>
struct has_forward : std::false_type {};

template <typename T>
struct has_forward<T, std::void_t<decltype(std::declval<T>().forward(std::declval<torch::Tensor>()))>> 
    : std::true_type {};

// SFINAE-based dispatcher
template <typename Functor>
auto forward(Functor&& fun, torch::Tensor x) 
    -> std::enable_if_t<has_forward<std::decay_t<Functor>>::value, torch::Tensor> {
    return fun.forward(x);
}

template <typename Functor>
auto forward(Functor&& fun, torch::Tensor x) 
    -> std::enable_if_t<!has_forward<std::decay_t<Functor>>::value, torch::Tensor> {
    return fun(x);
}

template<typename Target_T,typename Range_T>
struct Trainer
{
    Target_T&target;
    Range_T&range;
    FCN&f;

    int input_size;
    int output_size;

    Trainer(Target_T&target,Range_T&range,FCN&f,int input_size,int output_size):
        target(target),range(range),f(f),input_size(input_size),output_size(output_size)
    {

    }

    float get_grad(int base_batch_size,int64_t k)
    {
        float loss_sum=0;
        torch::nn::MSELoss mse;
        for(int64_t i=0;i<k;i++)
        {
            torch::Tensor x = torch::rand({ base_batch_size, input_size });
            torch::Tensor target_output = forward(target,x);;
            torch::Tensor range_output = forward(range,x);;

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
            torch::Tensor x = torch::rand({ base_batch_size, input_size });
            torch::Tensor target_output = forward(target,x);

            torch::Tensor y=f.forward(x);
            torch::Tensor loss=mse(y,target_output);

            loss_sum+=loss.item<float>();
        }

        printf("*** real_loss=%.8f ***\n",loss_sum/k);
    }

    void pre_train(const float lr,const int base_batch_size)
    {
        // for(int i=0;i<30;i++)
        // {
        //     f.zero_grad();
        //     get_grad(base_batch_size,1);
        //     update(5*lr,1);
        // }
        // for(int i=0;i<300;i++)
        // {
        //     f.zero_grad();
        //     get_grad(base_batch_size,1);
        //     update(1.5*lr,1);
        // }
        // for(int i=0;i<3000;i++)
        // {
        //     f.zero_grad();
        //     get_grad(base_batch_size,1);
        //     update(lr,1);
        // }
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

            //stick();

            if(epoch_id%10==0)
            {
                //k=std::min(k*2,int64_t(1000000));
                k+=1;
            }
        }

        //低学习率微调

        for(int i=0;i<50;i++)
        {
            epoch_id++;
            f.zero_grad();
            float loss=get_grad(base_batch_size,k);
            update(lr*(0.3-0.002*i),k);
        }
    }

};


constexpr int test_B=1<<21;

void calc_syd(torch::Tensor& yp,torch::Tensor& y,std::string name)
{
    torch::Tensor tmp=yp-y;
    auto acc=tmp.accessor<float,2>();
    std::vector<float> ex,ey;
    double sumex=0,sumey=0;
    for(int i=0;i<test_B;i++)
    {
        ex.push_back(acc[i][0]);
        sumex+=acc[i][0];
        ey.push_back(acc[i][1]);
        sumey+=acc[i][1];
    }

    std::sort(ex.begin(),ex.end());
    std::sort(ey.begin(),ey.end());

    printf("%s.ex:\n",name.c_str());
    printf("sumex = %.8f\n",sumex);
    for(int i=0;i<15;i++)
    {
        printf("%.8f ",ex[test_B/15*i]);
        if(i%4==3)puts("");
    }
    printf("%.8f\n\n",ex.back());

    printf("%s.ey:\n",name.c_str());
    printf("sumey = %.8f\n",sumey);
    for(int i=0;i<15;i++)
    {
        printf("%.8f ",ey[test_B/15*i]);
        if(i%4==3)puts("");
    }
    printf("%.8f\n\n",ey.back());
}

void test_multi()
{
    int input_size=2,output_size=2;
    float lr=0.15;

    std::vector<FCN> f;

    for(int i=0;i<10;i++)
    {
        f.emplace_back(input_size,16,output_size);
    }


    std::vector<decltype(Trainer(target_fun,range_fun,f[0],input_size,output_size))> trainer;

    for(int i=0;i<10;i++)
    {
        trainer.emplace_back(target_fun,range_fun,f[i],input_size,output_size);
        printf("start to train %d\n",i);
        trainer[i].train_simple(200,0.1,64);
    }

    torch::NoGradGuard no_grad;

    double sum_loss=0;

    torch::Tensor x=torch::rand({test_B,input_size});

    torch::nn::MSELoss mse;
    torch::Tensor y=target_fun(x);

    for(int i=0;i<10;i++)
    {
        torch::Tensor yp=f[i].forward(x);
        torch::Tensor loss=mse(y,yp);
        sum_loss+=loss.item<float>();
        printf("loss of f[%d]:%.8f\n",i,loss.item<float>());

        std::string name="f["+std::to_string(i)+"]";

        calc_syd(yp,y,name);

        //std::cout<<"f["<<i<<"]\n"<<yp-y;

    }

    torch::Tensor yp=f[0].forward(x);

    for(int i=1;i<10;i++)
    {
        yp=yp+f[i].forward(x);
    }

    yp=0.1*yp;

    torch::Tensor loss=mse(y,yp);

    calc_syd(yp,y,"all");

    printf("sum of loss=%.8f\n",sum_loss);
    printf("loss of all=%.8f\n",loss.item<float>());



    //std::cout<<"f["<<"all"<<"]\n"<<yp-y;
}

void test_multi()
{
    int input_size=2,output_size=2;
    float lr=0.15;

    std::vector<FCN> f;

    for(int i=0;i<10;i++)
    {
        f.emplace_back(input_size,16,output_size);
    }


    std::vector<decltype(Trainer(target_fun,range_fun,f[0],input_size,output_size))> trainer;

    for(int i=0;i<10;i++)
    {
        trainer.emplace_back(target_fun,range_fun,f[i],input_size,output_size);
        printf("start to train %d\n",i);
        trainer[i].train_simple(200,0.1,64);
    }

    torch::NoGradGuard no_grad;

    double sum_loss=0;

    torch::Tensor x=torch::rand({test_B,input_size});

    torch::nn::MSELoss mse;
    torch::Tensor y=target_fun(x);

    for(int i=0;i<10;i++)
    {
        torch::Tensor yp=f[i].forward(x);
        torch::Tensor loss=mse(y,yp);
        sum_loss+=loss.item<float>();
        printf("loss of f[%d]:%.8f\n",i,loss.item<float>());

        std::string name="f["+std::to_string(i)+"]";

        calc_syd(yp,y,name);

        //std::cout<<"f["<<i<<"]\n"<<yp-y;

    }

    torch::Tensor yp=f[0].forward(x);

    for(int i=1;i<10;i++)
    {
        yp=yp+f[i].forward(x);
    }

    yp=0.1*yp;

    torch::Tensor loss=mse(y,yp);

    calc_syd(yp,y,"all");

    printf("sum of loss=%.8f\n",sum_loss);
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
    int T;
    scanf("%d",&T);
    

    torch::Tensor x=torch::tensor({{0.5,0.6}});
    std::cout<< x.sizes() <<std::endl;
    auto ans=target_fun(x);
    printf("%f %f\n",ans[0][0].item<float>(),ans[0][1].item<float>());
    while(T--)test_multi();

}

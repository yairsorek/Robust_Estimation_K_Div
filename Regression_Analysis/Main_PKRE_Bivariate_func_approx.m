% Illustrating the penalized K-regression estimator (PKRE) for robust training
% of GELU neural networks used for Bivariate function approximation 
% in the presence of input-output contamination in the training data


%% Define parameters
N = 200; % Sample size


p=2; % Dimensionality

h_vec=1:0.5:5; % Range of bandwidth parameters 
Wd_vec=[1e-4,1e-3,1e-2,0.05]; % Range of regulrization parameters 

L_h=length(h_vec);
L_Wd=length(Wd_vec);

K1=20; % Number of neurons first hidden layer 5 default
K2=10; % Number of neurons second hidden layer 3 default

U=1000; %Monte-Carlo simulations
tool=10^-9;
MaxIter=1000;

%% Input distribution parameters
a=1; % For nominal input distribution U[-a,a]


%% Noise parameters and outliers
L_eps=13;
eps_vec=linspace(0,0.3,L_eps);

mu_noise=0;
sigma_noise = 0.05;
sigma2_noise = sigma_noise^2;

%% Output outliers parameters

sigma_o_c=sqrt(10); % 20 default
sigma2_o_c=sigma_o_c^2;
mu_o_c=10; % 10 default

% Input outliers parameters

b=15; % For contaminating input distribution U[-b,b]




% % Initializations of weights and biases via https://pytorch.org/docs/stable/generated/torch.nn.Linear.html

% First hidden-layer weights and biases - number of input featurs=p
b_vec_init=-sqrt(1/p)+2*sqrt(1/p).*rand(K1,1);
A_init=-sqrt(1/p)+2*sqrt(1/p).*rand(K1,p);

% Second hidden-layer weights and biases - number of input featurs=K1
d_vec_init=-sqrt(1/K1)+2*sqrt(1/K1).*rand(K2,1);
C_init=-sqrt(1/K1)+2*sqrt(1/K1).*rand(K2,K1);

% Output-layer weights - number of input featurs=K2
v_vec_init=-sqrt(1/K2)+2*sqrt(1/K2).*rand(K2,1);

%% Variance initialization
sigma2_init=1;


%% Hyperparameters for ADAM optimizer https://pytorch.org/docs/stable/generated/torch.optim.Adam.html
Lr=0.01;
beta1=0.9;
beta2=0.999; 
Wd=1e-4; % Weight decay for regularizaton, defalut 1e-4
amsgrad=0;

%%  Activation option 1- ReLU, 2- GeLU
Act_opt=2;
%% Parameters for the K-fold CV
K_CV=5; % Number of folds for the CV algorithm
hx_0=1; % Anchor input bandwidth
hy_0=1; % Anchor output bandwidth

%%  
error_PMSERE_test=zeros(U,L_eps);
error_PKRE_test=zeros(U,L_eps);

%% External loop running over different contamination parameters

for k= 1:L_eps

    cont_ratio= eps_vec(k); % Con



    disp(['k=',num2str(k)]);
    
    tic;

    %% Internal loop - Monte-Carlo trials
    for u=1:U

    %% Creating data sets

    % Draw outlier indexes
    alphabet = [0,1];
    prob_eps = [1-cont_ratio,cont_ratio];
    U_eps = randsrc(1,N,[alphabet; prob_eps]);
    outliers_inds=find(U_eps);
        
    % Generate contaminated training data
    [x_train_c,y_train_c,~]= Generate_contaminated_data_GIT(a,mu_noise,sigma2_noise,b,mu_o_c,sigma2_o_c,outliers_inds,N);

    % Generate clean test data
    outliers_inds_test=[]; % No outiliers in test data;
    [x_test,~,trg_func_test]= Generate_contaminated_data_GIT(a,mu_noise,sigma2_noise,b,mu_o_c,sigma2_o_c,outliers_inds_test,N);
      
    %% MSERE approach

    % Tuning parameters via K-fold CV
    Wd_MLE= K_Fold_CV_penalized_MSE_2_layers_GIT(x_train_c,y_train_c,Wd_vec,L_Wd,K_CV,A_init,b_vec_init,C_init,d_vec_init,v_vec_init,Lr,beta1,beta2,amsgrad,tool,MaxIter);
    % Obtaining PMSERE network parameters
    [A_MLE,b_vec_MLE,C_MLE,d_vec_MLE,v_vec_MLE] =ADAM_MLE_2_layers_GIT(x_train_c,y_train_c,A_init,b_vec_init,C_init,d_vec_init,v_vec_init,Lr,beta1,beta2,Wd_MLE,amsgrad,tool,MaxIter);
  
    % Computing robust scale parameter for initialization of the PKRE's
    % noise variance estimate
    y_MLE_train_est=Func_approx_2_layers_GIT(x_train_c,A_MLE,b_vec_MLE,C_MLE,d_vec_MLE,v_vec_MLE);
    e_MLE_train=y_train_c-y_MLE_train_est;
    rob_scale_par=1.4826.* mad(e_MLE_train,1);



       %% Proposed PKRE approach
       sigma2_init=(rob_scale_par)^2; % Initialize noise variance estimate 

       % Tuning parameters via K-fold CV
       [hx,hy,Wd_K,K_Fold_CV_error] = K_Fold_CV_penalized_KDIV_2_layers_GIT(x_train_c,y_train_c,h_vec,Wd_vec,L_h,L_Wd,hx_0,hy_0,K_CV,A_init,b_vec_init,C_init,d_vec_init,v_vec_init,sigma2_init,Lr,beta1,beta2,amsgrad,tool,MaxIter,5);

       D_x_train_c = abs(pdist(x_train_c.')).^2; 
       D_x_train_c = squareform(D_x_train_c);
       D_y_train = abs(pdist(y_train_c.')).^2; 
       D_y_train = squareform(D_y_train);
       [w_z_train_c,exp_D_x_train_c,exp_D_z_train_c,s_gz] = calc_weight(D_x_train_c,D_y_train,hx,hy,N);

       % The proposed PKRE-based network parameters
       [A_PKRE,b_vec_PKRE,C_PKRE,d_vec_PKRE,v_vec_PKRE,sigma2_PKRE] = ADAM_PKRE_2_layers_GIT(x_train_c,y_train_c,hy,w_z_train_c,exp_D_x_train_c,A_init,b_vec_init,C_init,d_vec_init,v_vec_init,sigma2_init,Lr,beta1,beta2,Wd_K,amsgrad,tool,MaxIter);



     %% Computing testing errors
     y_PMSERE_test_est=Func_approx_2_layers(x_test,Act_opt,A_MLE,b_vec_MLE,C_MLE,d_vec_MLE,v_vec_MLE);
     y_PKRE_test_est=Func_approx_2_layers(x_test,Act_opt,A_PKRE,b_vec_PKRE,C_PKRE,d_vec_PKRE,v_vec_PKRE);
     

     error_PMSERE_test(u,k)=(1/N).*norm(y_PMSERE_test_est-trg_func_test)^2;
     error_PKRE_test(u,k)=(1/N).*norm(y_PKRE_test_est-trg_func_test)^2;
    


    end
    toc

end

RMSE_PKRE_test= sqrt(mean(error_PKRE_test));
RMSE_PMSERE_test = sqrt(mean(error_PMSERE_test));


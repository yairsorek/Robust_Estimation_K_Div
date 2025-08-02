function [A,b_vec,C,d_vec,v_vec,sigma2] = ADAM_PKRE_2_layers_GIT(x,y,hy,w_z,exp_D_x,A,b_vec,C,d_vec,v_vec,sigma2,Lr,beta1,beta2,Wd,amsgrad,tool,maxIter)

%% Inputs parameters:

% x, y - inputs and outputs data, respectively
% hy - bandwidth for the output's kernel
% w_z - Parzen's KDE based weights
% exp_D_x - negative exponential distances of the inputs
% A, b_vec - first layer weights
% C, d_vec - second layer weights
% v - output layer weights
% sigma2 - noise variance (nusiance parameter)
% Lr,beta1,beta2 - ADAM hyperparameters
% Wd - regularization parameter
% amsgrad - improving ADAM convergence (option)
% tool - the minimum tool for the convergence error
% maxIter - The maximum number of iterations
%%
[p,N]=size(x);
K_1=length(b_vec);
K_2=length(v_vec);

hy_2=hy^2;


error=1e6;
epsilon=1e-8;
Iter=1;

if amsgrad
vt_A_max=zeros(K_1,p);
vt_b_max=zeros(K_1,1);
vt_C_max=zeros(K_2,K_1);
vt_d_max=zeros(K_2,1);

vt_v_max=zeros(K_2,1);

vt_sigma2_max=0;

end
vt_A=zeros(K_1,p);
vt_b=zeros(K_1,1);
vt_C=zeros(K_2,K_1);
vt_d=zeros(K_2,1);
vt_v=zeros(K_2,1);
vt_sigma2=0;

mt_A=zeros(K_1,p);
mt_b=zeros(K_1,1);
mt_C=zeros(K_2,K_1);
mt_d=zeros(K_2,1);
mt_v=zeros(K_2,1);
mt_sigma2=0;

% max_sigma2=3*mad(y,1);
while (Iter <= maxIter) && (error >= tool) 
    
    %% Calculating auxilarly parameters
    theta_old=[A(:);b_vec;C(:);d_vec;v_vec;sigma2];
    % theta_old=[A(:);b_vec;v_vec];

    sigma2_h=sigma2+hy_2;
    
    sigma2_inv=sigma2^(-1);
    sigma2_inv_h=sigma2_h^(-1);

   
    Act_Func1=Gelu_activation(x,A,b_vec); % Applying activation function layer 1
    Act_Func_deriv1=Gelu_first_deriv(x,A,b_vec);  % Activation function layer 1 derivative
    Act_Func2=Gelu_activation(Act_Func1,C,d_vec); % Applying activation function layer 2
    Act_Func_deriv2=Gelu_first_deriv(Act_Func1,C,d_vec); % Activation function layer 2 derivative
   
    Phi_reg=v_vec'*Act_Func2; %The reression model
    y_phi_sub=repmat(y.',1,N)-repmat(Phi_reg,N,1); %Substraction of y_m-phi(x_k) (diffrent indexes)
    y_phi_sub_pow2 =y_phi_sub.^2;
    exp_y_phi_h=exp(-y_phi_sub_pow2./(2*sigma2_h));


    %% Estimating regression parameters

    %Calculating Lambda weights

    t_h=exp_D_x.*exp_y_phi_h;
    s_h=sum(t_h)-diag(exp_y_phi_h).';
    sum_s_h=sum(s_h);
    gamma_h=t_h./sum_s_h;
    zeta_h=sum(gamma_h.*y_phi_sub)-diag(gamma_h.*y_phi_sub).';

    lambda_h=sigma2_inv.*w_z.*diag(y_phi_sub).'-sigma2_inv_h.*zeta_h;
    lambda_h(find(isnan(lambda_h)))=1e-50;

    %% SA (steepest ascent) for regression parameters
    
    % Calculating gradients of the loss
    Grad_A=(((-lambda_h.*(v_vec.*Act_Func_deriv2))'*C)'.*Act_Func_deriv1)*x.'+Wd.*A;
    Grad_b=sum(((-lambda_h.*(v_vec.*Act_Func_deriv2))'*C)'.*Act_Func_deriv1,2);
    Grad_C=((-lambda_h.*(v_vec.*Act_Func_deriv2)))*Act_Func1.'+Wd.*C;
    Grad_d=sum((-lambda_h.*(v_vec.*Act_Func_deriv2)),2);
    Grad_v=-Act_Func2*lambda_h'+Wd.*v_vec;
    
       % Updating first and second order moment
       mt_A=beta1.*mt_A+(1-beta1).*Grad_A;
       mt_b=beta1.*mt_b+(1-beta1).*Grad_b;
       mt_C=beta1.*mt_C+(1-beta1).*Grad_C;
       mt_d=beta1.*mt_d+(1-beta1).*Grad_d;
       mt_v=beta1.*mt_v+(1-beta1).*Grad_v;

       vt_A=beta2.*vt_A+(1-beta2).*(Grad_A.^2);
       vt_b=beta2.*vt_b+(1-beta2).*(Grad_b.^2);
       vt_C=beta2.*vt_C+(1-beta2).*(Grad_C.^2);
       vt_d=beta2.*vt_d+(1-beta2).*(Grad_d.^2);
       vt_v=beta2.*vt_v+(1-beta2).*(Grad_v.^2);

       % Correction of first and second order moment
       mt_A_c=mt_A./(1-(beta1^Iter));
       mt_b_c=mt_b./(1-(beta1^Iter));
       mt_C_c=mt_C./(1-(beta1^Iter));
       mt_d_c=mt_d./(1-(beta1^Iter));
       mt_v_c=mt_v./(1-(beta1^Iter));

       vt_A_c=vt_A./(1-(beta2^Iter));
       vt_b_c=vt_b./(1-(beta2^Iter));
       vt_C_c=vt_C./(1-(beta2^Iter));
       vt_d_c=vt_d./(1-(beta2^Iter));
       vt_v_c=vt_v./(1-(beta2^Iter));

      % Check amsgrad case
       if amsgrad
       vt_A_max=max(vt_A_c,vt_A_max);
       vt_b_max=max(vt_b_c,vt_b_max);
       vt_C_max=max(vt_C_c,vt_C_max);
       vt_d_max=max(vt_d_c,vt_d_max);
       vt_v_max=max(vt_v_c,vt_v_max);

       vt_A_c=vt_A_max;
       vt_b_c=vt_b_max;
       vt_C_c=vt_C_max;
       vt_d_c=vt_d_max;
       vt_v_c=vt_v_max;
       end

       % Update network parameters

       
       A=A-Lr.*mt_A_c./(sqrt(vt_A_c)+epsilon);
       b_vec=b_vec-Lr.*mt_b_c./(sqrt(vt_b_c)+epsilon);
       C=C-Lr.*mt_C_c./(sqrt(vt_C_c)+epsilon);
       d_vec=d_vec-Lr.*mt_d_c./(sqrt(vt_d_c)+epsilon);
       v_vec=v_vec-Lr.*mt_v_c./(sqrt(vt_v_c)+epsilon);
       
    %% Calculating noise variance

    % Gradient calc
    factor_h_sigma=hy_2*sigma2_inv*sigma2_inv_h;
    psi_h=sum(gamma_h.*y_phi_sub_pow2)-diag(gamma_h.*y_phi_sub_pow2).';
    Grad_J_sigma2=(sigma2_inv^2)*(w_z*diag(y_phi_sub_pow2))-(sigma2_inv_h^2)*sum(psi_h)...
                  -factor_h_sigma;
    Grad_Loss_sigma2=-0.5.*Grad_J_sigma2;  %+Wd.*sigma2;

     % Updating first and second order moment
       mt_sigma2=beta1.*mt_sigma2+(1-beta1).*Grad_Loss_sigma2;
       vt_sigma2=beta2.*vt_sigma2+(1-beta2).*(Grad_Loss_sigma2.^2);

     % Correction of first and second order moment
       mt_sigma2_c=mt_sigma2./(1-(beta1^Iter));
       vt_sigma2_c=vt_sigma2./(1-(beta2^Iter));

     % Check amsgrad case
       if amsgrad
       vt_sigma2_max=max(vt_sigma2_c,vt_sigma2_max);
       vt_sigma2_c=vt_sigma2_max;
       end

     % Update variance
     sigma2=sigma2-Lr.*mt_sigma2_c./(sqrt(vt_sigma2_c)+epsilon);

    %% Update the vector of parameters and calc error
    theta=[A(:);b_vec;C(:);d_vec;v_vec;sigma2];

    error=norm(theta-theta_old)/norm(theta_old);

    Iter=Iter+1;
end

end


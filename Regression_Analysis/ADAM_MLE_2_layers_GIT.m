function [A,b_vec,C,d_vec,v_vec] = ADAM_MLE_2_layers_GIT(x,y,A,b_vec,C,d_vec,v_vec,Lr,beta1,beta2,Wd,amsgrad,tool,maxIter)
%%% Inputs:
% x - input training data
% y - output training data
% A,b_vec,C,d_vec,v_vec - Network parameters
% Lr,beta1,beta2,Wd,amsgrad - Hypoerparameters for ADAM optimizer (Wd- weight decay) https://pytorch.org/docs/stable/generated/torch.optim.Adam.html
% tool,maxIter - stop criterion and maximum iterations
%%% Outputs:
% A,b_vec,C,d_vec,v_vec - Network parameters


[p,N]=size(x);
K_1=length(b_vec);
K_2=length(v_vec);

error=1e6;
epsilon=1e-8;
Iter=1;
if amsgrad
vt_A_max=zeros(K_1,p);
vt_b_max=zeros(K_1,1);
vt_C_max=zeros(K_2,K_1);
vt_d_max=zeros(K_2,1);

vt_v_max=zeros(K_2,1);


end
vt_A=zeros(K_1,p);
vt_b=zeros(K_1,1);
vt_C=zeros(K_2,K_1);
vt_d=zeros(K_2,1);
vt_v=zeros(K_2,1);

mt_A=zeros(K_1,p);
mt_b=zeros(K_1,1);
mt_C=zeros(K_2,K_1);
mt_d=zeros(K_2,1);
mt_v=zeros(K_2,1);

while (Iter <= maxIter) && (error >= tool) 
    
    %% Calculating auxilarly parameters

    %% Calculating auxilarly parameters

    theta_old=[A(:);b_vec;C(:);d_vec;v_vec];
    % theta_old=[A(:);b_vec;v_vec];

    Act_Func1=Gelu_activation(x,A,b_vec); % Applying activation function layer 1
    Act_Func_deriv1=Gelu_first_deriv(x,A,b_vec);  % Activation function layer 1 derivative
    Act_Func2=Gelu_activation(Act_Func1,C,d_vec); % Applying activation function layer 2
    Act_Func_deriv2=Gelu_first_deriv(Act_Func1,C,d_vec); % Activation function layer 2 derivative

    Phi_reg=v_vec'*Act_Func2; %The reression model
    y_phi_sub=y-Phi_reg; %Substraction of y_m-phi(x_k) (diffrent indexes)
    
    %% Computing Gradients

    lambda=(2/N).*y_phi_sub;
    
    %% SA (steepest ascent) for regression parameters

    % Calculating gradients of the loss
    Grad_A=(((-lambda.*(v_vec.*Act_Func_deriv2))'*C)'.*Act_Func_deriv1)*x.'+Wd.*A;
    Grad_b=sum(((-lambda.*(v_vec.*Act_Func_deriv2))'*C)'.*Act_Func_deriv1,2);
    Grad_C=((-lambda.*(v_vec.*Act_Func_deriv2)))*Act_Func1.'+Wd.*C;
    Grad_d=sum((-lambda.*(v_vec.*Act_Func_deriv2)),2);
    Grad_v=-Act_Func2*lambda'+Wd.*v_vec;
    
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


 
    theta=[A(:);b_vec;C(:);d_vec;v_vec];

    error=norm(theta-theta_old)/norm(theta_old);

    Iter=Iter+1;
end

end


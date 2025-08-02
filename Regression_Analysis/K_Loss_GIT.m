function    Lh = K_Loss_GIT(x,y,hy,w_z,exp_D_x,Phi_reg,sigma2)
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here
[~,N]=size(x);
hy_2=hy^2;


    %% Calculating auxilarly parameters

    % theta_old=[A(:);b_vec;v_vec];

    sigma2_h=sigma2+hy_2;
    
    sigma2_inv=sigma2^(-1);



    y_phi_sub=repmat(y.',1,N)-repmat(Phi_reg,N,1); %Substraction of y_m-phi(x_k) (diffrent indexes)

    y_phi_sub_pow2 =y_phi_sub.^2;
    exp_y_phi_h=exp(-y_phi_sub_pow2./(2*sigma2_h));
    
    t_h=exp_D_x.*exp_y_phi_h;
    s_h=sum(t_h)-diag(exp_y_phi_h).';
    U_h=sum(s_h);

    Lh=0.5.*sigma2_inv.*(w_z*diag(y_phi_sub_pow2))+log(U_h)+0.5*log(sigma2/sigma2_h);
    
    
end


function R_hat = EMP_MSE_KPME(w,y,exp_D,s_g,h,a,a_dot,a_dot_2,alpha,phase,Sigma,Sigma_inv,p,N,sigma_2)
%% Computing empirical asymptotic MSE matrix R_hat

tau_2 = 2*(h^2);
alpha_2=alpha^2;
Sigma_h=Sigma+tau_2.*eye(p);
Sigma_inv_h=Sigma_h^(-1);
y_n_h=(Sigma_inv_h^(0.5))*y;
norm_y_n_h=sum(y_n_h.*conj(y_n_h)).';
norm_a_n=real(a'*Sigma_inv*a);
eta=2.*real(a_dot'*real(Sigma_inv)*a);
Psi=2.*real(a_dot_2'*real(Sigma_inv)*a)+2.*real(a_dot'*real(Sigma_inv)*a_dot);

norm_a_n_h=real(a'*Sigma_inv_h*a);
rho=2.*real(a_dot'*real(Sigma_inv_h)*a);
Gamma=2.*real(a_dot_2'*real(Sigma_inv_h)*a)+2.*real(a_dot'*real(Sigma_inv_h)*a_dot);

e_n_y_h = exp(-norm_y_n_h);

temp_R = y'*Sigma_inv*(exp(1j.*phase).*a);

temp_R_dot = y'*Sigma_inv*(exp(1j.*phase).*a_dot);

R = alpha.*real(temp_R);

temp_R_h = y'*Sigma_inv_h*(exp(1j.*phase).*a);

temp_R_dot_h = y'*Sigma_inv_h*(exp(1j.*phase).*a_dot);

R_h = alpha.*real(temp_R_h);

h_vec = Sigma_inv*y*(w.*tanh(2.*R));

temp_h = exp(1j.*phase)*h_vec'*a;

temp_h_dot = exp(1j.*phase)*h_vec'*a_dot;

temp_h_dot_2 = exp(1j.*phase)*h_vec'*a_dot_2;

T_A(1,1) = 2*alpha*real(temp_h_dot_2)-alpha_2*Psi; T_A(1,2) = 2*real(temp_h_dot)-2*alpha*eta; T_A(1,3) = -2*alpha*imag(temp_h_dot); 

T_A(2,1) = T_A(1,2); T_A(3,1) = T_A(1,3);

T_A(2,2) = -2*norm_a_n; T_A(2,3) = -2*imag(temp_h); 

T_A(3,2) = T_A(2,3); T_A(3,3) = -2*alpha*real(temp_h); 


gs_1 = exp(2*R);  gs_2 = exp(-2*R);

lambda_1 = gs_1./(gs_1 + gs_2); lambda_2 = 1 - lambda_1;

gs_3 = exp(-(norm_y_n_h - 2*R_h));  gs_4 = exp(-(norm_y_n_h + 2*R_h));
    
lambda_3 = gs_3./sum(gs_3 + gs_4); lambda_4 = gs_4./sum(gs_3 + gs_4);

w_lambda_1 = w.*lambda_1;  w_lambda_2 = w.*lambda_2;


b_1 = ([2*alpha*real(temp_R_dot)-alpha_2*eta, (2*(real(temp_R) - alpha*norm_a_n)), (-2*alpha*imag(temp_R))]).';

b_2 =([-2*alpha*real(temp_R_dot)-alpha_2*eta, (-2*(real(temp_R) + alpha*norm_a_n)), (2*alpha*imag(temp_R))]).';

b_1_cov = ((b_1.*(repmat(w_lambda_1.',3,1)))*b_1.');

b_2_cov = ((b_2.*(repmat(w_lambda_2.',3,1)))*b_2.');

T_B = b_1_cov + b_2_cov;

v_1 = 2*alpha*tanh(2*R).*real(temp_R_dot)-alpha_2*eta;

v_2 = 2*tanh(2*R).*real(temp_R) - 2*alpha*norm_a_n;

v_3 = -2*alpha.*tanh(2*R).*imag(temp_R);


v = [v_1,v_2,v_3].';

T_C = ((v.*(repmat(w.',3,1)))*v.');

s_vec = Sigma_inv_h*y*((e_n_y_h.*sinh(2.*R_h))./(sum(e_n_y_h.*cosh(2.*R_h))));
 
temp_s = exp(1j.*phase)*s_vec'*a;

temp_s_dot = exp(1j.*phase)*s_vec'*a_dot;

temp_s_dot_2 = exp(1j.*phase)*s_vec'*a_dot_2;



g_2 = [2*alpha*real(temp_s_dot)-alpha_2*rho,(2*real(temp_s)-2*alpha*norm_a_n_h),(-2*alpha*imag(temp_s))].';

T_F = g_2*g_2.';

T_D(1,1) = 2*alpha*real(temp_s_dot_2)-alpha_2*Gamma; T_D(1,2) = 2*real(temp_s_dot)-2*alpha*rho; T_D(1,3) = -2*alpha*imag(temp_s_dot); 
T_D(2,1) = T_D(1,2); T_D(3,1) = T_D(1,3);
T_D(2,2) = -2*norm_a_n_h; T_D(2,3) = -2*imag(temp_s); 
T_D(3,2) = T_D(2,3); T_D(3,3) = -2*alpha*real(temp_s); 


b_3_1 = 2*alpha*real(temp_R_dot_h)-alpha_2*rho; b_3_2 = 2*real(temp_R_h) - 2*alpha*norm_a_n_h; b_3_3 = -2*alpha*imag(temp_R_h); 
b_3 = [b_3_1,b_3_2,b_3_3].';

b_4_1 = -2*alpha*real(temp_R_dot_h)-alpha_2*rho; b_4_2 = -2*real(temp_R_h) - 2*alpha*norm_a_n_h; b_4_3 = 2*alpha*imag(temp_R_h); 

b_4 = [b_4_1,b_4_2,b_4_3].';

b_3_cov = ((b_3.*(repmat(lambda_3.',3,1)))*b_3.');

b_4_cov = ((b_4.*(repmat(lambda_4.',3,1)))*b_4.');

T_E = b_3_cov + b_4_cov;

C = T_A + T_B - T_C - T_D - T_E + T_F;

psi_G_hat = (N-1).*w; %%%

z_hat = N.*(lambda_3.'.*(b_3 - g_2) + lambda_4.'.*(b_4 - g_2));

c_hat = v - g_2; %%%

d_hat = (N-1).*(c_hat*exp_D.' - c_hat)./s_g; %%%

v_hat = psi_G_hat.'.*c_hat + d_hat - z_hat;

D = (v_hat*v_hat.')./N;

R_hat = (C\D/C)./N;

end
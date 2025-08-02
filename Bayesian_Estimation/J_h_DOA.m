function J_h = J_h_DOA(theta,y,h,sum_w_cosh,Sigma,Sigma_inv,q)

% Comutiong J_h(theta) objective

phi=theta(1,:);
alpha=theta(2,:);
phase=theta(3,:);

tau_2 = 2*(h^2);
Sigma_tau=Sigma+tau_2.*eye(q);
Sigma_tau_inv=Sigma_tau^(-1);
y_normalized_tau=(Sigma_tau_inv^(0.5))*y;
n_y_normalized_tau=sum(y_normalized_tau.*conj(y_normalized_tau)).';
Sigma_inv_diff_sqrt=(Sigma_inv-Sigma_tau_inv)^(0.5);

e_y_tau = exp(-n_y_normalized_tau);


a = exp(-1i*pi.*(0:q-1).'*sin(phi))./sqrt(q);


n_a_normalized=sum((Sigma_inv_diff_sqrt*a).*conj(Sigma_inv_diff_sqrt*a));

y_a_tau=(y'*Sigma_tau_inv)*a;

R_tau = alpha.*real(y_a_tau.*(exp(1j.*phase)));
J_h = sum_w_cosh-log(e_y_tau'*cosh(2.*R_tau))-(alpha.^2).*n_a_normalized;
end



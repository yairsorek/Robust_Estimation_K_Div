function [w,exp_D,s_g] = calc_weight(D,h)

h_2 = h^2;
D = D./(2*h_2);
% y_R = real(y);
% y_I = imag(y);
% x = [y_R;y_I];
% D = pdist(x.'); 
% D = (abs(D).^2)./(2*h_2);
% D = squareform(D);
exp_D = exp(-D);
g = sum(exp_D,2) - 1;
s_g = sum(g);
w = g./s_g;
w(w==0)=1e-100;
w(find(isnan(w)))=1e-100;
function [wz,exp_Dx,exp_Dz,s_gz] = calc_weight(Dx,Dy,hx,hy,N)

hx_2 = hx^2;
Dx = Dx./(2*hx_2);
exp_Dx = exp(-Dx);
hy_2 = hy^2;
Dy = Dy./(2*hy_2);
exp_Dy = exp(-Dy);
exp_Dz=exp_Dx.*exp_Dy;
gz = (sum(exp_Dz) - 1)./N;
s_gz = sum(gz);
wz = gz./s_gz;

end
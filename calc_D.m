function D = calc_D(y)

y_R = real(y);
y_I = imag(y);
x = [y_R;y_I];
D = abs(pdist(x.')).^2; 
D = squareform(D);


function y = Generate_Obs(a,a_1,a_2,epsilon,p,gain,phase,Sigma_sqrt,sigma_int,N)

alphabet = [-1,1];
s = randsrc(1,N,alphabet);
s_1 = (1/sqrt(2)).*(randn(1,N)+1i*randn(1,N));
s_2 = (1/sqrt(2)).*(randn(1,N)+1i*randn(1,N));

alphabet = [0,1];
nu = 1-sqrt(1-epsilon);
prob = [1-nu,nu];
U_1 = randsrc(1,N,[alphabet; prob]);
U_2 = randsrc(1,N,[alphabet; prob]);

w = Sigma_sqrt*((1/sqrt(2)).*(randn(p,N)+1i*randn(p,N)));

y = gain*exp(1i*phase)*a*s + sigma_int*a_1*(s_1.*U_1) + sigma_int*a_2*(s_2.*U_2) + w;


function [x_c,y_c,Trg_func] = Generate_contaminated_data_GIT(a,mu_n,sigma2_n,b,mu_o_c,sigma2_o_c,outliers_inds,N)

%% Inputs

% c - The support of the nominal input uniform distribution [-a,a]

% mu_n, sigma2_n - noise mean and variance, respectively

% b - The support of the contminated input uniform distribution [-b,b]

% mu_o_c, sigma2_o_c - output outlies mean and variance, respectively

% outliers_inds - locations of outliers 

% N - sample size

%% Outputs

% x_c, y_c - contaminated inputs and outputs

% Trg_func - target function (replace y_c when we generate testing data)

%% Function

L_outliers=length(outliers_inds);

     
 % Generating nominal data
 x_c=-a+2*a.*rand(2,N);
 Trg_func=exp(-x_c(1,:).^2-(x_c(2,:)-0.75).^2)+exp(-2.*((x_c(1,:)-0.5).^2 + (x_c(2,:)+0.5).^2))+exp(-2.*(x_c(2,:).^2+(x_c(1,:)+0.75).^2));
 y_c=Trg_func+mvnrnd(mu_n,sigma2_n,N).';

 % Contaminating the data with outliers
 x_c(:,outliers_inds)=-b+2*b.*rand(2,L_outliers);
 y_c(:,outliers_inds)=mvnrnd(mu_o_c,sigma2_o_c,L_outliers).';

end


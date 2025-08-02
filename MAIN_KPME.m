%% DOA Bayesian estimation under intermittent jamming via the
%% K-posterior mean estimator (KPME)

%% Testing estimation performance versus contamination ratio

%% Dimension and Sample size

q = 12; % Number of antennas
N = 300;


%% Jammers

theta_i_1 = 20*pi/180; 
theta_i_2 = 15*pi/180; 

a_i_1 = exp(-1i*pi.*(0:q-1).'*sin(theta_i_1))./sqrt(q);
a_i_2 = exp(-1i*pi.*(0:q-1).'*sin(theta_i_2))./sqrt(q);

%% Range for selection of bandwidth parameter

K=37;
h_vec = linspace(1,10,K);

%% Contamination ratios
L = 13;
epsilon_vec = linspace(0,0.3,L);

%% Number of Monte-Carlo trials

U = 2e4;


%% Priors parameters
phiPriorMean_2=10*pi/180;
phiPriorMean=0;
phiPriorSigma2=(10*pi/180)^2; %0.1
gainPriorMean=1;
gainPriorSigma2=(0.1)^2; %0.1
phasePriorMean=0;
phasePriorSigma2=(5*pi/180)^2; %0.1

%% Importance distribution parameters
phiRefMean=0;
phiRefSigma2=(25*pi/180)^2; 
gainRefMean=1;
gainRefSigma2=(0.2)^2; %0.2
phaseRefMean=0;
phaseRefSigma2=(10*pi/180)^2;


RefParms=[phiRefMean phiRefSigma2; gainRefMean gainRefSigma2; phaseRefMean phaseRefSigma2];
PriorPars=[phiPriorMean phiPriorSigma2; gainPriorMean gainPriorSigma2;phasePriorMean phasePriorSigma2];

% Noise parameters
SNR = 4; % Default 4
M2_gain=gainPriorMean^2+gainPriorSigma2;
rho = sqrt(M2_gain)*sqrt(10.^(-SNR./10));
rho_2=rho^2;
sigma_2=rho_2;
Sigma_noise=zeros(q,q);

b=0.1+0.2i;
  for k=1:q
       for l=k:q
          Sigma_noise(k,l)=rho_2*(b^(abs(k-l))); 
          Sigma_noise(l,k)=conj(Sigma_noise(k,l));
       end
  end

Sigma_noise_sqrt=Sigma_noise^(0.5);
Sigma_noise_inv=Sigma_noise^(-1);
Sigma_noise_inv_sqrt=Sigma_noise_inv^(0.5);

%% Define signal-to-interference ratio (SIR) 
SIR = -16; %-16
sigma_int = sqrt(M2_gain)*sqrt(10.^(-SIR./10)); % Std of jammers


%% Sampling from the reference distribution

Ms=2e3; % Number of samples drawn from the reference distribution

phi_samples_target=  phiRefMean+sqrt(phiRefSigma2)*randn(1,Ms);
% % 
phase_samples_target= phaseRefMean+sqrt(phaseRefSigma2)*randn(1,Ms);

gain_samples_target= gainRefMean+sqrt(gainRefSigma2)*randn(1,Ms);
% % % 
theta_samples_ref=[phi_samples_target;gain_samples_target;phase_samples_target];

%% Steering vector of the reference samples
a_phi_ref= exp(-1i*pi.*(0:q-1).'*sin(theta_samples_ref(1,:)))./sqrt(q);


%% Initialization
error_KPME=zeros(U,L);
h_selected=zeros(U,L);
AMSE_DOA_selected=zeros(U,L);


for l = 1:L
    
    epsilon = epsilon_vec(l);
    tic
    parfor u=1:U
     
        %% Real parameters
        phi_s= phiPriorMean+sqrt(phiPriorSigma2)*randn;
        gain= gainPriorMean+sqrt(gainPriorSigma2)*randn;
        phase=phasePriorMean+sqrt(phasePriorSigma2)*randn;
    
        
        %% Creating data
        a = exp(-1i*pi.*(0:q-1).'*sin(phi_s))./sqrt(q);
        y = Generate_Obs(a,a_i_1,a_i_2,epsilon,q,gain,phase,Sigma_noise_sqrt,sigma_int,N);
        D = calc_D(y);
        n_y_normalized = sum((Sigma_noise_inv_sqrt*y).*conj(Sigma_noise_inv_sqrt*y)).';

    
    %% Proposed approcah - KPME
    % Initialization
    phi_h_k=zeros(1,K);
    AMSE_DOA=zeros(1,K);
    log_cosh=log(cosh(2.*theta_samples_ref(2,:).*real((y'*Sigma_noise_inv*a_phi_ref).*(exp(1j.*theta_samples_ref(3,:)))))); % Auxiliary parameter
    
    % Searching for optimal bandwidth which minimize the empirical
    % asymptotic MSE
    for k=1:K
    h_k=h_vec(k);
    [w,exp_D,s_g] = calc_weight(D,h_k);
    
    sum_w_cosh=w'*log_cosh;
    
    % KPME for specific h value
    theta_h_k=KPME_DOA(theta_samples_ref,Sigma_noise,Sigma_noise_inv,PriorPars,RefParms,y,h_k,sum_w_cosh,N,q);
    phi_h_k(k)=theta_h_k(1);
    gain_h_k=theta_h_k(2);
    phase_h_k=theta_h_k(3);
    
    
    
    % Computing AMSE
    a_phi = exp(-1i*pi.*(0:q-1).'*sin(phi_h_k(k)))./sqrt(q);
    a_dot_phi = -(0:q-1).'.*exp(-1i*pi.*(0:q-1).'*sin(phi_h_k(k)))*1i*pi*cos(phi_h_k(k))/sqrt(q);
    a_dot_2_phi = 1i*pi*(0:q-1).'.*exp(-1i*pi.*(0:q-1).'*sin(phi_h_k(k))).*(1i*pi*(0:q-1).'.*(cos(phi_h_k(k))^2) + sin(phi_h_k(k)))/sqrt(q);
    AMSE_matrix=EMP_MSE_KPME(w,y,exp_D,s_g,h_k,a_phi,a_dot_phi,a_dot_2_phi,gain_h_k,phase_h_k,Sigma_noise,Sigma_noise_inv,q,N,sigma_2);
     AMSE_DOA(k)= AMSE_matrix(1,1);
    
    
    end
    
    [~,min_ind]=min(AMSE_DOA);
    phi_hat_K=phi_h_k(min_ind); % The KPME of phi with the optimal bandwidth
    h_selected(u,l)=h_vec(min_ind);
    AMSE_DOA_selected(u,l)=AMSE_DOA(min_ind);

    error_KPME(u,l)=(phi_hat_K-phi_s)^2;
    end

    
end
RMSE_KPME = sqrt(mean(error_KPME))*180./pi;

h_selected_avg=mean(h_selected);

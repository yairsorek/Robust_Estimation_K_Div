function theta_est = KPME_IS_DOA(theta_samples_ref,Sigma,Sigma_inv,prior_par,ref_par,y,h,sum_w_h,N,q)
 
      %% KPME based on importance-sampling (IS)

      % Computing objective function values for each theta
      J_h = J_h_DOA(theta_samples_ref,y,h,sum_w_h,Sigma,Sigma_inv,q); 
      
      % Compting values of reference and prior probabilities
      ref_phi=mvnpdf(theta_samples_ref(1,:)',ref_par(1,1),ref_par(1,2)).';
      prior_phi=mvnpdf(theta_samples_ref(1,:)',prior_par(1,1),prior_par(1,2)).';
      ref_gain=mvnpdf(theta_samples_ref(2,:)',ref_par(2,1),ref_par(2,2)).';
      prior_gain=mvnpdf(theta_samples_ref(2,:)',prior_par(2,1),prior_par(2,2)).';
      ref_phase=mvnpdf(theta_samples_ref(3,:)',ref_par(3,1),ref_par(3,2)).';
      prior_phase=mvnpdf(theta_samples_ref(3,:)',prior_par(3,1),prior_par(3,2)).';

      % Computing weights for the KPME IS approximation
      weight= exp(N.*(J_h-max(J_h))).*(prior_phi./ref_phi).*(prior_gain./ref_gain).*(prior_phase./ref_phase);
      weight_normalized=weight./sum(weight);

      % Estimation result
      theta_est=weight_normalized*theta_samples_ref.';

end


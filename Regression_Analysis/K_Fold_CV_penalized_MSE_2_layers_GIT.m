function Wd = K_Fold_CV_penalized_MSE_2_layers_GIT(x,y,Wdvec,L_Wd,K_CV,A_init,b_vec_init,C_init,d_vec_init,v_vec_init,Lr,beta1,beta2,amsgrad,tool,MaxIter)

N=length(y);
N_test=floor(N/K_CV);
K_Fold_CV_error_Wd=zeros(1,L_Wd);


    %% Searching for Wd
    for l=1:L_Wd
        Wd_l=Wdvec(l);
            % Applying K-Fold CV method
            for jj=1:K_CV
                ind_test=((jj-1)*N_test+1):jj*N_test;
                ind_train=1:N;
                ind_train(ind_test)=[];
                
                % Training
                x_train_cv=x(:,ind_train);
                y_train_cv=y(ind_train);
                [A_MLE,b_vec_MLE,C_MLE,d_vec_MLE,v_vec_MLE] = ADAM_MLE_2_layers_GIT(x_train_cv,y_train_cv,A_init,b_vec_init,C_init,d_vec_init,v_vec_init,Lr,beta1,beta2,Wd_l,amsgrad,tool,MaxIter);
    
               
                % Testing
    
                x_test_cv=x(:,ind_test);
                y_test_cv=y(ind_test);
                Phi_reg=Func_approx_2_layers_GIT(x_test_cv,A_MLE,b_vec_MLE,C_MLE,d_vec_MLE,v_vec_MLE);
    
                %Calculating K-fold CV error
    
                K_Fold_CV_error_Wd(l)=K_Fold_CV_error_Wd(l)+norm(Phi_reg-y_test_cv)^2;
     
            end
     end

               [~,ind]=min(K_Fold_CV_error_Wd);
               Wd=Wdvec(ind);

end
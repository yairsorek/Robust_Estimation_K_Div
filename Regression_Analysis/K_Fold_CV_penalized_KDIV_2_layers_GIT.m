function [hx,hy,Wd,K_Fold_CV_error_hx] = K_Fold_CV_penalized_KDIV_2_layers_GIT(x,y,hvec,Wdvec,L_h,L_Wd,hx_0,hy_0,K_CV,A_init,b_vec_init,C_init,d_vec_init,v_vec_init,sigma2_init,Lr,beta1,beta2,amsgrad,tool,MaxIter,MaxIter2)

N=length(y);
N_test=N/K_CV;


Iter=1;
hy=hvec(1);
Wd=Wdvec(1);
tool_conv=1e-9;
error_conv=inf;

             
while (Iter <= MaxIter2)&&(error_conv > tool_conv)

if Iter>1
K_Fold_CV_error_prev=K_Fold_CV_error_current;
end

K_Fold_CV_error_hx=zeros(1,L_h);
K_Fold_CV_error_hy=zeros(1,L_h);

K_Fold_CV_error_Wd=zeros(1,L_Wd);
%% Searching for hx

    for l=1:L_h
        hx_l=hvec(l);
            % Applying K-Fold CV method
            for jj=1:K_CV
                ind_test=((jj-1)*N_test+1):jj*N_test;
                ind_train=1:N;
                ind_train(ind_test)=[];
                
                % Training
                x_train_cv=x(:,ind_train);
                y_train_cv=y(ind_train);
                D_x_train_cv = abs(pdist(x_train_cv.')).^2; 
                D_x_train_cv = squareform(D_x_train_cv);
                D_y_train_cv = abs(pdist(y_train_cv.')).^2; 
                D_y_train_cv = squareform(D_y_train_cv);
    
               [w_z_train,exp_D_x_train,~,~] = calc_weight(D_x_train_cv,D_y_train_cv,hx_l,hy,N);
               [A_PKRE,b_vec_PKRE,C_PKRE,d_vec_PKRE,v_vec_PKRE,sigma2_PKRE] = ADAM_PKRE_2_layers_GIT(x_train_cv,y_train_cv,hy,w_z_train,exp_D_x_train,A_init,b_vec_init,C_init,d_vec_init,v_vec_init,sigma2_init,Lr,beta1,beta2,Wd,amsgrad,tool,MaxIter);
    
               
                % Testing
                % For testing we use the loss Lh with hy_0 and hx_0
    
                x_test_cv=x(:,ind_test);
                y_test_cv=y(ind_test);
                D_x_test_cv = abs(pdist(x_test_cv.')).^2; 
                D_x_test_cv = squareform(D_x_test_cv);
                D_y_test_cv = abs(pdist(y_test_cv.')).^2; 
                D_y_test_cv = squareform(D_y_test_cv);
                [w_z_test,exp_D_x_test,~,~] = calc_weight(D_x_test_cv,D_y_test_cv,hx_0,hy_0,N);
                Phi_reg=Func_approx_2_layers_GIT(x_test_cv,A_PKRE,b_vec_PKRE,C_PKRE,d_vec_PKRE,v_vec_PKRE);
                L_K_h=K_Loss_new(x_test_cv,y_test_cv,hy_0,w_z_test,exp_D_x_test,Phi_reg,sigma2_PKRE);
    
                %Calculating K-fold CV error
    
                K_Fold_CV_error_hx(l)=K_Fold_CV_error_hx(l)+L_K_h;
     
            end
     end
               [~,ind]=min(K_Fold_CV_error_hx);
               hx=hvec(ind);

%% Searching for hy
    for l=1:L_h
        hy_l=hvec(l);
            % Applying K-Fold CV method
            for jj=1:K_CV
                ind_test=((jj-1)*N_test+1):jj*N_test;
                ind_train=1:N;
                ind_train(ind_test)=[];
                
                % Training
                x_train_cv=x(:,ind_train);
                y_train_cv=y(ind_train);
                D_x_train_cv = abs(pdist(x_train_cv.')).^2; 
                D_x_train_cv = squareform(D_x_train_cv);
                D_y_train_cv = abs(pdist(y_train_cv.')).^2; 
                D_y_train_cv = squareform(D_y_train_cv);
    
               [w_z_train,exp_D_x_train,~,~] = calc_weight(D_x_train_cv,D_y_train_cv,hx,hy_l,N);
               [A_PKRE,b_vec_PKRE,C_PKRE,d_vec_PKRE,v_vec_PKRE,sigma2_PKRE] = ADAM_PKRE_2_layers_GIT(x_train_cv,y_train_cv,hy_l,w_z_train,exp_D_x_train,A_init,b_vec_init,C_init,d_vec_init,v_vec_init,sigma2_init,Lr,beta1,beta2,Wd,amsgrad,tool,MaxIter);
    
               
                % Testing
    
                x_test_cv=x(:,ind_test);
                y_test_cv=y(ind_test);
                D_x_test_cv = abs(pdist(x_test_cv.')).^2; 
                D_x_test_cv = squareform(D_x_test_cv);
                D_y_test_cv = abs(pdist(y_test_cv.')).^2; 
                D_y_test_cv = squareform(D_y_test_cv);
                [w_z_test,exp_D_x_test,~,~] = calc_weight(D_x_test_cv,D_y_test_cv,hx_0,hy_0,N);
                Phi_reg=Func_approx_2_layers_GIT(x_test_cv,A_PKRE,b_vec_PKRE,C_PKRE,d_vec_PKRE,v_vec_PKRE);
                L_K_h=K_Loss_new(x_test_cv,y_test_cv,hy_0,w_z_test,exp_D_x_test,Phi_reg,sigma2_PKRE);
    
                %Calculating K-fold CV error
    
                K_Fold_CV_error_hy(l)=K_Fold_CV_error_hy(l)+L_K_h;
     
            end
     end

               [~,ind]=min(K_Fold_CV_error_hy);
               hy=hvec(ind);



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
                D_x_train_cv = abs(pdist(x_train_cv.')).^2; 
                D_x_train_cv = squareform(D_x_train_cv);
                D_y_train_cv = abs(pdist(y_train_cv.')).^2; 
                D_y_train_cv = squareform(D_y_train_cv);
    
               [w_z_train,exp_D_x_train,~,~] = calc_weight(D_x_train_cv,D_y_train_cv,hx,hy,N);
               [A_PKRE,b_vec_PKRE,C_PKRE,d_vec_PKRE,v_vec_PKRE,sigma2_PKRE] = ADAM_PKRE_2_layers_GIT(x_train_cv,y_train_cv,hy,w_z_train,exp_D_x_train,A_init,b_vec_init,C_init,d_vec_init,v_vec_init,sigma2_init,Lr,beta1,beta2,Wd_l,amsgrad,tool,MaxIter);
    
               
                % Testing    
                x_test_cv=x(:,ind_test);
                y_test_cv=y(ind_test);
                D_x_test_cv = abs(pdist(x_test_cv.')).^2; 
                D_x_test_cv = squareform(D_x_test_cv);
                D_y_test_cv = abs(pdist(y_test_cv.')).^2; 
                D_y_test_cv = squareform(D_y_test_cv);
                [w_z_test,exp_D_x_test,~,~] = calc_weight(D_x_test_cv,D_y_test_cv,hx_0,hy_0,N);
                Phi_reg=Func_approx_2_layers_GIT(x_test_cv,A_PKRE,b_vec_PKRE,C_PKRE,d_vec_PKRE,v_vec_PKRE);
                L_K_h=K_Loss_GIT(x_test_cv,y_test_cv,hy_0,w_z_test,exp_D_x_test,Phi_reg,sigma2_PKRE);
    
                %Calculating K-fold CV error
    
                K_Fold_CV_error_Wd(l)=K_Fold_CV_error_Wd(l)+L_K_h;
     
            end
     end

               [~,ind]=min(K_Fold_CV_error_Wd(1:L_Wd));
               Wd=Wdvec(ind);


             K_Fold_CV_error_current=K_Fold_CV_error_Wd(ind); % Setting the current K-Fold CV error
             
             if Iter>1
             error_conv=abs(K_Fold_CV_error_prev-K_Fold_CV_error_current);
             end
             Iter=Iter+1;
end
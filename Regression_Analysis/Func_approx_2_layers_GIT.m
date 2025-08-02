function func_approx = Func_approx_2_layers_GIT(x,A_inner,b_inner,A_outer,b_outer,v)


    Act_Func_inner=Gelu_activation(x,A_inner,b_inner); % Applying activation function for first layer
    Act_Func_outer=Gelu_activation(Act_Func_inner,A_outer,b_outer); % Applying activation function for second layer

    func_approx=v.'*Act_Func_outer;
end


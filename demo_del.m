% This code was implemented by Makoto Nakatsuji by updating original
% BPTF code provided at "http://www.cs.cmu.edu/~lxiong/bptf/bptf.html".
% Epsilon is epsilons used in Eq. (6) or (7).
for D=50:50:50
% If you do not have matlab parallel tool box, you can comment out the
% below line.
%matlabpool
Epsilon=0.9

            build;

            cd ./lib
            build;
            cd ..
                addpath ./lib

            load TTrN2234
            load TTeN2234
            TTrN2234 = spTensor(TTrN2234.subs, TTrN2234.vals, TTrN2234.size);
            TTeN2234 = spTensor(TTeN2234.subs, TTeN2234.vals, TTeN2234.size);
            CTr = TTrN2234.Reduce(1:2);
            CTe = TTeN2234.Reduce(1:2);

            pn = 50e-3;
            max_iter = 1000;
            learn_rate = 1e-3;
            n_sample = 500;


            data.sp = length(TTrN2234.vals)/( TTrN2234.size(1)* TTrN2234.size(2)*TTrN2234.size(3));
            data.sp = data.sp*1000000;
            data.TTr = length(TTrN2234.vals);
            data.TTe = length(TTeN2234.vals);
            data.size = TTrN2234.size


            [U, V, dummy, r_pmf] = PMF_Grad(CTr, CTe, D, ...
                                            struct('ridge',pn,'learn_rate',learn_rate,'range',[],'max_iter',max_iter));

	    	    	    
            alpha = 2;
            
	    
	    [Us_bptf Vs_bptf Ts_bptf] = BPTF(TTrN2234, TTeN2234, D, struct('Walpha',alpha, 'nuAlpha',1), ...
				 {U,V,ones(D,TTrN2234.size(3))}, struct('max_iter',n_sample,'n_sample',n_sample,'save_sample',false,'run_name','alpha2-1'));
			    [Y_bptf] = BPTF_Predict(Us_bptf,Vs_bptf,Ts_bptf,D,TTeN2234,[]);
			     Y_bptf_N2234 = Y_bptf.vals;
			     r_bptf = RMSE(Y_bptf.vals-TTeN2234.vals);
			     fprintf('BPTF: %.4f\n', r_bptf)	    
	    	    

	    
	    [Us_bptf Vs_bptf Ts_bptf VORGs_bptf TORGs_bptf] = SSTF(TTrN2234, TTeN2234, D,Epsilon,struct('Walpha',alpha, 'nuAlpha',1), ...
                                             {U,V,ones(D,TTrN2234.size(3))}, struct('max_iter',n_sample,'n_sample',n_sample,'save_sample',false,'run_name','alpha2-1'));
            [Y_bptf] = SSTF_Predict(Us_bptf,Vs_bptf,Ts_bptf,VORGs_bptf,TORGs_bptf,D,TTeN2234,[]);
            Y_bptf_N2234 = Y_bptf.vals;

            r_sstf = RMSE(Y_bptf.vals-TTeN2234.vals);
            fprintf('SSTF: %.4f\n', r_sstf)
            r = [r_bptf r_sstf];
	    fprintf('BPTF SSTF:%.4f %.4f\n', r_bptf,r_sstf);

            clearvars -except Epsilon D

%matlabpool close
end

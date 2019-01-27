% This code was implemented by anonymous authors by updating original
% BPTF code provided at "http://www.cs.cmu.edu/~lxiong/bptf/bptf.html".
% Epsilon is epsilons used in Eq. (6) or (7).
% We set delta1=0.95, delta2=0.9, delta3=0.85, delta4=0.8 in Eq. (2)
% and Eq. (4). We set X=4.
function [Us Vs Ts VORGs TORGs rmseTe alpha] = SSTF(TTr, TTe, D, Epsilon, ...
                                        hyper_params, init, options)
%[Us Vs Ts] = SSTF(TTr, TTe, D, hyper_params, init, max_iter, n_sample, run_name)
% full bayesian probabilistic tensor factorization
% TTr: training tensor
% TTe: testing tensor
% D: feature dimension
% hyper_params: hyper parameters: (nuAlpha, Walpha, mu, muT, nu, beta, W, WT)
%       refer to the report for details
%       if nuAlpha=0, then the value of alpha is fixed at the value of Walpha
% init: initalization value: (U, V, T). if a scalar, then
% the sampling will start from the existing sample. The sample
% number will follow the given sample, which will overwrite the
% original subsequent ones.
% max_iter: maximum number of samples to draw
% n_sample: # samples used to calculate the result. if <= 0, then
% samples will be output to files
% run_name: the name of this run. used for sample files.
% Us, Vs, Ts: factor samples. each column is a sample. need reshape
% before use.
% File "clsPclsMap.dat" describes item and item's class id.
% File "clsPclsMap2.dat" describes tag and tag's class id.

% itemId-clsId relationship

if nargin < 6; options = []; end
[max_iter save_sample n_sample, run_name] = GetOptions(options,...
                                                  'max_iter',50,'save_sample',false,'n_sample',200,'run_name','');

if save_sample
    sample_file_format = GetSampleFileFormat('t');
end


[M N K] = size(TTr);
Xb = load('clsPclsMap.dat');
Xb2 = [Xb(:,2),Xb(:,1)];
J=max(Xb(:,2));
JJ=max(Xb2(:,2));
CLS_bag = GroupIndex(Xb2(:, 2), JJ);


Xbb = load('clsPclsMap2.dat');
Xbb2 = [Xbb(:,2),Xbb(:,1)];
J2=max(Xbb(:,2));
JJ2=max(Xbb2(:,2));
CLS_bag2 = GroupIndex(Xbb2(:, 2), JJ2);

subsMatrix = load('cube0-1ORG.dat');
lens=size(subsMatrix(:,2));
IT=zeros(1,N);
totalFreq=0;
for i=1:lens(1,1)
    IT(subsMatrix(i,2))= IT(subsMatrix(i,2))+1;
    totalFreq=totalFreq+1;
end
IT_NUM=IT;
IT=IT./totalFreq;

subsMatrix2 = load('cube0-1ORG.dat');
lens2=size(subsMatrix2(:,3));
ITT=zeros(1,K);
totalFreq=0;
for i=1:lens2(1,1)
    ITT(subsMatrix2(i,3))= ITT(subsMatrix2(i,3))+1;
    totalFreq=totalFreq+1;
end
ITT_NUM=ITT;
ITT=ITT./totalFreq;


[nuAlpha Walpha mu0 mu0T nu0 beta0 W0 W0T] = GetOptions(hyper_params,...
                                                  'nuAlpha',1, 'Walpha',1, 'mu',0, 'muT',1, 'nu',D, 'beta',1, 'W',eye(D), 'WT', eye(D));
iWalpha = inv(Walpha);
iW0 = inv(W0);
iW0T = inv(W0T);

%tensor data
subs = TTr.subs;
assert(all(min(subs) >= 1));
%assert(all(max(subs) <= [M N K]));
vals = TTr.vals;
L = length(vals);
clear TTr;

fprintf('Bayesian PTF for tensor (%d, %d, %d):%d. D = %d.\n', M, N, K, L, D);
fprintf('nu_alpha=%g, W_alpha=%g, mu=%g, mu_T=%g, nu=%g, beta=%g, W=%g, W_T=%g\n', ...
        nuAlpha, Walpha, mu0(1), mu0T(1), nu0, beta0, W0(1,1), W0T(1,1));

fprintf('Initialization...');
sample_idx = 1;
if isempty(init)
    U = randn(D, M)*0.1;
    V = randn(D, N)*0.1;
    T = randn(D, K)*0.1;

elseif iscell(init)
        
    [U V T] = cell2vars(init);clear init;
    assert(size(U,1) == D);
else

    assert(save_sample);
    fprintf('Continue from sample %d...', init);U=[];V=[];T=[];C=[];
    load(sprintf(sample_file_format, run_name, D, init));
    sample_idx = init + 1;
    
    assert(n_sample<=0, 'can only generate sample when start from samples');
    assert(all(size(U)==[D M]));
    assert(all(size(V)==[D N]));
    assert(all(size(T)==[D K]));
end


te = ~isempty(TTe);

if te
    subsTe = TTe.subs;
    valsTe = TTe.vals;
    LTe = length(valsTe);
    clear TTe;
end



rmseTe = nan;
yTr = PTF_Reconstruct(subs, U, V, T);
rmseTr = RMSE(yTr - vals);
if te
    yTe = PTF_Reconstruct(subsTe, U, V, T);
    rmseTe = RMSE(yTe - valsTe);
end



fprintf('complete. RMSE = %0.4f/%0.4f.\n', rmseTr, rmseTe);

%sample buffer
if n_sample > 0
    ysTr = zeros(L, n_sample);
    if te
        ysTe = zeros(LTe, n_sample);
    end
    Us = zeros(D*M, n_sample);
    Vs = zeros(D*N, n_sample);
    Ts = zeros(D*K, n_sample);
    VORGs = zeros(D*N, n_sample);
    TORGs = zeros(D*K, n_sample);
end

fprintf('Pre-calculating the index...');
subU = subs(:, 1);
subV = subs(:, 2);
subT = subs(:, 3);
fprintf('U');subU_bag = GroupIndex(subU, M);
fprintf('V');subV_bag = GroupIndex(subV, N);
fprintf('T');subT_bag = GroupIndex(subT, K);
fprintf('C');

subVU_bag = cell(1, M);subTU_bag = cell(1, M);valsU_bag = cell(1, M);
for ind = 1:M
    subVU_bag{ind} = subV(subU_bag{ind});
    subTU_bag{ind} = subT(subU_bag{ind});
    valsU_bag{ind} = vals(subU_bag{ind});
end
subUV_bag = cell(1, N);subTV_bag = cell(1, N);valsV_bag = cell(1, N);
for ind = 1:N
    subUV_bag{ind} = subU(subV_bag{ind});
    subTV_bag{ind} = subT(subV_bag{ind});
    valsV_bag{ind} = vals(subV_bag{ind});
end
subUT_bag = cell(1, K);subVT_bag = cell(1, K);valsT_bag = cell(1, K);
for ind = 1:K
    subUT_bag{ind} = subU(subT_bag{ind});
    subVT_bag{ind} = subV(subT_bag{ind});
    valsT_bag{ind} = vals(subT_bag{ind});
end


subXb2_bag = cell(1, JJ);
for ind = 1:JJ
    subXb2_bag{ind} = Xb(ind,2);
end

subXbb2_bag = cell(1, JJ2);
for ind = 1:JJ2
    subXbb2_bag{ind} = Xbb(ind,2);
end

%%%%%%%%%% From here, code prepares Matlab objects for items (N) for CV4 (CV4 is the fourth most sparse items) %%%%%%%%%%%

total=0;
for i = 1:N
    if (0 < IT_NUM(1,i))
        FREQ(i,2)=IT_NUM(1,i)/sum(IT_NUM);
    else
        FREQ(i,2)=0;
    end
    total=total+FREQ(i,2);
end

SORT1=sort(FREQ(:,2),'descend');
total2=0;
THRESHOLD7=0;
for i = 1:N
  total2=total2+SORT1(i,1);
  FRAC2=total2/total;
  if FRAC2>0.8
    THRESHOLD7=SORT1(i,1);
    break;    
  end
end


clsset=[];
counter7=0;
for ind = 1:N
  if THRESHOLD7>=FREQ(ind,2)  &&   FREQ(ind,2) ~= 0
    CLSS=(Xb2(CLS_bag{ind}));
    for snd = 1:size(Xb2(CLS_bag{ind}))
      classC=CLSS(snd,1);
      if classC ~= 0
      if find(clsset==classC);
      else
	counter7=counter7+1;	
	subXb10_bag{classC} = counter7;
	clsset=[clsset classC];
      end
      end
    end
  end        
  counter7=counter7+1;
  subXb11_bag{ind} = counter7;       
end


C4 = randn(D, counter7)*0.1;

subUC4_bag = cell(1, counter7);subTC4_bag = cell(1, counter7);valsC4_bag = cell(1, counter7);
for ind = 1:N
  if THRESHOLD7>=FREQ(ind,2)  &&   FREQ(ind,2) ~= 0
    CLSS=(Xb2(CLS_bag{ind}));
    for snd = 1:size(Xb2(CLS_bag{ind}))
      classC=CLSS(snd,1);
      if classC ~= 0
      subUC4_bag{subXb10_bag{classC}} = subU(subV_bag{ind});
      subTC4_bag{subXb10_bag{classC}} = subT(subV_bag{ind});
      valsC4_bag{subXb10_bag{classC}} = vals(subV_bag{ind});
      end
    end
  end
  subUC4_bag{subXb11_bag{ind}} = subU(subV_bag{ind});
  subTC4_bag{subXb11_bag{ind}} = subT(subV_bag{ind});
  valsC4_bag{subXb11_bag{ind}} = vals(subV_bag{ind});   
end

%%%%%%%%%% From here, code prepares Matlab objects for items (N) for CV3 (CV3 is the third most sparse items) %%%%%%%%%%%

total2=0;
THRESHOLD5=0;
for i = 1:N
  total2=total2+SORT1(i,1);
  FRAC2=total2/total;
  if FRAC2>0.85
    THRESHOLD5=SORT1(i,1);
    break;    
  end
end


clsset=[];
counter5=0;
for ind = 1:N
  if THRESHOLD5>=FREQ(ind,2)  &&   FREQ(ind,2) ~= 0
    CLSS=(Xb2(CLS_bag{ind}));
    for snd = 1:size(Xb2(CLS_bag{ind}))
      classC=CLSS(snd,1);
      if classC ~= 0
      if find(clsset==classC);
      else
	counter5=counter5+1;	
	subXb8_bag{classC} = counter5;
	clsset=[clsset classC];
      end
      end
    end
  end        
  counter5=counter5+1;
  subXb9_bag{ind} = counter5;       
end


C3 = randn(D, counter5)*0.1;

subUC3_bag = cell(1, counter5);subTC3_bag = cell(1, counter5);valsC3_bag = cell(1, counter5);
for ind = 1:N
  if THRESHOLD5>=FREQ(ind,2)  &&   FREQ(ind,2) ~= 0
    CLSS=(Xb2(CLS_bag{ind}));
    for snd = 1:size(Xb2(CLS_bag{ind}))
      classC=CLSS(snd,1);
      if classC ~= 0
      subUC3_bag{subXb8_bag{classC}} = subU(subV_bag{ind});
      subTC3_bag{subXb8_bag{classC}} = subT(subV_bag{ind});
      valsC3_bag{subXb8_bag{classC}} = vals(subV_bag{ind});
      end
    end
  end
  subUC3_bag{subXb9_bag{ind}} = subU(subV_bag{ind});
  subTC3_bag{subXb9_bag{ind}} = subT(subV_bag{ind});
  valsC3_bag{subXb9_bag{ind}} = vals(subV_bag{ind});   
end

%%%%%%%%%% From here, code prepares Matlab objects for items (N) for CV (CV is the second most sparse items) %%%%%%%%%%%
    

total2=0;
THRESHOLD3=0;
for i = 1:N
  total2=total2+SORT1(i,1);
  FRAC2=total2/total;
  if FRAC2>0.9
    THRESHOLD3=SORT1(i,1);
    break;    
  end
end


clsset=[];
counter1=0;
for ind = 1:N
  if THRESHOLD3>=FREQ(ind,2)  &&   FREQ(ind,2) ~= 0
    CLSS=(Xb2(CLS_bag{ind}));
    for snd = 1:size(Xb2(CLS_bag{ind}))
      classC=CLSS(snd,1);
      if classC ~= 0
      if find(clsset==classC);
      else
	counter1=counter1+1;
	
	subXb4_bag{classC} = counter1;
	clsset=[clsset classC];
      end
      end
    end
  end        
  counter1=counter1+1;
  subXb5_bag{ind} = counter1;       
end


C = randn(D, counter1)*0.1;

subUC_bag = cell(1, counter1);subTC_bag = cell(1, counter1);valsC_bag = cell(1, counter1);
for ind = 1:N
  if THRESHOLD3>=FREQ(ind,2)  &&   FREQ(ind,2) ~= 0
    CLSS=(Xb2(CLS_bag{ind}));
    for snd = 1:size(Xb2(CLS_bag{ind}))
      classC=CLSS(snd,1);
      if classC ~= 0
      subUC_bag{subXb4_bag{classC}} = subU(subV_bag{ind});
      subTC_bag{subXb4_bag{classC}} = subT(subV_bag{ind});
      valsC_bag{subXb4_bag{classC}} = vals(subV_bag{ind});
      end
    end
  end
  subUC_bag{subXb5_bag{ind}} = subU(subV_bag{ind});
  subTC_bag{subXb5_bag{ind}} = subT(subV_bag{ind});
  valsC_bag{subXb5_bag{ind}} = vals(subV_bag{ind});   
end


%%%%%%%%%% From here, code prepares Matlab objects for items (N) for CV2 (CV2 is the most sparse items) %%%%%%%%%%%


total2=0;
THRESHOLD=0;
for i = 1:N
    total2=total2+SORT1(i,1);
    FRAC2=total2/total;
    if FRAC2>0.95
        THRESHOLD=SORT1(i,1);
        break;
    end
end


clsset=[];
counter3=0;
for ind = 1:N
  if THRESHOLD>=FREQ(ind,2)  &&   FREQ(ind,2) ~= 0
    CLSS=(Xb2(CLS_bag{ind}));
    for snd = 1:size(Xb2(CLS_bag{ind}))
      classC=CLSS(snd,1);
      if classC ~= 0
      if find(clsset==classC);
      else
	counter3=counter3+1;
	
	subXb6_bag{classC} = counter3;
	clsset=[clsset classC];
      end
      end
    end
  end        
  counter3=counter3+1;
  subXb7_bag{ind} = counter3;       
end


C2 = randn(D, counter3)*0.1;

subUC2_bag = cell(1, counter3);subTC2_bag = cell(1, counter3);valsC2_bag = cell(1, counter3);
for ind = 1:N
  if THRESHOLD>=FREQ(ind,2)  &&   FREQ(ind,2) ~= 0
    CLSS=(Xb2(CLS_bag{ind}));
    for snd = 1:size(Xb2(CLS_bag{ind}))
      classC=CLSS(snd,1);
      if classC ~= 0
      subUC2_bag{subXb6_bag{classC}} = subU(subV_bag{ind});
      subTC2_bag{subXb6_bag{classC}} = subT(subV_bag{ind});
      valsC2_bag{subXb6_bag{classC}} = vals(subV_bag{ind});
      end
    end
  end
  subUC2_bag{subXb7_bag{ind}} = subU(subV_bag{ind});
  subTC2_bag{subXb7_bag{ind}} = subT(subV_bag{ind});
  valsC2_bag{subXb7_bag{ind}} = vals(subV_bag{ind});   
end


%%%%%%%%%% From here, code prepares Matlab objects for tags (K) for CT4 (CT4 is the third most sparse tags) %%%%%%%%%%%

total=0;
for i = 1:K
    if (0 < ITT_NUM(1,i))
        FREQ2(i,2)=ITT_NUM(1,i)/sum(ITT_NUM);
    else
        FREQ2(i,2)=0;
    end
    total=total+FREQ2(i,2);
end

SORT=sort(FREQ2(:,2),'descend');


total2=0;
THRESHOLD8=0;
for i = 1:K
    total2=total2+SORT(i,1);
    FRAC2=total2/total;
    if FRAC2>0.8
        THRESHOLD8=SORT(i,1);
       break
    end
end

clsset=[];
counter8=0;
for ind = 1:K
  if THRESHOLD8>=FREQ2(ind,2)  &&   FREQ2(ind,2) ~= 0
    CLSS=(Xbb2(CLS_bag2{ind}));
    for snd = 1:size(Xbb2(CLS_bag2{ind}))
      classC=CLSS(snd,1);  
      if classC ~= 0
      if find(clsset==classC);
      else
	counter8=counter8+1;
	
	subXbb10_bag{classC} = counter8;
	clsset=[clsset classC];
      end
      end
    end
  end
  counter8=counter8+1;
  subXbb11_bag{ind} = counter8;   
end


CT4 = randn(D, counter8)*0.1;

subUCT4_bag = cell(1, counter8);subTCT4_bag = cell(1, counter8);valsCT4_bag = cell(1, counter8);
for ind = 1:K
    if THRESHOLD8>=FREQ2(ind,2)  &&   FREQ2(ind,2) ~= 0     
        CLSS=(Xbb2(CLS_bag2{ind}));
        for snd = 1:size(Xbb2(CLS_bag2{ind}))
            classC=CLSS(snd,1);
	    if classC ~= 0
            subUCT4_bag{subXbb10_bag{classC}} = subU(subT_bag{ind});
            subTCT4_bag{subXbb10_bag{classC}} = subV(subT_bag{ind});
            valsCT4_bag{subXbb10_bag{classC}} = vals(subT_bag{ind});
	    end
        end        
    end
    subUCT4_bag{subXbb11_bag{ind}} = subU(subT_bag{ind});
    subTCT4_bag{subXbb11_bag{ind}} = subV(subT_bag{ind});
    valsCT4_bag{subXbb11_bag{ind}} = vals(subT_bag{ind});
end



%%%%%%%%%% From here, code prepares Matlab objects for tags (K) for CT3 (CT3 is the third most sparse tags) %%%%%%%%%%%

total2=0;
THRESHOLD6=0;
for i = 1:K
    total2=total2+SORT(i,1);
    FRAC2=total2/total;
    if FRAC2>0.85
        THRESHOLD6=SORT(i,1);
       break
    end
end

clsset=[];
counter6=0;
for ind = 1:K
  if THRESHOLD6>=FREQ2(ind,2)  &&   FREQ2(ind,2) ~= 0
    CLSS=(Xbb2(CLS_bag2{ind}));
    for snd = 1:size(Xbb2(CLS_bag2{ind}))
      classC=CLSS(snd,1);  
      if classC ~= 0
      if find(clsset==classC);
      else
	counter6=counter6+1;
	
	subXbb8_bag{classC} = counter6;
	clsset=[clsset classC];
      end
      end
    end
  end
  counter6=counter6+1;
  subXbb9_bag{ind} = counter6;   
end


CT3 = randn(D, counter6)*0.1;

subUCT3_bag = cell(1, counter6);subTCT3_bag = cell(1, counter6);valsCT3_bag = cell(1, counter6);
for ind = 1:K
    if THRESHOLD6>=FREQ2(ind,2)  &&   FREQ2(ind,2) ~= 0     
        CLSS=(Xbb2(CLS_bag2{ind}));
        for snd = 1:size(Xbb2(CLS_bag2{ind}))
            classC=CLSS(snd,1);
	    if classC ~= 0
            subUCT3_bag{subXbb8_bag{classC}} = subU(subT_bag{ind});
            subTCT3_bag{subXbb8_bag{classC}} = subV(subT_bag{ind});
            valsCT3_bag{subXbb8_bag{classC}} = vals(subT_bag{ind});
	    end
        end        
    end
    subUCT3_bag{subXbb9_bag{ind}} = subU(subT_bag{ind});
    subTCT3_bag{subXbb9_bag{ind}} = subV(subT_bag{ind});
    valsCT3_bag{subXbb9_bag{ind}} = vals(subT_bag{ind});
end



%%%%%%%%%% From here, code describes for tags (K) for CT (CT is the second most sparse tags) %%%%%%%%%%%
total2=0;
THRESHOLD4=0;
for i = 1:K
    total2=total2+SORT(i,1);
    FRAC2=total2/total;
    if FRAC2>0.9
        THRESHOLD4=SORT(i,1);
       break
    end
end

clsset=[];
counter2=0;
for ind = 1:K
  if THRESHOLD4>=FREQ2(ind,2)  &&   FREQ2(ind,2) ~= 0
    CLSS=(Xbb2(CLS_bag2{ind}));
    for snd = 1:size(Xbb2(CLS_bag2{ind}))
      classC=CLSS(snd,1);  
      if classC ~= 0
      if find(clsset==classC);
      else
	counter2=counter2+1;
                
	subXbb4_bag{classC} = counter2;
	clsset=[clsset classC];
      end
      end
    end
  end
  counter2=counter2+1;
  subXbb5_bag{ind} = counter2;   
end


CT = randn(D, counter2)*0.1;

subUCT_bag = cell(1, counter2);subTCT_bag = cell(1, counter2);valsCT_bag = cell(1, counter2);
for ind = 1:K
    if THRESHOLD4>=FREQ2(ind,2)  &&   FREQ2(ind,2) ~= 0     
        CLSS=(Xbb2(CLS_bag2{ind}));
        for snd = 1:size(Xbb2(CLS_bag2{ind}))
            classC=CLSS(snd,1);
	    if classC ~= 0
            subUCT_bag{subXbb4_bag{classC}} = subU(subT_bag{ind});
            subTCT_bag{subXbb4_bag{classC}} = subV(subT_bag{ind});
            valsCT_bag{subXbb4_bag{classC}} = vals(subT_bag{ind});
	    end
        end        
    end
    subUCT_bag{subXbb5_bag{ind}} = subU(subT_bag{ind});
    subTCT_bag{subXbb5_bag{ind}} = subV(subT_bag{ind});
    valsCT_bag{subXbb5_bag{ind}} = vals(subT_bag{ind});
end


%%%%%%%%%% From here, code prepares Matlab objects for CT2 (CT2 is the most sparse tag set) %%%%%%%%%%%

total2=0;
THRESHOLD2=0;
for i = 1:K
    total2=total2+SORT(i,1);
    FRAC2=total2/total;
    if FRAC2>0.95
        THRESHOLD2=SORT(i,1);
        break;
    end
end



clsset=[];
counter4=0;
for ind = 1:K
  if THRESHOLD2>=FREQ2(ind,2)  &&   FREQ2(ind,2) ~= 0
    CLSS=(Xbb2(CLS_bag2{ind}));
    for snd = 1:size(Xbb2(CLS_bag2{ind}))
      classC=CLSS(snd,1);   
      if classC ~= 0
      if find(clsset==classC);
      else
	counter4=counter4+1;
                
	subXbb6_bag{classC} = counter4;
	clsset=[clsset classC];
      end
      end
    end
  end
  counter4=counter4+1;
  subXbb7_bag{ind} = counter4;   
end


CT2 = randn(D, counter4)*0.1;

subUCT2_bag = cell(1, counter4);subTCT2_bag = cell(1, counter4);valsCT2_bag = cell(1, counter4);
for ind = 1:K
    if THRESHOLD2>=FREQ2(ind,2)  &&   FREQ2(ind,2) ~= 0     
        CLSS=(Xbb2(CLS_bag2{ind}));
        for snd = 1:size(Xbb2(CLS_bag2{ind}))
            classC=CLSS(snd,1);
	    if classC ~= 0
            subUCT2_bag{subXbb6_bag{classC}} = subU(subT_bag{ind});
            subTCT2_bag{subXbb6_bag{classC}} = subV(subT_bag{ind});
            valsCT2_bag{subXbb6_bag{classC}} = vals(subT_bag{ind});
	    end
        end        
    end
    subUCT2_bag{subXbb7_bag{ind}} = subU(subT_bag{ind});
    subTCT2_bag{subXbb7_bag{ind}} = subV(subT_bag{ind});
    valsCT2_bag{subXbb7_bag{ind}} = vals(subT_bag{ind});
end


clear subU subV subT subU_bag subV_bag subT_bag
fprintf('. complete.\n');




alpha = Walpha;%precision for users

%%%%%%%%  MCMC procedure of SSTF %%%%%
for iter = sample_idx:(sample_idx + max_iter - 1)
    fprintf('-Iter%d... ', iter);
    
%%%%%%%%%% From here, code samples hyper parameters %%%%%%%%%%%

%%%% Equation (9) in the paper.    
    if nuAlpha < 0
        if iter > -nuAlpha
            nuAlpha = 1;
        end
    end
    if nuAlpha > 0
        nualpha_ = nuAlpha + L;
        iWalpha_ = iWalpha + sum((vals - yTr).^2);
        alpha = wishrnd(1./iWalpha_, nualpha_);
    end
    
    %sample the prior of U
    Umean = mean(U, 2);
    beta0_ = beta0 + M;
    mu0_ = (beta0*mu0 + M*Umean)/beta0_;
    nu0_ = nu0 + M;
    dMu = mu0 - Umean;
    iW0_ = iW0 + U*U' - M*(Umean*Umean') + (beta0*M/beta0_)*(dMu*dMu');
    W0_ = inv(iW0_);
    A_U = wishrnd((W0_ + W0_')*0.5, nu0_);
    mu_U = mvnrndpre(mu0_, beta0_*A_U);

%%%% Equation (10) in the paper.       
    %sample the prior of V
    Vmean = mean(V, 2);
    beta0_ = beta0 + N;
    mu0_ = (beta0*mu0 + N*Vmean)/beta0_;
    nu0_ = nu0 + N;
    dMu = mu0 - Vmean;
    iW0_ = iW0 + V*V' - N*(Vmean*Vmean') + (beta0*N/beta0_)*(dMu*dMu');
    W0_ = inv(iW0_);
    A_V = wishrnd((W0_ + W0_')*0.5, nu0_);
    mu_V = mvnrndpre(mu0_, beta0_*A_V);
    
    %sample the prior of T
    Tmean = mean(T, 2);
    beta0_ = beta0 + K;
    mu0_ = (beta0*mu0 + K*Tmean)/beta0_;
    nu0_ = nu0 + K;
    dMu = mu0 - Tmean;
    iW0_ = iW0 + T*T' - K*(Tmean*Tmean') + (beta0*K/beta0_)*(dMu*dMu');
    W0_ = inv(iW0_);
    A_T = wishrnd((W0_ + W0_')*0.5, nu0_);
    mu_T = mvnrndpre(mu0_, beta0_*A_T);

%%%% Equation (11) in the paper.       
    %sample the prior of CV (i=2).   
    Cmean = mean(C,2);
    beta0_ = beta0 + counter1;
    mu0_ = (beta0*mu0 + counter1*Cmean)/beta0_;
    nu0_ = nu0 + counter1;
    dMu = mu0 - Cmean;
    iW0_ = iW0 + C*C' - counter1*(Cmean*Cmean') + (beta0*counter1/beta0_)*(dMu*dMu');
    W0_ = inv(iW0_);
    A_C = wishrnd((W0_ + W0_')*0.5, nu0_);
    mu_C = mvnrndpre(mu0_, beta0_*A_C);
    %sample the prior of of CV2 (i=1).    
    Cmean2 = mean(C2,2);
    beta0_ = beta0 + counter3;
    mu0_ = (beta0*mu0 + counter3*Cmean2)/beta0_;
    nu0_ = nu0 + counter3;
    dMu = mu0 - Cmean2;
    iW0_ = iW0 + C2*C2' - counter3*(Cmean2*Cmean2') + (beta0*counter3/beta0_)*(dMu*dMu');
    W0_ = inv(iW0_);
    A_C2 = wishrnd((W0_ + W0_')*0.5, nu0_);
    mu_C2 = mvnrndpre(mu0_, beta0_*A_C2);
    %sample the prior of of CV3 (i=3).       
    Cmean3 = mean(C3,2);
    beta0_ = beta0 + counter5;
    mu0_ = (beta0*mu0 + counter5*Cmean3)/beta0_;
    nu0_ = nu0 + counter5;
    dMu = mu0 - Cmean3;
    iW0_ = iW0 + C3*C3' - counter5*(Cmean3*Cmean3') + (beta0*counter5/beta0_)*(dMu*dMu');
    W0_ = inv(iW0_);
    A_C3 = wishrnd((W0_ + W0_')*0.5, nu0_);
    mu_C3 = mvnrndpre(mu0_, beta0_*A_C3);
    %sample the prior of of CV4 (i=4).   
    Cmean4 = mean(C4,2);
    beta0_ = beta0 + counter7;
    mu0_ = (beta0*mu0 + counter7*Cmean4)/beta0_;
    nu0_ = nu0 + counter7;
    dMu = mu0 - Cmean4;
    iW0_ = iW0 + C4*C4' - counter7*(Cmean4*Cmean4') + (beta0*counter7/beta0_)*(dMu*dMu');
    W0_ = inv(iW0_);
    A_C4 = wishrnd((W0_ + W0_')*0.5, nu0_);
    mu_C4 = mvnrndpre(mu0_, beta0_*A_C4);

    %sample the prior of of CT (i=2).       
    CTmean = mean(CT,2);
    beta0_ = beta0 + counter2;
    mu0_ = (beta0*mu0 + counter2*CTmean)/beta0_;
    nu0_ = nu0 + counter2;
    dMu = mu0 - CTmean;
    iW0_ = iW0 + CT*CT' - counter2*(CTmean*CTmean') + (beta0*counter2/beta0_)*(dMu*dMu');
    W0_ = inv(iW0_);
    A_CT = wishrnd((W0_ + W0_')*0.5, nu0_);
    mu_CT = mvnrndpre(mu0_, beta0_*A_CT);
    %sample the prior of of CT2 (i=1).      
    CTmean2 = mean(CT2,2);
    beta0_ = beta0 + counter4;
    mu0_ = (beta0*mu0 + counter4*CTmean2)/beta0_;
    nu0_ = nu0 + counter4;
    dMu = mu0 - CTmean2;
    iW0_ = iW0 + CT2*CT2' - counter4*(CTmean2*CTmean2') + (beta0*counter4/beta0_)*(dMu*dMu');
    W0_ = inv(iW0_);
    A_CT2 = wishrnd((W0_ + W0_')*0.5, nu0_);
    mu_CT2 = mvnrndpre(mu0_, beta0_*A_CT2);
    %sample the prior of of CT3 (i=3).      
    CTmean3 = mean(CT3,2);
    beta0_ = beta0 + counter6;
    mu0_ = (beta0*mu0 + counter6*CTmean3)/beta0_;
    nu0_ = nu0 + counter6;
    dMu = mu0 - CTmean3;
    iW0_ = iW0 + CT3*CT3' - counter6*(CTmean3*CTmean3') + (beta0*counter6/beta0_)*(dMu*dMu');
    W0_ = inv(iW0_);
    A_CT3 = wishrnd((W0_ + W0_')*0.5, nu0_);
    mu_CT3 = mvnrndpre(mu0_, beta0_*A_CT3);
    %sample the prior of of CT4 (i=4).      
    CTmean4 = mean(CT4,2);
    beta0_ = beta0 + counter8;
    mu0_ = (beta0*mu0 + counter8*CTmean4)/beta0_;
    nu0_ = nu0 + counter8;
    dMu = mu0 - CTmean4;
    iW0_ = iW0 + CT4*CT4' - counter8*(CTmean4*CTmean4') + (beta0*counter8/beta0_)*(dMu*dMu');
    W0_ = inv(iW0_);
    A_CT4 = wishrnd((W0_ + W0_')*0.5, nu0_);
    mu_CT4 = mvnrndpre(mu0_, beta0_*A_CT4);
    
%%%% Sample the feature vectors    
    fprintf('U');
    parfor ind = 1:M
        U(:, ind) = UpdateFactor(alpha, A_U, mu_U, ...
                                 PTF_ComputeQ(V, T, subVU_bag{ind}, subTU_bag{ind}), valsU_bag{ind});
    end
 

%%%% Equation (12) in the paper.     
    fprintf('V');
    parfor jnd = 1:N
        V(:, jnd) = UpdateFactor(alpha, A_V, mu_V, ...
                                 PTF_ComputeQ(U, T, subUV_bag{jnd}, subTV_bag{jnd}), valsV_bag{jnd});
    end

    fprintf('T');
    parfor knd = 1:K
        T(:, knd) = UpdateFactor(alpha, A_T, mu_T, ...
                                 PTF_ComputeQ(U, V, subUT_bag{knd}, subVT_bag{knd}), ...
                                 valsT_bag{knd});
    end    

%%%% Sample the semantically-biased feature vectors        
%%%% Equation (13) in the paper.      
    fprintf('CV');
    parfor jnd = 1:counter1
        C(:, jnd) = UpdateFactor(alpha, A_C, mu_C, ...
                                 PTF_ComputeQ(U, T, subUC_bag{jnd}, subTC_bag{jnd}), valsC_bag{jnd});
    end    
    fprintf('CV2');
    parfor jnd = 1:counter3
        C2(:, jnd) = UpdateFactor(alpha, A_C2, mu_C2, ...
                                 PTF_ComputeQ(U, T, subUC2_bag{jnd}, subTC2_bag{jnd}), valsC2_bag{jnd});
    end
    fprintf('CV3');
    parfor jnd = 1:counter5
        C3(:, jnd) = UpdateFactor(alpha, A_C3, mu_C3, ...
                                 PTF_ComputeQ(U, T, subUC3_bag{jnd}, subTC3_bag{jnd}), valsC3_bag{jnd});
    end
    fprintf('CV4');
    parfor jnd = 1:counter7
        C4(:, jnd) = UpdateFactor(alpha, A_C4, mu_C4, ...
                                 PTF_ComputeQ(U, T, subUC4_bag{jnd}, subTC4_bag{jnd}), valsC4_bag{jnd});
    end
			   
			   
    fprintf('CT');
    parfor jnd = 1:counter2
        CT(:, jnd) = UpdateFactor(alpha, A_CT, mu_CT, ...
                                  PTF_ComputeQ(U, V, subUCT_bag{jnd}, subTCT_bag{jnd}), valsCT_bag{jnd});
    end

    fprintf('CT2');
    parfor jnd = 1:counter4
        CT2(:, jnd) = UpdateFactor(alpha, A_CT2, mu_CT2, ...
                                  PTF_ComputeQ(U, V, subUCT2_bag{jnd}, subTCT2_bag{jnd}), valsCT2_bag{jnd});
    end

    fprintf('CT3');
    parfor jnd = 1:counter6
        CT3(:, jnd) = UpdateFactor(alpha, A_CT3, mu_CT3, ...
                                  PTF_ComputeQ(U, V, subUCT3_bag{jnd}, subTCT3_bag{jnd}), valsCT3_bag{jnd});
    end
			    
    fprintf('CT4');
    parfor jnd = 1:counter8
        CT4(:, jnd) = UpdateFactor(alpha, A_CT4, mu_CT4, ...
                                  PTF_ComputeQ(U, V, subUCT4_bag{jnd}, subTCT4_bag{jnd}), valsCT4_bag{jnd});
    end

VORG=V;
TORG=T;         

%%%% Update feature vectors by incorporating semantic biases into those.
    for knd = 1:N
      if (FREQ(knd,2)<=(THRESHOLD) && FREQ(knd,2)~=0)
	A5=0;
	A2=V(:, knd);
	for snd = 1:size(Xb2(CLS_bag{knd}))
	  CLSS=(Xb2(CLS_bag{knd}));
	  classC=CLSS(snd,1);
	  if classC ~= 0
	  A5=A5+(C2(:, subXb6_bag{classC}));
	  end
	end
	SIZE1=(size(Xb2(CLS_bag{knd})));
	SIZE1(1,1);
	A5=A5./SIZE1(1,1);
	if classC ~= 0
	  V(:, knd) =  A5*(1)+ A2*(0);
	end
      elseif (FREQ(knd,2)<=(THRESHOLD3) && FREQ(knd,2)~=0)
	A5=0;
	A2=V(:, knd);
	for snd = 1:size(Xb2(CLS_bag{knd}))
	  CLSS=(Xb2(CLS_bag{knd}));
	  classC=CLSS(snd,1);
	  if classC ~= 0
	  A5=A5+(C(:, subXb4_bag{classC}));
	  end
	end
	SIZE1=(size(Xb2(CLS_bag{knd})));
	SIZE1(1,1);
	A5=A5./SIZE1(1,1);
	if classC ~= 0
	  V(:, knd) =  A5*(1)+ A2*(0);
	end
      elseif (FREQ(knd,2)<=(THRESHOLD5) && FREQ(knd,2)~=0)
	A5=0;
	A2=V(:, knd);
	for snd = 1:size(Xb2(CLS_bag{knd}))
	  CLSS=(Xb2(CLS_bag{knd}));
	  classC=CLSS(snd,1);
	  if classC ~= 0
	  A5=A5+(C3(:, subXb8_bag{classC}));
	  end
	end
	SIZE1=(size(Xb2(CLS_bag{knd})));
	SIZE1(1,1);
	A5=A5./SIZE1(1,1);
	if classC ~= 0
	  V(:, knd) =  A5*(1)+ A2*(0);
	end
      elseif (FREQ(knd,2)<=(THRESHOLD7) && FREQ(knd,2)~=0)
	A5=0;
	A2=V(:, knd);
	for snd = 1:size(Xb2(CLS_bag{knd}))
	  CLSS=(Xb2(CLS_bag{knd}));
	  classC=CLSS(snd,1);
	  if classC ~= 0
	  A5=A5+(C4(:, subXb10_bag{classC}));
	  end
	end
	SIZE1=(size(Xb2(CLS_bag{knd})));
	SIZE1(1,1);
	A5=A5./SIZE1(1,1);
	if classC ~= 0
	  V(:, knd) =  A5*(Epsilon)+ A2*(1-Epsilon);
	end
     end
    end
    

    for knd = 1:K
      if (FREQ2(knd,2)<=THRESHOLD2 && FREQ2(knd,2)~=0)
	A5=0;
	A2=T(:, knd);
	for snd = 1:size(Xbb2(CLS_bag2{knd}))
	  CLSS=(Xbb2(CLS_bag2{knd}));
	  classC=CLSS(snd,1);
	  if classC ~= 0
	  A5=A5+(CT2(:, subXbb6_bag{classC}));
	  end
	end
	SIZE1=(size(Xbb2(CLS_bag2{knd})));
	SIZE1(1,1);
	A5=A5./SIZE1(1,1);
	if classC ~= 0
	  T(:, knd) =  A5*(1)+ A2*(0);
	end
      elseif (FREQ2(knd,2)<=THRESHOLD4 && FREQ2(knd,2)~=0)
	A5=0;
	A2=T(:, knd);
	for snd = 1:size(Xbb2(CLS_bag2{knd}))
	  CLSS=(Xbb2(CLS_bag2{knd}));
	  classC=CLSS(snd,1);
	  if classC ~= 0
	  A5=A5+(CT(:, subXbb4_bag{classC}));
	  end
	end
	SIZE1=(size(Xbb2(CLS_bag2{knd})));
	SIZE1(1,1);
	A5=A5./SIZE1(1,1);
	if classC ~= 0
	  T(:, knd) =  A5*(1)+ A2*(0);
	end
      elseif (FREQ2(knd,2)<=THRESHOLD6 && FREQ2(knd,2)~=0)
	A5=0;
	A2=T(:, knd);
	for snd = 1:size(Xbb2(CLS_bag2{knd}))
	  CLSS=(Xbb2(CLS_bag2{knd}));
	  classC=CLSS(snd,1);
	  if classC ~= 0
	  A5=A5+(CT3(:, subXbb8_bag{classC}));
	  end
	end
	SIZE1=(size(Xbb2(CLS_bag2{knd})));
	SIZE1(1,1);
	A5=A5./SIZE1(1,1);
	if classC ~= 0
	  T(:, knd) =  A5*(1)+ A2*(0);
	end
      elseif (FREQ2(knd,2)<=THRESHOLD8 && FREQ2(knd,2)~=0)
	A5=0;
	A2=T(:, knd);
	for snd = 1:size(Xbb2(CLS_bag2{knd}))
	  CLSS=(Xbb2(CLS_bag2{knd}));
	  classC=CLSS(snd,1);
	  if classC ~= 0
	  A5=A5+(CT4(:, subXbb10_bag{classC}));
	  end
	end
	SIZE1=(size(Xbb2(CLS_bag2{knd})));
	SIZE1(1,1);
	A5=A5./SIZE1(1,1);
	if classC ~= 0
	  T(:, knd) =  A5*(Epsilon)+ A2*(1-Epsilon);
	end
      end
    end


    
    %record samples
    count = mod(iter - 1, n_sample) + 1;
    if save_sample
        save(sprintf(sample_file_format, run_name, D, iter), ...
	     'alpha', 'U', 'V', 'T', 'C', 'A_U', 'mu_U', 'A_V', ...
             'mu_V', 'A_T', 'mu_T', 'A_C', 'mu_C');
    end
    
    Us(:, count) = U(:);
    Vs(:, count) = V(:);
    Ts(:, count) = T(:);
    VORGs(:, count) = VORG(:);
    TORGs(:, count) = TORG(:);
  
    ysTr1(:, count) = PTF_Reconstruct(subs, U, VORG, T);
    ysTr2(:, count) = PTF_Reconstruct(subs, U, V, TORG);
    ysTr(:, count) = (ysTr1(:, count) + ysTr2(:, count))/2;

    yTr = mean(ysTr(:, 1:min(iter, n_sample)), 2);
    rmseTr = RMSE(yTr - vals);
     
%%%% Sample the unobserved ratings for original tensor.
    rmseTe = nan;
    if te
      ysTe(:, count) = PTF_Reconstruct(subsTe, U, VORG, TORG);
      yTe = mean(ysTe(:, 1:min(iter, n_sample)), 2);
      rmseTe = RMSE(yTe - valsTe);       
     end
     
     fprintf('. alpha = %g. Using %d samples. RMSE=%0.4f/%0.4f. ETA=%0.2f hr\n', ...
	 alpha, min(iter, n_sample), rmseTr, rmseTe,...
	 (max_iter + sample_idx - 1 - iter)*toc/(iter - sample_idx + 1)/3600);

     if mod(iter,25)==0
       !pwd
     end
     
     
     
end


function [r] = UpdateFactor(alpha, A, mu, Q, vv)

if isempty(vv); 
    Q = 0;vv = 0;
end

Aj_ = A + alpha*(Q*Q');
muj_ = Aj_\(A*mu + alpha*(Q*vv));

r = mvnrndpre(muj_, Aj_);


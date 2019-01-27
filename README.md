# sstf
The code for "Semantic sensitive tensor factorization"
https://www.sciencedirect.com/science/article/pii/S000437021500137X

This code was originally implemented by the authors of the following paper and updated by tokoma7777 to suit SSTF.

BPTF: Liang Xiong, Xi Chen, Tzu-kuo Huang, Jeff Schneider, and Jaime Carbonell, Temporal Collaborative Filtering with Bayesian Probabilistic Tensor Factorization, SIAM Data Mining 2010

To run the prototype, simply enter "demo_del" in a Matlab terminal window. It outputs the results of BPTF and SSTF.  Currently, the
movieLens dataset is set up for this demo.  The data is one of our three-fold evaluations. Please note that we used the Parallel
Computing Toolbox so as to compute feature vectors in parallel (see Section 4.3.4). If you do not have this toolbox, comment out the lines
mentioning matlabpool though time taken will ballon if you do not use this toolbox.  We also used the Statistics Toolbox.  

We added the comments for the codes that correspond to Eq. (9), (10), (11), (12), and (13) in the paper to the file "SSTF.m". This greatly
helps readers to understand our MCMC and the code.  We also added several simple comments on the file "SSTF.m".  Note that the code is
not implemented in a sophisticated manner, however, it works well.

"TTeN2234.mat" includes test dataset and "TTrN2234.mat" includes training dataset. 
"clsPclsMap1.dat" contains the relationship of item id and class id. Item with class id "0" means that it has no classes in the taxonomy. Item can have multiple classes.
"clsPclsMap2.dat" contains the relationship of tag id and class id. Tag with class id "0" means that it has no classes in the taxonomy. Tag can have one class.
"cube0-1ORG.dat" contains userId, itemId, and tagId in training dataset.

This code is used only for research purposes. Note that we do not responsible for any problems that may occur due to this code.


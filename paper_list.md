## Optimization & Generalization
1. [ProxSGD](https://openreview.net/pdf?id=HygpthEtvr): adjusted gradients for L1 norm to get more accurate 0 weights.
2. *[Unordered gradient estimator for discrete random variable](https://openreview.net/pdf?id=rklEj2EFvB): an alternative solution of Gumbel-softmax with smaller variance and non-biased.
3. *[EFFECT OF ACTIVATION FUNCTIONS ON THE TRAINING OF OVERPARAMETRIZED NEURAL NETS](https://openreview.net/pdf?id=rkgfdeBYvH): check Eq.2, may be related to theory proof of DRL, check related work [A Convergence Theory for Deep Learning via Over-Parameterization](http://proceedings.mlr.press/v97/allen-zhu19a.html), [Neural Tangent Kernel](http://papers.nips.cc/paper/8076-neural-tangent-kernel-convergence-and-generalization-in-neural-networks.pdf).
4. [LAMB](https://openreview.net/pdf?id=Syx4wnEtvH): A method of layer-wise adaptive learning rate for large batch optimization  for deep learning, which can significantly reduce training time of large model of large dataset, such as BERT.
   
5. [RETHINKING SOFTMAX CROSS-ENTROPY LOSS
FOR ADVERSARIAL ROBUSTNESS](https://openreview.net/pdf?id=Byg9A24tvB):  demonstrates that softmax cross-entropy loss (and its variants) convey unexpected supervisory signals on the training points, which make the learned features tend to spread over the space sparsely, proposes  Max-Mahalanobis center (MMC) loss which can explicitly control the inter-class dispersion by a single hyperparameter (pre-selected class centers) and further concentrate on improving intra-class compactness in the training procedure to induce high-density regions. 
Other related works focus on intra-class compactness without controlling inter-class dispersion, such as constrative loss, triplet loss, center loss. 

1. [TRUTH OR BACKPROPAGANDA? AN EMPIRICAL INVESTIGATION OF DEEP LEARNING THEORY](https://openreview.net/pdf?id=HyxyIgHFvr): provides some interesting findings, e.g. 1) Yet for neural networks, it is not at all clear which form of $l_2$-regularization is optimal. We show this by constructing a simple alternative: biasing solutions toward a non-zero norm still works and can even measurably improve performance for modern architectures; 2). ResNets do not conform to wide-network theories, such as the neural tangent kernel, and that the interaction between skip connections and batch normalization plays a  role;  3). Generalization theory has provided guarantees for the performance of low-rank networks. However, high-rank weight matrices often outperforms that which promotes low-rank matrices and even more robust against adversarial attacks.


   
7. [FANTASTIC GENERALIZATION MEASURES
AND WHERE TO FIND THEM](https://openreview.net/pdf?id=SJgIPJBFvH): comparing more than 40 measures of generalization, including theoretical, empirical, and optimization based measures, where sharpness-based measures (PAC-Bayes bound and worst-case sharpness) show better predictive power on generalization (especially an amended version in Eq.55), and the variance of gradients also show strong predictive power on generalization across every type of hyper-parameters.

8. [PICKING WINNING TICKETS BEFORE TRAINING
BY PRESERVING GRADIENT FLOW](https://openreview.net/pdf?id=SkgsACVKPH): utilising  Hessian matrix for selecting parameters after initialization and before training.

6. [UNDERSTANDING WHY NEURAL NETWORKS GENERALIZE WELL THROUGH GSNR OF PARAMETERS](https://openreview.net/pdf?id=HyevIJStwH)

## Generative models
1. [Hamiltonian Generative Networks](https://openreview.net/pdf?id=HJenn6VFvB): similar with ODE, using dynamic modeling, invertible and volume-preserving.
2. [Convolutional Conditional Neural Process](http://www.openreview.net/pdf?id=Skey4eBYPS):adding convolutional operations in encoder and decoder, better in capturing invariant representations than Conditional Neural Process (CNP).
3. [GENERAL INCOMPRESSIBLE-FLOW NETWORKS (GIN)](https://openreview.net/pdf?id=rygeHgSFDH): based on non-linear Independent Component Analysis (ICA) and RealNVP architecture, can learn disentangled latent variables. (related work: [Nonlinear ICA Using Auxiliary Variables
and Generalized Contrastive Learning](https://arxiv.org/pdf/1805.08651.pdf))

4. [Target-Embedding Autoencoders for Supervised Representation Learning](https://openreview.net/pdf?id=BygXFkSYDH): for high-dimensional target space, such as multi-label or sequence forecasting. 

## Inference methods
1. [SUMO](https://openreview.net/pdf?id=SylkYeHtwr): combined by estimators from IWAE(Importance Weighted AutorEncoder) and Russian roulette estimator, unbiased estimator of log probability of latent variable models.

## Continual learning

1. [BatchEnsemble: AN ALTERNATIVE APPROACH TO
EFFICIENT ENSEMBLE AND LIFELONG LEARNING](https://openreview.net/pdf?id=Sklf1yrYDr): defining each weight matrix to be the Hadamard
product of a shared weight among all ensemble members and a rank-one matrix per member.

2. [UNCERTAINTY-GUIDED CONTINUAL LEARNING WITH
BAYESIAN NEURAL NETWORKS](https://openreview.net/pdf?id=HklUCCVKDB): update learning rate of mean of weights by its stddev, which is very similar with natural gradient, the prior is a mixture Gaussian with 2 components in this paper.

3. [CONTINUAL LEARNING WITH ADAPTIVE WEIGHTS
(CLAW)](https://openreview.net/pdf?id=Hklso24Kwr), using 3 extra parameters to control the dropout rate of hidden units and those parameters are inferred by an extra amortized inference network (like VAE's encoder)

## Transfer learning

1. [HYPERPARAMETERS FOR FINE-TUNING](https://openreview.net/pdf?id=B1g8VkHFPH): 

    1).Optimal hyperparameters for fine-tuning are not only dataset dependent, but are also dependent on the similarity between the source and target domains, e.g. zero momentum sometimes work better for fine-tuning on tasks that are similar with the source domain, while nonzero momentum works better for target domains that are different from the source domain; 

    2). Regularization methods that were designed to keep models close to the initial model does not necessarily work for “dissimilar” datasets, especially for nets with Batch Normalization.

2. [UNDERSTANDING AND IMPROVING INFORMATION
TRANSFER IN MULTI-TASK LEARNING](https://openreview.net/pdf?id=SylzhkBtDB)(
this paper used multi-head setting for training multiple tasks at the same time):
     
    1). show that capacity of shared module (i.e. the output dimension of shared module $r$) determines whether or not there is information transfer between tasks: If the shared module is too small, then tasks may interfere negatively with each other. But if it is too large, then there may be no transfer between tasks. For linear models, $r \ge k$ results in no transfer, where $k$ is number of tasks;
    2). show that more similar two tasks are less samples from the source task are required for positive transfer, proposed  **covariance similarity score** for measuring similarity between two tasks (only needs data);

    [The condition number of the matrix measures the ratio of the maximum relative stretching to the maximum relative shrinking that matrix does to any non zero vectors.] 
3. [GRADIENTS AS FEATURES FOR DEEP REPRESENTATION LEARNING](https://openreview.net/pdf?id=BkeoaeHKDS): use gradients w.r.t. parameters of upper layers (close to output) of a pre-trained model as features, which is a local linear approximation of fine-tuning (by Taylor expansion) and then can be theoretically analyzed by [NTK](http://papers.nips.cc/paper/8076-neural-tangent-kernel-convergence-and-generalization-in-neural-networks.pdf) or related approaches. 

## Meta learning
1. [META DROPOUT](https://openreview.net/pdf?id=BJgd81SYwr)
2. 
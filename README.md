This is a woefully short README.  More of a 200 word summary really.  

This is a project which implements a Deep Belief Net using Stacked Restricted Boltzmann Machines, for such tasks as: Classification, Regression, Auto-encoding or Clustering.  Clustering can be seen as a special case of auto-encoding, with the only difference that the bottleneck layer contains categorical/soft-max neurons that encode the input into a single class/category. Thus, the auto-encoder learns to cluster optimally such that the input can be decoded/reconstructed from the cluster label most accurately.

Please see this paper by G. E. Hinton and R. R. Salakhutdinov for more details of the deep auto-encoder: http://www.cs.toronto.edu/~rsalakhu/papers/science.pdf.  

The implementation allows three types of neurons in any layer of the auto-encoder:

1.  Linear nodes with the Gaussian activation function
2.  Binary nodes with the logistic activation function
3.  Categorical nodes with the soft-max activation function

The back-propagation phase for fine-tuning the weights of the network uses Stochastic Meta Descent algorithm (see http://www.schraudolph.org/teach/MLSS4up.pdf) to minimize the loss, which is a faster variant of Stochastic Gradient Descent.

The implementation is written entirely in C++ and is built on top of Apple's Accelerate framework, using the fast vecLib and BLAS libraries for extensive data concurrency. Thus, for now this implementation works only on OS X or iOS.  However, it should be very easy to find analogues of these libraries for Windows or Linux.

Furthermore, the implementation makes use of the OpenMP platform for extensive parallelization of the algorithm (essentially every portion that can be parallelized is parallelized).
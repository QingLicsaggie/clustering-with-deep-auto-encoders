This is a woefully short README.  More of a 200 word summary really.  

This in a fast implementation of a clustering algorithm that's based on the well known deep auto-encoder, proposed by G. E. Hinton and R. R. Salakhutdinov (see http://www.cs.toronto.edu/~rsalakhu/papers/science.pdf).

The implementation allows three types of neurons in any layer of the auto-encoder:

1.  Linear nodes with the Gaussian activation function
2.  Binary nodes with the logistic activation function
3.  Categorical nodes with the soft-max activation function

The main change made to the deep auto encoder algorithm that allows clustering is to make the bottleneck layer contain a single categorical/soft-max neuron that encodes the input into a single class/category.  Thus, this can be seen as an extreme encoding.  The auto-encoder thus learns to cluster optimally such that the input can be decoded/reconstructed from the cluster label most accurately.

The fine-tuning stage of the auto-encoder training using the Stochastic Meta Descent algorithm (see http://www.schraudolph.org/teach/MLSS4up.pdf) to minimize the loss.  

The implementation is built on top of Apple's Accelerate framework, using the fast vecLib and BLAS libraries. Thus, for now this implementation works only on OS X and iOS.  However, it should be very easy to find analogues of these libraries for Windows or Linux.

Furthermore, the implementation makes use of the OpenMP platform for extensive parallelization of the algorithm (essentially every portion that can be parallelized is parallelized).
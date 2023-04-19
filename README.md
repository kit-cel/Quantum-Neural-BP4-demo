# Neural Belief Propagation Decoding of Quantum LDPC Codes Using Overcomplete Check Matrices
## Sisi Miao, Alexander Schnerring, Haizheng Li, and Laurent Schmalen

This jupyter notebook provides the implementation which produces the results in [1]. In specific, this notebook contains:
* The neural BP (NBP) decoder
* The function for training, can be run on CPU or GPU
* Call on the C++ functions which evaluate the trained decoder
* Additionally, the overcomplete check matrices used in [1] are provided in the ./PCM folder

[1] S. Miao, A. Schnerring, H. Li and L. Schmalen, "Neural belief propagation decoding of quantum LDPC codes using overcomplete check matrices," Proc. IEEE Inform. Theory Workshop (ITW), Saint-Malo, France, Apr. 2023, https://arxiv.org/abs/2212.10245

## Abstract
The recent success in constructing asymptotically good quantum low-density
parity-check (QLDPC) codes makes this family of codes a promising candidate for
error-correcting schemes in quantum computing. However, conventional belief
propagation (BP) decoding of QLDPC codes does not yield satisfying performance
due to the presence of unavoidable short cycles in their Tanner graph and the
special degeneracy phenomenon. In this work, we propose to decode QLDPC codes
based on a check matrix with redundant rows, generated from linear combinations
of the rows in the original check matrix. This approach yields a significant
improvement in decoding performance with the additional advantage of very low
decoding latency. Furthermore, we propose a novel neural belief propagation
decoder based on the quaternary BP decoder of QLDPC codes which leads to
further decoding performance improvements.

# Progressive Growing of Variational Gradient Flow
A tensorflow implementation of VGrow by using progressive growing method descriped in the following paper:
* [Deep Generative Learning via Variational Gradient Flow](https://arxiv.org/abs/1901.08469).
* [Progressive Growing of GANs for Improved Quality, Stability, and Variation](https://arxiv.org/abs/1710.10196).

## System requirements

* We only test our model on Linux. 
* 64-bit Python 3.6 and Tensorflow 1.12.0
* When you want to generate higher resolution image than 128x128, We recommend GPU with at least 16GB memory.
* NVIDIA driver 384.145  or newer, CUDA toolkit 9.0 or newer, cuDNN 7.1.2 or newer. We test the code based on the following two configuration.
  * NIVDIA driver 384.145, CUDA V9.0.176, Tesla V100
  * NVIDIA driver 410.93 , CUDA V10.0.130, RTX 2080 Ti
  
## Reference
The implementation is motivated based on the projects:  
[1]https://github.com/tkarras/progressive_growing_of_gans   
[2]https://github.com/gefeiwang/PGVGrow 

# :robot: PROJECT: CERCI (**C**alculating **E**rror in **R**ealtime from **C**omputer **I**nteractions)

## :brain: An advanced AI designed to simulate human interaction 
- CERCI, short for '**C**alculating **E**rror in **R**ealtime from **C**omputer **I**nteractions', is a custom made deep learning model aimed at simulating a human interaction
- CERCI will be created using Python and CUDA

## :hourglass: Plans
* Have CERCI be able to have a one on one conversation with a memory bank of previous interactions using recurrent neural networks
* Provide CERCI with sight and hearing using convolutional neural networks
* Train CERCI on the GPU using Nvidia CUDA
* Provide CERCI with her own voice to communicate back
* Have realtime analysis for fluent conversations

## :pushpin: TODO
* Add batch normalization to the network
* Add weight initializing functions using the proper weight generation techniques
* Cleanup the files and optimize the matrix operations and misc functions
* Implement 3D tensor class and operations and network which allows for parallel processing of multiple training sets at the same time
	* Do this with a stretch tensor function that makes a tensor the length of the input set then perform forward and back propogation on the same big tensor which does all the operations at once 
	* To update those weights take an average of the error calculated and then update the original

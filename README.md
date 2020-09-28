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
* Remove unused functions from the misc library
* Have the returnNetwork() function return the ADAM values aswell for further training which can be loaded in
* Add a random weight initialization tool so that the weights do not have to be initialized each time and can rather be a number which will be generated as a tensor or a matrix
* Refactor the tensor classes into one big class which can support tensors of any length for simplicity, as well as reconfiguring the convolutional layers into one bigger class
* Some of the training predictions and things have been broken by the change to the requirements from the single convnet

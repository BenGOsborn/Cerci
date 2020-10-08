#include "optimizers.cuh"

// Make sure that this calls the constructor of the previous class too
FullyConnectedAdam::FullyConnectedAdam(std::unique_ptr<Matrix>& weight_set, std::unique_ptr<Matrix>& bias_set, float lr = 0.1, float beta1 = 0.9) : FullyConnected(weight_set, bias_set, lr) {

}


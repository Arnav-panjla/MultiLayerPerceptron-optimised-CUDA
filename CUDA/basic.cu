#include <iostream>
#include <cmath>
#include <cstdlib>

#define INPUT_SIZE 784
#define HIDDEN_SIZE 256
#define OUTPUT_SIZE 10
#define LEARNING_RATE 0.001

__device__ float relu(float x) {
    return x > 0 ? x : 0;
}

__device__ float relu_derivative(float x) {
    return x > 0 ? 1.0f : 0.0f;
}

__global__ void linear_forward(float* input, float* weight, float* bias, float* output, int in_dim, int out_dim) {
    int idx = threadIdx.x;
    if (idx < out_dim) {
        float sum = 0.0;
        for (int i = 0; i < in_dim; i++) {
            sum += input[i] * weight[i * out_dim + idx];
        }
        output[idx] = sum + bias[idx];
    }
}

__global__ void apply_relu(float* input, float* output, int size) {
    int idx = threadIdx.x;
    if (idx < size) {
        output[idx] = relu(input[idx]);
    }
}

__global__ void softmax(float* input, float* output, int size) {
    int idx = threadIdx.x;
    __shared__ float sum_exp;
    __shared__ float max_val;

    if (idx == 0) {
        max_val = input[0];
        for (int i = 1; i < size; ++i) {
            if (input[i] > max_val) max_val = input[i];
        }
        sum_exp = 0.0f;
        for (int i = 0; i < size; ++i) {
            sum_exp += expf(input[i] - max_val);
        }
    }
    __syncthreads();

    if (idx < size) {
        output[idx] = expf(input[idx] - max_val) / sum_exp;
    }
}

__global__ void compute_loss_grad(float* probs, int label, float* grad, int size) {
    int idx = threadIdx.x;
    if (idx < size) {
        grad[idx] = probs[idx];
        if (idx == label)
            grad[idx] -= 1.0f;
    }
}

__global__ void sgd_update(float* weight, float* grad, float lr, int size) {
    int idx = threadIdx.x;
    if (idx < size) {
        weight[idx] -= lr * grad[idx];
    }
}


__global__ void linear_backward_W2(float* grad_output, float* activation_input, float* grad_weight, int in_dim, int out_dim) {
    int idx = threadIdx.x;  // [0, in_dim * out_dim)
    if (idx < in_dim * out_dim) {
        int i = idx / out_dim; // input index
        int j = idx % out_dim; // output index
        grad_weight[idx] = activation_input[i] * grad_output[j];  // dW = a^T * dL/dz
    }
}

__global__ void linear_backward_b2(float* grad_output, float* grad_bias, int out_dim) {
    int idx = threadIdx.x;
    if (idx < out_dim) {
        grad_bias[idx] = grad_output[idx];
    }
}

__global__ void linear_backward_hidden(float* grad_output, float* weight, float* grad_input, int in_dim, int out_dim) {
    int idx = threadIdx.x;  // [0, in_dim)
    if (idx < in_dim) {
        float sum = 0.0;
        for (int j = 0; j < out_dim; ++j) {
            sum += grad_output[j] * weight[idx * out_dim + j];
        }
        grad_input[idx] = sum;
    }
}



int main() {
    float h_input[INPUT_SIZE] = { /* fill with MNIST image data */ };
    int h_label = 3; // Actual image value

    float *d_input, *d_fc1_out, *d_relu_out, *d_fc2_out, *d_probs;
    float *d_W1, *d_b1, *d_W2, *d_b2;
    float *d_grad_fc2, *d_grad_W2;

    cudaMalloc(&d_input, INPUT_SIZE * sizeof(float));
    cudaMalloc(&d_W1, INPUT_SIZE * HIDDEN_SIZE * sizeof(float));
    cudaMalloc(&d_b1, HIDDEN_SIZE * sizeof(float));
    cudaMalloc(&d_fc1_out, HIDDEN_SIZE * sizeof(float));
    cudaMalloc(&d_relu_out, HIDDEN_SIZE * sizeof(float));
    cudaMalloc(&d_W2, HIDDEN_SIZE * OUTPUT_SIZE * sizeof(float));
    cudaMalloc(&d_b2, OUTPUT_SIZE * sizeof(float));
    cudaMalloc(&d_fc2_out, OUTPUT_SIZE * sizeof(float));
    cudaMalloc(&d_probs, OUTPUT_SIZE * sizeof(float));
    cudaMalloc(&d_grad_fc2, OUTPUT_SIZE * sizeof(float));
    cudaMalloc(&d_grad_W2, HIDDEN_SIZE * OUTPUT_SIZE * sizeof(float));

    cudaMemcpy(d_input, h_input, INPUT_SIZE * sizeof(float), cudaMemcpyHostToDevice);

    // Forward pass
    linear_forward<<<1, HIDDEN_SIZE>>>(d_input, d_W1, d_b1, d_fc1_out, INPUT_SIZE, HIDDEN_SIZE);
    apply_relu<<<1, HIDDEN_SIZE>>>(d_fc1_out, d_relu_out, HIDDEN_SIZE);
    linear_forward<<<1, OUTPUT_SIZE>>>(d_relu_out, d_W2, d_b2, d_fc2_out, HIDDEN_SIZE, OUTPUT_SIZE);
    softmax<<<1, OUTPUT_SIZE>>>(d_fc2_out, d_probs, OUTPUT_SIZE);

    // Compute gradient of loss
    compute_loss_grad<<<1, OUTPUT_SIZE>>>(d_probs, h_label, d_grad_fc2, OUTPUT_SIZE);

    // Backward pass
    // FC2 backward: compute dW2 and db2
    linear_backward_W2<<<1, HIDDEN_SIZE * OUTPUT_SIZE>>>(d_grad_fc2, d_relu_out, d_grad_W2, HIDDEN_SIZE, OUTPUT_SIZE);
    linear_backward_b2<<<1, OUTPUT_SIZE>>>(d_grad_fc2, d_b2, OUTPUT_SIZE);

    // SGD step for W2 and b2
    sgd_update<<<1, HIDDEN_SIZE * OUTPUT_SIZE>>>(d_W2, d_grad_W2, LEARNING_RATE, HIDDEN_SIZE * OUTPUT_SIZE);
    sgd_update<<<1, OUTPUT_SIZE>>>(d_b2, d_grad_fc2, LEARNING_RATE, OUTPUT_SIZE);  // reuse grad_fc2 as db2

    linear_backward_W2<<<1, INPUT_SIZE * HIDDEN_SIZE>>>(d_grad_relu, d_input, d_grad_W1, INPUT_SIZE, HIDDEN_SIZE);
    linear_backward_b2<<<1, HIDDEN_SIZE>>>(d_grad_relu, d_b1, HIDDEN_SIZE);

    // SGD update
    sgd_update<<<1, INPUT_SIZE * HIDDEN_SIZE>>>(d_W1, d_grad_W1, LEARNING_RATE, INPUT_SIZE * HIDDEN_SIZE);
    sgd_update<<<1, HIDDEN_SIZE>>>(d_b1, d_grad_relu, LEARNING_RATE, HIDDEN_SIZE);



    // Free memory
    cudaFree(d_input); cudaFree(d_fc1_out); cudaFree(d_relu_out);
    cudaFree(d_W1); cudaFree(d_b1); cudaFree(d_W2); cudaFree(d_b2);
    cudaFree(d_fc2_out); cudaFree(d_probs); cudaFree(d_grad_fc2);
    cudaFree(d_grad_W2);

    cudaFree(d_grad_relu);
    cudaFree(d_grad_fc1);
    cudaFree(d_grad_W1);

    return 0;
}

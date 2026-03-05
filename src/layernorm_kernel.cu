#include "includes/block_reduce.h"
#include "includes/cuda_util.h"
#include "includes/kernels.h"

#include <cooperative_groups.h>
#include <cstddef>

namespace cg = cooperative_groups;
namespace lightseq {
namespace cuda {

const float LN_EPSILON = 1e-8f;
#define TILE_DIM 32

/**
@brief: ker_layer_norm
Standard layer normalization.
It will not only output the layer norm result,
  but also outputs variance.
  may also output means, depends on whether
  the means argument is nullptr

@thread
gridDim.x = batch_size * seq_len
blockDim.x = hidden_size

@param
ln_res: [batch_size * seq_len, hidden_size], ln result.
vars: [batch_size * seq_len], variance per token
means: [batch_size * seq_len], means per token, can be nullput
inp: [batch_size * seq_len, hidden_size], ln input.
scale: [hidden_size], ln scale
bias: [hidden_size], ln bias
*/
template <typename T>
__global__ void ker_layer_norm(T *ln_res, T *vars, T *means, const T *inp,
                               const T *scale, const T *bias, int hidden_size) {

  /// BEGIN ASSIGN4_2_1
  /// TODO
  // Hints:
  // 1. Compute x and x^2 with reinterpret_cast by casting to float4 for speedup
  // 2. Compute reduce sum with blockReduce and add epsilon with LN_EPSILON
  // 3. Compute layernorm result with reinterpret_cast by casting to float4 for
  // speedup

  // Step 1
  float l_sum = 0;
  float l_sum_sq = 0;
  const float4 *inp_f4 =
      reinterpret_cast<const float4 *>(inp) + blockIdx.x * hidden_size / 4;
  for (uint idx = threadIdx.x; idx < hidden_size / 4; idx += blockDim.x) {
    float4 val = inp_f4[idx];
    l_sum += val.x + val.y + val.z + val.w;
    l_sum_sq += val.x * val.x + val.y * val.y + val.z * val.z + val.w * val.w;
  }

  // Step 2
  float mean = 0;
  float var = 0;
  blockReduce<ReduceType::kSum, 1>(&l_sum);
  blockReduce<ReduceType::kSum, 1>(&l_sum_sq);

  if (threadIdx.x == 0) {
    mean = l_sum / hidden_size;
    var = l_sum_sq / hidden_size - mean * mean;
    if (means)
      means[blockIdx.x] = mean;
    if (vars)
      vars[blockIdx.x] = var;
  }
  __shared__ float s_mean, s_var;
  if (threadIdx.x == 0) {
    s_mean = mean;
    s_var = var;
  }
  __syncthreads();
  mean = s_mean;
  var = s_var;

  // Step 3
  float4 *ln_res_f4 =
      reinterpret_cast<float4 *>(ln_res) + blockIdx.x * hidden_size / 4;
  const float4 *scale_f4 = reinterpret_cast<const float4 *>(scale);
  const float4 *bias_f4 = reinterpret_cast<const float4 *>(bias);
  float inv_std = rsqrtf(var + LN_EPSILON);

  for (uint idx = threadIdx.x; idx < hidden_size / 4; idx += blockDim.x) {
    float4 val = inp_f4[idx];
    float4 s = scale_f4[idx];
    float4 b = bias_f4[idx];
    float4 res;
    res.x = (val.x - mean) * inv_std * s.x + b.x;
    res.y = (val.y - mean) * inv_std * s.y + b.y;
    res.z = (val.z - mean) * inv_std * s.z + b.z;
    res.w = (val.w - mean) * inv_std * s.w + b.w;
    ln_res_f4[idx] = res;
  }
  // END ASSIGN4_2_1
}

extern "C" {
void launch_layernorm(float *ln_res, float *vars, float *means,
                      const float *inp, const float *scale, const float *bias,
                      int batch_size, int hidden_dim, cudaStream_t stream) {
  if (hidden_dim % 4 != 0) {
    throw std::runtime_error("violate hidden_dim % 4 = 0");
  }
  int float_size = sizeof(float);
  int input_size = batch_size * hidden_dim * float_size;
  int scale_size = hidden_dim * float_size;
  int bias_size = hidden_dim * float_size;
  int output_size = batch_size * hidden_dim * float_size;
  int mean_size = batch_size * float_size;
  int var_size = batch_size * float_size;

  float *d_ln_res, *d_vars, *d_means, *d_inp, *d_scale, *d_bias;
  cudaMalloc((void **)&d_ln_res, output_size);
  cudaMalloc((void **)&d_vars, var_size);
  cudaMalloc((void **)&d_means, mean_size);
  cudaMalloc((void **)&d_inp, input_size);
  cudaMalloc((void **)&d_scale, scale_size);
  cudaMalloc((void **)&d_bias, bias_size);

  cudaMemcpy(d_inp, inp, input_size, cudaMemcpyHostToDevice);
  cudaMemcpy(d_scale, scale, scale_size, cudaMemcpyHostToDevice);
  cudaMemcpy(d_bias, bias, bias_size, cudaMemcpyHostToDevice);

  // For using float4
  int hidden_dim_f4 = hidden_dim >> 2;
  int nthread = min(((hidden_dim_f4 + 31) / 32) * 32, MAX_THREADS);
  dim3 grid_dim(batch_size);
  dim3 block_dim(nthread);

  ker_layer_norm<float><<<grid_dim, block_dim, 0, stream>>>(
      d_ln_res, d_vars, d_means, d_inp, d_scale, d_bias, hidden_dim);

  // Copy back to the host
  cudaMemcpy(ln_res, d_ln_res, output_size, cudaMemcpyDeviceToHost);
  cudaMemcpy(vars, d_vars, var_size, cudaMemcpyDeviceToHost);
  cudaMemcpy(means, d_means, mean_size, cudaMemcpyDeviceToHost);
  cudaDeviceSynchronize();

  // Check CUDA execution
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    fprintf(stderr, "launch_layernorm Error: %s\n", cudaGetErrorString(err));
    // Handle the error (e.g., by exiting the program)
    exit(EXIT_FAILURE);
  }

  // Free memory on device
  cudaFree(d_ln_res);
  cudaFree(d_vars);
  cudaFree(d_means);
  cudaFree(d_inp);
  cudaFree(d_scale);
  cudaFree(d_bias);
}
}

/**
@brief: ker_ln_bw_dgamma_dbetta
Layer norm backword kernel, compute the gradient of gamma and betta.
dbetta = sum(dout, dim=0)
dgamma = sum(xhat * dout, dim=0)
xhat = (input - mean) * rsqrt(var) or
  (output - betta) / gamma

@thread
gridDim.x = hidden_size / 32
blockDim.x = 32
blockDim.y = 32

@param
gamma_grad: [hidden_size], gradient of gamma
betta_grad: [hidden_size], gradient of betta
out_grad: [batch_size * seq_len, hidden_size], gradient of betta ln output
inp_or_out: [batch_size * seq_len, hidden_size], ln output if means is nullptr
  ln input if means is not nullptr
gamma: [hidden_size], gamma of ln,
  used to compute xhat, maybe nullptr
betta: [hidden_size], betta of ln,
  used to compute xhat, maybe nullptr
vars: [batch_size * seq_len], variance of ln forward,
  used to compute xhat, maybe nullptr
means: [batch_size * seq_len], mean of ln forward,
  used to compute xhat, maybe nullptr
(gamma && betta) ^ (vars && means) should be true
*/
template <typename T>
__global__ void
ker_ln_bw_dgamma_dbetta(T *gamma_grad, T *betta_grad, const T *out_grad,
                        const T *inp, const T *gamma, const T *betta,
                        const T *vars, const T *means, int rows, int width) {

  /// BEGIN ASSIGN4_2_2
  /// TODO
  // Hints:
  // 1. Compute the partial gradients by looping across inp rows
  // 2. Store the partial gradients in the shared memory arrays
  // 3. Compute the reduce sum of the shared memory arrays with g.shfl_down
  //      -> More hints about `g.shfl_down`:
  //      ->
  //      https://developer.nvidia.com/blog/cooperative-groups/#:~:text=Using%20thread_block_tile%3A%3Ashfl_down()%20to%20simplify%20our%20warp%2Dlevel%20reduction%20does%20benefit%20our%20code%3A%20it%20simplifies%20it%20and%20eliminates%20the%20need%20for%20shared%20memory
  //      -> The highlighted line gives you a conceptual understanding of what
  //      the g.shfl_down is doing. Usually, the threads inside a block need to
  //      load everything to shared memory and work together to reduce the
  //      result (like what you have implemented in the hw1 for reduce
  //      function).
  //      -> Now g.shfl_down helps you do so without consuming any shared
  //      memory. g.shfl_down makes it more efficient.
  // 4. Assign the final result to the correct position in the global output

  __shared__ float gamma_buffer[TILE_DIM][TILE_DIM];
  __shared__ float betta_buffer[TILE_DIM][TILE_DIM];

  cg::thread_block b = cg::this_thread_block();
  cg::thread_block_tile<TILE_DIM> g = cg::tiled_partition<TILE_DIM>(b);

  float local_dgamma = 0.0f;
  float local_dbetta = 0.0f;

  int col_idx = blockIdx.x * TILE_DIM + threadIdx.x;

  if (col_idx < width) {
    for (int row_idx = threadIdx.y; row_idx < rows; row_idx += blockDim.y) {
      int offset = row_idx * width + col_idx;
      float dout = (float)out_grad[offset];

      float xhat;
      if (means) {
        xhat = ((float)inp[offset] - (float)means[row_idx]) *
               rsqrtf((float)vars[row_idx] + LN_EPSILON);
      } else {
        xhat = ((float)inp[offset] - (float)betta[col_idx]) /
               ((float)gamma[col_idx] + LN_EPSILON);
      }

      local_dbetta += dout;
      local_dgamma += xhat * dout;
    }
  }

  // Store partial sums in shared memory
  gamma_buffer[threadIdx.y][threadIdx.x] = local_dgamma;
  betta_buffer[threadIdx.y][threadIdx.x] = local_dbetta;
  __syncthreads();

  // Transpose read: now shfl_down reduces across the original y dimension
  float sum_dgamma = gamma_buffer[threadIdx.x][threadIdx.y];
  float sum_dbetta = betta_buffer[threadIdx.x][threadIdx.y];

  for (int offset = 16; offset > 0; offset /= 2) {
    sum_dgamma += g.shfl_down(sum_dgamma, offset);
    sum_dbetta += g.shfl_down(sum_dbetta, offset);
  }

  // threadIdx.x == 0 has the final sum; column is now indexed by threadIdx.y
  if (threadIdx.x == 0) {
    int final_col = blockIdx.x * TILE_DIM + threadIdx.y;
    if (final_col < width) {
      atomicAdd(&gamma_grad[final_col], sum_dgamma);
      atomicAdd(&betta_grad[final_col], sum_dbetta);
    }
  }
  // END ASSIGN4_2_2
}

/**
@brief: ker_ln_bw_dinp
Layer norm backword kernel, compute the gradient of input.
dinp = (dxhat - (sum(dxhat) + xhat * sum(dxhat * xhat)) / hidden_dim)
  * rsqrt(var)
xhat = (input - mean) * rsqrt(var) if mean is not nullptr
       (output - betta) / gamma if mean is nullptr
dxhat = dout * gamma


@thread
gridDim.x = batch_size * seq_len
blockDim.x = hidden_size

@param
inp_grad: [batch_size * seq_len, hidden_size], gradient of betta ln output
out_grad: [batch_size * seq_len, hidden_size], gradient of betta ln output
residual_grad: [batch_size * seq_len, hidden_size], gradient of residual input,
  usually appear in pre-layer-norm for transformer layer, maybe nullptr
inp_or_out: [batch_size * seq_len, hidden_size], ln output if means is nullptr
  ln input if means is not nullptr
gamma: [hidden_size], gamma of ln,
  used to compute xhat and dxhat
betta: [hidden_size], betta of ln,
  used to compute xhat, maybe nullptr
vars: [batch_size * seq_len], variance of ln forward,
  used to compute xhat and dinp
means: [batch_size * seq_len], mean of ln forward,
  used to compute xhat, maybe nullptr
*/
template <typename T>
__global__ void ker_ln_bw_dinp(T *inp_grad, const T *out_grad, const T *inp,
                               const T *gamma, const T *betta, const T *vars,
                               const T *means, int hidden_dim) {

  /// BEGIN ASSIGN4_2_2
  /// TODO
  // Hints:
  // 1. Compute dxhat=dy*w with reinterpret_cast by casting to float4 for
  // speedup
  // 2. Compute xhat with reinterpret_cast by casting to float4 for speedup
  // 3. Compute reduce sum for dxhat and dxhat*xhat with blockReduce
  // 4. Compute final gradient

  float l_sum_1 = 0; // sum(dxhat)
  float l_sum_2 = 0; // sum(dxhat * xhat)

  int row_idx = blockIdx.x;
  float var_val = vars[row_idx];
  float mean_val = (means) ? means[row_idx] : 0.0f;
  float inv_std = rsqrtf(var_val + LN_EPSILON);

  // Calculate local sums
  for (int col_idx = threadIdx.x; col_idx < hidden_dim; col_idx += blockDim.x) {
    int offset = row_idx * hidden_dim + col_idx;
    float val = (float)inp[offset];
    float dout = (float)out_grad[offset];
    float gam = (float)gamma[col_idx];
    float bet = (betta) ? (float)betta[col_idx] : 0.0f;

    float xhat;
    if (means) {
      xhat = (val - mean_val) * inv_std;
    } else {
      xhat = (val - bet) / (gam + LN_EPSILON);
    }

    float dxhat = dout * gam;
    l_sum_1 += dxhat;
    l_sum_2 += dxhat * xhat;
  }

  // Block reduce sums using shared memory reduction
  extern __shared__ float s_mem[];
  float *s_sum_1 = s_mem;
  float *s_sum_2 = &s_mem[blockDim.x];

  s_sum_1[threadIdx.x] = l_sum_1;
  s_sum_2[threadIdx.x] = l_sum_2;
  __syncthreads();

  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (threadIdx.x < stride) {
      s_sum_1[threadIdx.x] += s_sum_1[threadIdx.x + stride];
      s_sum_2[threadIdx.x] += s_sum_2[threadIdx.x + stride];
    }
    __syncthreads();
  }

  float sum_dxhat = s_sum_1[0];
  float sum_dxhat_xhat = s_sum_2[0];
  float feat_dim_inv = 1.0f / (float)(hidden_dim);

  // Calculate and assign final gradient
  for (int col_idx = threadIdx.x; col_idx < hidden_dim; col_idx += blockDim.x) {
    int offset = row_idx * hidden_dim + col_idx;
    float val = (float)inp[offset];
    float dout = (float)out_grad[offset];
    float gam = (float)gamma[col_idx];
    float bet = (betta) ? (float)betta[col_idx] : 0.0f;

    float xhat;
    if (means) {
      xhat = (val - mean_val) * inv_std;
    } else {
      xhat = (val - bet) / (gam + LN_EPSILON);
    }

    float dxhat = dout * gam;
    float dinp_val =
        (dxhat - (sum_dxhat + xhat * sum_dxhat_xhat) * feat_dim_inv) * inv_std;
    inp_grad[offset] = (T)dinp_val;
  }

  // END ASSIGN4_2_2
}
extern "C" {
void launch_layernorm_bw(float *gamma_grad, float *betta_grad, float *inp_grad,
                         const float *out_grad, const float *inp,
                         const float *gamma, const float *betta,
                         const float *vars, const float *means, int batch_size,
                         int hidden_dim, cudaStream_t stream_1,
                         cudaStream_t stream_2) {

  // Allocate device memory
  float *d_gamma_grad, *d_betta_grad, *d_inp_grad, *d_out_grad, *d_inp,
      *d_gamma, *d_betta, *d_vars, *d_means;
  int grad_output_size = batch_size * hidden_dim * sizeof(float);
  int gamma_betta_size = hidden_dim * sizeof(float);
  int vars_means_size = batch_size * sizeof(float);

  cudaMalloc((void **)&d_gamma_grad, gamma_betta_size);
  cudaMalloc((void **)&d_betta_grad, gamma_betta_size);
  cudaMalloc((void **)&d_inp_grad, grad_output_size);
  cudaMalloc((void **)&d_out_grad, grad_output_size);
  cudaMalloc((void **)&d_inp, grad_output_size);
  cudaMalloc((void **)&d_gamma, gamma_betta_size);
  cudaMalloc((void **)&d_betta, gamma_betta_size);
  cudaMalloc((void **)&d_vars, vars_means_size);
  cudaMalloc((void **)&d_means, vars_means_size);

  // Copy memory to device
  cudaMemcpy((void *)d_out_grad, out_grad, grad_output_size,
             cudaMemcpyHostToDevice);
  cudaMemcpy((void *)d_inp, inp, grad_output_size, cudaMemcpyHostToDevice);
  cudaMemcpy((void *)d_gamma, gamma, gamma_betta_size, cudaMemcpyHostToDevice);
  cudaMemcpy((void *)d_betta, betta, gamma_betta_size, cudaMemcpyHostToDevice);
  cudaMemcpy((void *)d_vars, vars, vars_means_size, cudaMemcpyHostToDevice);
  cudaMemcpy((void *)d_means, means, vars_means_size, cudaMemcpyHostToDevice);

  // Launch kernels
  // Compute grad of gamma and betta
  // This calculates the number of blocks needed to cover the data along the
  // specified dimension, rounds it up.
  dim3 grid_dim((hidden_dim + TILE_DIM - 1) / TILE_DIM);
  dim3 block_dim(TILE_DIM, TILE_DIM);
  ker_ln_bw_dgamma_dbetta<float><<<grid_dim, block_dim, 0, stream_1>>>(
      d_gamma_grad, d_betta_grad, d_out_grad, d_inp, d_gamma, d_betta, d_vars,
      d_means, batch_size, hidden_dim);

  if (hidden_dim > 4096) {
    throw std::runtime_error("hidden_dim > 4096");
  }
  int nthread = min(((hidden_dim + 31) / 32) * 32, MAX_THREADS);
  int shared_mem_size = 2 * nthread * sizeof(float);
  ker_ln_bw_dinp<<<batch_size, nthread, shared_mem_size, stream_2>>>(
      d_inp_grad, d_out_grad, d_inp, d_gamma, d_betta, d_vars, d_means,
      hidden_dim);

  // Synchronize and check for errors
  cudaDeviceSynchronize();
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    fprintf(stderr, "launch_layernorm_bw Error: %s\n", cudaGetErrorString(err));
    exit(EXIT_FAILURE);
  }

  // Copy back to host
  cudaMemcpy(gamma_grad, d_gamma_grad, gamma_betta_size,
             cudaMemcpyDeviceToHost);
  cudaMemcpy(betta_grad, d_betta_grad, gamma_betta_size,
             cudaMemcpyDeviceToHost);
  cudaMemcpy(inp_grad, d_inp_grad, grad_output_size, cudaMemcpyDeviceToHost);

  // Free device memory
  cudaFree(d_gamma_grad);
  cudaFree(d_betta_grad);
  cudaFree(d_inp_grad);
  cudaFree((void *)d_out_grad);
  cudaFree((void *)d_inp);
  cudaFree((void *)d_gamma);
  cudaFree((void *)d_betta);
  cudaFree((void *)d_vars);
  cudaFree((void *)d_means);
}
}
} // namespace cuda
} // namespace lightseq

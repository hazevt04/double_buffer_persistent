#include <stdio.h>

// From Robert Crovella on StackOverflow.com
// https://stackoverflow.com/questions/33150040/doubling-buffering-in-cuda-so-the-cpu-can-operate-on-data-produced-by-a-persiste/33158954#33158954
// with format cleanup for readability preference

constexpr int num_iterations = 1000;

constexpr size_t num_vals = 65536;

constexpr int threads_per_block = 256;

enum ready_status_e { not_full, full };

inline void show_ready_status( char* status_str, ready_status_e status ) {
   int status_index = static_cast<int>(status);
   const char *status_strings[] = {
      "Not Full",
      "Full"
   };
   if ((status_str != nullptr)) {
      if ( (status_index > 0) && (status_index < 2)) {
         strncpy( status_str, status_strings[status_index], sizeof(status_strings[status_index]) );
      } else {
         printf( "%s(): ERROR: status, %d was invalid\n", __func__, (int)status );
         strncpy( status_str, "Invalid Status", 14 );
      }
   } else {
      printf( "%s(): ERROR: status_str pointer was somehow nullptr\n", __func__ );
      return;
   }

}

#define cudaCheckErrors(msg) \
    do { \
        cudaError_t __err = cudaGetLastError(); \
        if (__err != cudaSuccess) { \
            fprintf(stderr, "Fatal error: %s (%s at %s:%d)\n", \
                msg, cudaGetErrorString(__err), \
                __FILE__, __LINE__); \
            fprintf(stderr, "*** FAILED - ABORTING\n"); \
            exit(1); \
        } \
    } while (0)


__device__ volatile int d_block_num1 = 0;
__device__ volatile int d_block_num2 = 0;
__device__ volatile int d_iteration_num = 0;

__device__ void my_compute_function(int *buffer, int idx, int data){
   buffer[idx] = data;  // put your work code here
}

inline
__host__ __device__
bool is_odd( int val ) { return val & 1; }

__global__ void testkernel(int *buffer1, int *buffer2, volatile ready_status_e *buffer1_ready_status, 
      volatile ready_status_e *buffer2_ready_status,  const int buffer_size, const int num_iterations) {

   // assumption of persistent block-limited kernel launch
   int idx = threadIdx.x+blockDim.x*blockIdx.x;
   int iteration_num = 0;
   
   // persistent until iterations complete
   while (iteration_num < num_iterations) { 
      // ping pong between buffers
      int *current_buffer = (iteration_num & 1)? buffer2:buffer1; 
      volatile ready_status_e *current_buffer_ready_status = (is_odd(iteration_num)) ? (buffer2_ready_status) : (buffer1_ready_status);
      volatile int *current_d_block_num = (is_odd(iteration_num)) ? (&d_block_num2) : (&d_block_num1);
      int my_idx = idx;
      
      // don't overrun buffers on device
      while (iteration_num - d_iteration_num > 1); 
      
      // wait for buffer to be consumed
      while (*current_buffer_ready_status == ready_status_e::full);  
      
      // perform the "work"
      while (my_idx < buffer_size) { 
         my_compute_function(current_buffer, my_idx, iteration_num);
         my_idx += gridDim.x*blockDim.x; // grid-striding loop
      }
      __syncthreads(); // wait for my block to finish
      __threadfence(); // make sure global buffer writes are "visible"

      if (!threadIdx.x) atomicAdd((int *)current_d_block_num, 1); // mark my block done
      
      if (!idx) { // am I the main block/thread?
         while (*current_d_block_num < gridDim.x);  // wait for all blocks to finish

         *current_d_block_num = 0;
         *current_buffer_ready_status = ready_status_e::full;  // indicate that buffer is ready
         __threadfence_system(); // push it out to mapped memory
         d_iteration_num++;
      }
      iteration_num++;
   } // end of while (iteration_num < num_iterations ) { // persistent until num_iterations complete
}


bool validate(const int *actual_vals, const int num_vals, const int expected_val) {
   for (int val_num = 0; val_num < num_vals; ++val_num) {
      if (actual_vals[val_num] != expected_val) {
         printf("mismatch at %d, was: %d, should be: %d\n", val_num, actual_vals[val_num], expected_val); 
         return false;
      }
   }
   return true;
} // end of bool validate(const int *data, const int dsize, const int expected){



int main(){

   int *h_buffer1, *d_buffer1, *h_buffer2, *d_buffer2;
   volatile ready_status_e *buffer1_ready_status, *buffer2_ready_status;
   
   // buffer and "mailbox" setup
   cudaHostAlloc(&h_buffer1, num_vals*sizeof(int), cudaHostAllocDefault);
   cudaCheckErrors("cudaHostAlloc failed for h_buffer1");
   cudaHostAlloc(&h_buffer2, num_vals*sizeof(int), cudaHostAllocDefault);
   cudaCheckErrors("cudaHostAlloc failed for h_buffer2");
   cudaHostAlloc(&buffer1_ready_status, sizeof(int), cudaHostAllocMapped);
   cudaCheckErrors("cudaHostAlloc failed for buffer1_ready_status");
   cudaHostAlloc(&buffer2_ready_status, sizeof(int), cudaHostAllocMapped);
   cudaCheckErrors("cudaHostAlloc failed for buffer2_ready_status");

   cudaMalloc(&d_buffer1, num_vals*sizeof(int));
   cudaCheckErrors("cudaMalloc failed for d_buffer1");
   cudaMalloc(&d_buffer2, num_vals*sizeof(int));
   cudaCheckErrors("cudaMalloc failed for d_buffer2");
   
   cudaStream_t streamk, streamc;
   cudaStreamCreate(&streamk);
   cudaCheckErrors("cudaStreamCreate failed for streamk");
   cudaStreamCreate(&streamc);
   cudaCheckErrors("cudaStreamCreate failed for streamc");

   *buffer1_ready_status = ready_status_e::not_full;
   *buffer2_ready_status = ready_status_e::not_full;
   cudaMemset(d_buffer1, 0xFF, num_vals*sizeof(int));
   cudaCheckErrors("cudaMemset (to 0xFF) failed for d_buffer1");
   cudaMemset(d_buffer2, 0xFF, num_vals*sizeof(int));
   cudaCheckErrors("cudaMemset (to 0xFF) failed for d_buffer2");
   
   // inefficient crutch for choosing number of blocks
   int num_blocks = 0;
   cudaDeviceGetAttribute(&num_blocks, cudaDevAttrMultiProcessorCount, 0);
   cudaCheckErrors("get multiprocessor count failed");
   
   testkernel<<<num_blocks, threads_per_block, 0, streamk>>>(d_buffer1, d_buffer2, 
      buffer1_ready_status, buffer2_ready_status, num_vals, num_iterations);
   cudaCheckErrors("testkernel launch failed");
   
   volatile ready_status_e *current_buffer_ready_status;
   int *h_current_buffer, *d_current_buffer;
   
   for (int iteration_num = 0; iteration_num < num_iterations; ++iteration_num) {
      if (is_odd(iteration_num)) {  // ping pong on the host side
         current_buffer_ready_status = buffer2_ready_status;
         h_current_buffer = h_buffer2;
         d_current_buffer = d_buffer2;
      } else {
         current_buffer_ready_status = buffer1_ready_status;
         h_current_buffer = h_buffer1;
         d_current_buffer = d_buffer1;
      }
      
      // int qq = 0; // add for failsafe - otherwise a machine failure can hang
      while ((*current_buffer_ready_status)!= ready_status_e::full); 
      // use this for a failsafe:  
      // if (++qq > 1000000) {
      //    printf("current_buffer_ready_status = %d\n", *current_buffer_ready_status);
      //    return 0;
      // } // wait for buffer to be full;
      cudaMemcpyAsync(h_current_buffer, d_current_buffer, num_vals*sizeof(int), cudaMemcpyDeviceToHost, streamc);
      
      cudaStreamSynchronize(streamc);
      cudaCheckErrors("cudaMemcpyAsync failed for d_current_buffer to h_current_buffer");
      
      *current_buffer_ready_status = ready_status_e::not_full; // release buffer back to device
      if (!validate(h_current_buffer, num_vals, iteration_num)) {
         printf("validation of h_current_buffer failed at iter %d\n", iteration_num); 
         exit(1);
      }
   } // end of for (int iteration_num = 0; iteration_num < num_iterations; ++iteration_num) {

   printf("Completed %d iterations successfully\n", num_iterations);
}

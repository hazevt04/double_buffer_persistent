#pragma once

#include "my_cuda_utils.hpp"

/////////////////////////////
// CUFFT Stuff
/////////////////////////////

// Returns string based on the cuffResult value returned by a CUFFT call
// Why doesnt CUFFT already have something like this in the API?
//char const* get_cufft_status_msg(const cufftResult cufft_status);
inline char const* status_strings[] = {
   "The cuFFT operation was successful\n",
   "cuFFT was passed an invalid plan handle\n",
   "cuFFT failed to allocate GPU or CPU memory\n",
   "No longer used\n",
   "User specified an invalid pointer or parameter\n",
   "Driver or internal cuFFT library error\n",
   "Failed to execute an FFT on the GPU\n",
   "The cuFFT library failed to initialize\n",
   "User specified an invalid transform size\n",
   "No longer used\n",
   "Missing parameters in call\n",
   "Execution of a plan was on different GPU than plan creation\n",
   "Internal plan database error\n",
   "No workspace has been provided prior to plan execution\n",
   "Function does not implement functionality for parameters given.\n",
   "Used in previous versions.\n",
   "Operation is not supported for parameters given.\n" };

   if ( cufft_status < 16 ) {
      return status_strings[cufft_status];
   }
   return "Unknown cufftResult value\n";
}

#define check_cufft_status(cufft_status) { \
  if ( cufft_status != CUFFT_SUCCESS ) { \
    printf( "%s(): ERROR: %s\n", __func__, \
      get_cufft_status_msg( cufft_status ) ); \
    exit(EXIT_FAILURE); \
  } \
}

#define check_cufft_status_error_flag(cufft_status) { \
  if ( cufft_status != CUFFT_SUCCESS ) { \
    printf( "%s(): ERROR: %s\n", __func__, \
      get_cufft_status_msg( cufft_status ) ); \
    error_flag = true; \
    return FAILURE; \
  } \
}

#define check_cufft_status_return(cufft_status) { \
  if ( cufft_status != CUFFT_SUCCESS ) { \
    printf( "%s(): ERROR: %s\n", __func__, \
      get_cufft_status_msg( cufft_status ) ); \
    return FAILURE; \
  } \
}

#define check_cufft_status_throw(cufft_status) { \
  if ( cufft_status != CUFFT_SUCCESS ) { \
   throw std::runtime_error( get_cufft_status_msg( cufft_status ) ); \
  } \
}

#define try_cufft_func(cufft_status, func) { \
  cufft_status = func; \
  check_cufft_status( cufft_status ); \
}

#define try_cufft_func_error_flag(cufft_status, func) { \
  cufft_status = func; \
  check_cufft_status_error_flag( cufft_status ); \
}

#define try_cufft_func_return(cufft_status, func) { \
  cufft_status = func; \
  check_cufft_status_return( cufft_status ); \
}

#define try_cufft_func_throw(cufft_status, func) { \
  cufft_status = func; \
  check_cufft_status_throw( cufft_status ); \
}


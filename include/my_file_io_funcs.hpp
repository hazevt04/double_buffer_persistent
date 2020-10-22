#pragma once

#include "my_utils.hpp"
#include <string.h>
#include <sstream>
#include <fstream>

void write_binary_floats_file( float* vals, const char* filename, const int num_vals, const bool debug );

void check_file_size( long long& file_size, const char* filename, const bool debug );

void read_binary_floats_file( float* vals, const char* filename, const int num_vals, const bool debug );

void test_my_file_io_funcs( std::string filename, const int num_vals, const bool inject_error, const bool debug );


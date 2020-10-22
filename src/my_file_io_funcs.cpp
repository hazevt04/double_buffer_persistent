// C++ File for my_file_io_funcs

#include "my_file_io_funcs.hpp"

void write_binary_floats_file( float* vals, const char* filename, const int num_vals, 
      const bool debug = false ) {

   std::ofstream ofile;
   int val_num = 0;
   std::streampos file_size;
   std::ostringstream  err_msg_stream( std::ostringstream::ate );
   try {
      ofile.open( filename, std::ios::out | std::ios::binary );
      if ( ofile.is_open() ) {
         std::streamsize val_size = num_vals * sizeof(float);
         debug_printf( debug, "%s(): Val size is %zu bytes\n\n", __func__, val_size );
         ofile.write( (char*)vals, val_size );
      } else {
         err_msg_stream << "Unable to open file, " << filename << ", for writing.";
         throw std::runtime_error(err_msg_stream.str());
      }
      ofile.close();
      if ( ( ofile.rdstate() & std::ofstream::failbit ) != 0 ) {
         err_msg_stream << "Logical error while writing file, " << filename << ".";
         throw std::runtime_error(err_msg_stream.str());
      }
      if ( ( ofile.rdstate() & std::ofstream::badbit ) != 0 ) {
         err_msg_stream << "Write error while writing file, " << filename << ".";
         throw std::runtime_error(err_msg_stream.str());
      }
   } catch( std::exception& ex ) {
      printf( "Error: %s\n", ex.what() );
   }
}


void check_file_size( long long& file_size, const char* filename, const bool debug = false) {
   
   std::ifstream ifile;
   std::ostringstream  err_msg_stream( std::ostringstream::ate );
   try {
      ifile.open( filename, std::ios::in | std::ios::binary );
      if ( ifile.is_open() ) {
         // get length of file:
         ifile.seekg (0, ifile.end);
         file_size = (long long)ifile.tellg();
         ifile.seekg (0, ifile.beg);
         debug_printf( debug, "%s(): File size for %s is %llu bytes\n\n", __func__, filename, file_size ); 
      } else {
         err_msg_stream << "Unable to open file, " << filename << ", for checking filesize.";
         throw std::runtime_error(err_msg_stream.str());
      }
      ifile.close();
   } catch( std::exception& ex ) {
      printf( "Error: %s\n", ex.what() );
      if ( ifile.is_open() ) {
         ifile.close();
      }
      exit(EXIT_FAILURE);
   } // end of catch
   
}


void read_binary_floats_file( float* vals, const char* filename, const int num_vals, const bool debug = false ) {
   
   std::ifstream ifile;
   int val_num = 0;
   std::streampos file_size;
   std::ostringstream  err_msg_stream( std::ostringstream::ate );
   try {
      ifile.open( filename, std::ios::in | std::ios::binary );
      if ( ifile.is_open() ) {
         size_t val_size = num_vals * sizeof(float);
         debug_printf( debug, "%s(): Val size is %zu bytes\n", __func__, val_size ); 
         ifile.seekg (0, ifile.end);
         long long file_size = (long long)ifile.tellg();
         ifile.seekg (0, ifile.beg);
         debug_printf( debug, "%s(): File size is %llu bytes\n\n", __func__, (long long)file_size ); 
         if ( file_size < val_size ) {
            err_msg_stream << "Expected file size, " << file_size << " bytes, less than expected: "
               << val_size << " bytes, for file " << filename << ".";
            throw std::runtime_error(err_msg_stream.str());
         }
         ifile.read( (char*)vals, file_size );
         ifile.close();
      } else {
         err_msg_stream << "Unable to open file: " << filename << ".";
         throw std::runtime_error(err_msg_stream.str());
      } // end of if ( ifile.is_open() ) {
   } catch( std::exception& ex ) {
      printf( "Error: %s\n", ex.what() ); 
      if ( ifile.is_open() ) {
         ifile.close();
      }
      exit(EXIT_FAILURE);
   }
}


void test_my_file_io_funcs( std::string filename, const int num_vals, const bool inject_error, const bool debug ) {
   try {
      long long file_size = 0;

      float* write_vals = new float[ num_vals * sizeof( float ) ];
      float* read_vals = new float[ num_vals * sizeof( float ) ];
      std::ostringstream err_msg_stream( std::ostringstream::ate );
      
      gen_vals<float>( write_vals, 100, 0, num_vals );
      printf( "Write Values:\n" );
      printf_floats( write_vals, num_vals );

      debug_printf( debug, "The input text file is %s\n", filename.c_str() ); 

      write_binary_floats_file( write_vals, filename.c_str(), num_vals, debug );

      check_file_size( file_size, filename.c_str(), debug ); 

      if ( inject_error ) {
         filename = "wrong_file.bin";
      }

      read_binary_floats_file( read_vals, filename.c_str(), num_vals, debug ); 
      
      printf( "Read Values:\n" );
      printf_floats( read_vals, num_vals );

      int num_mismatches = 0;
      if ( !compare_floats( read_vals, write_vals, num_vals ) ) {
         err_msg_stream <<  "Values read from " << filename 
            << " don't match values written";      
         throw std::runtime_error( err_msg_stream.str() );
      } else {
         printf( "All %d values read from %s matched the values written\n", 
               num_vals, filename.c_str() );
      }

   } catch ( std::exception& ex ) {
      printf( "ERROR: %s\n", ex.what() ); 
   }

}


// end of C++ file for my_file_io_funcs

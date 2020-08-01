#include <albp/page_locked_string.hpp>
#include <albp/read_fasta.hpp>

namespace albp {

struct PageLockedFasta {
  PageLockedFasta(const size_t sequence_count, const size_t sequence_bytes);
  size_t sequence_count;                //< Number of sequences
  size_t maximum_sequence_length;       //< Length of the longest sequence
  PageLockedString sequences;           //< Compressed sequences
  std::vector<std::string> headers;     //< Headers for the sequences (not sent to GPU)
  PageLockedString modifiers;           //< Modifiers for the sequences
  size_t *starts;                       //< Starting indices of each sequence
  size_t *ends;                         //< Ending indices of each sequence
  size_t *sizes;                        //< Lengths of the sequences
};



struct PageLockedFastaPair {
  PageLockedFasta a;                    //< Sequences from the first file
  PageLockedFasta b;                    //< Sequences from the second file
  ///@brief How many cells need to be calculated if the sequences are compared in pairs
  uint64_t total_cells_1_to_1;
  ///@brief Number of sequences (each file has the same number)
  size_t sequence_count;
};

PageLockedFasta     page_lock(const FastaInput &inp);
PageLockedFastaPair page_lock(const FastaPair &fp);



// struct FastaInput {
//   std::vector<std::string> sequences; //< Sequences in the file
//   std::vector<std::string> headers;   //< Names/sequence identifiers
//   std::vector<uint8_t>     modifiers; //< Character starting the new sequence, usually '>'
//   size_t maximum_sequence_length = 0; //< Length of the longest sequence in the file
//   size_t total_sequence_bytes    = 0; //< Number of bytes in all the sequences combined
//   size_t sequence_count() const;      //< Number of sequences
// };

// struct FastaPair {
//   FastaInput a;                        //< Sequences from the first file
//   FastaInput b;                        //< Sequences from the second file
//   ///@brief Calculates how many cells need to be calculated if the sequences are compared in pairs
//   uint64_t total_cells_1_to_1() const;
//   ///@brief Returns the number of sequences (each file has the same number)
//   size_t sequence_count() const;
// };








}
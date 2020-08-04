#pragma once

#include <albp/memory.hpp>
#include <albp/page_locked_string.hpp>
#include <albp/ranges.hpp>
#include <albp/read_fasta.hpp>

namespace albp {

struct PageLockedFasta {
  PageLockedFasta(const size_t sequence_count, const size_t sequence_bytes);
  size_t sequence_count;                  //< Number of sequences
  size_t maximum_sequence_length;         //< Length of the longest sequence
  PageLockedString sequences;             //< Compressed sequences
  std::vector<std::string> headers;       //< Headers for the sequences (not sent to GPU)
  PageLockedString modifiers;             //< Modifiers for the sequences

  ///Starting indices of each sequence (1 longer than sequence count so
  ///a[i+1]-a[i] is always the length of a given sequence)
  cuda_unique_hptr<size_t> starts;
  cuda_unique_hptr<size_t> ends;          //< Ending indices of each sequence
  cuda_unique_hptr<size_t> sizes;         //< Lengths of the sequences

  char *       seq_begin(size_t i);       //< Pointer to beginning of Sequence i in `sequences`
  const char * seq_begin(size_t i) const; //< Pointer to beginning of Sequence i in `sequences`
  char *       seq_end  (size_t i);       //< Pointer to 1 past end of Sequence i in `sequences`
  const char * seq_end  (size_t i) const; //< Pointer to 1 past end of Sequence i in `sequences`
  size_t       total_bytes() const;       //< Number of bytes in all sequences
  size_t       bytes_between(size_t a, size_t b) const;  //< Bytes in `sequences` between the beginning of sequence a and beginning of sequence b
  size_t       bytes_between(const RangePair &rp) const; //< Bytes in `sequences` between the beginnings of the two sequences identified by the pair
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

///@brief Given a page-locked fasta, return the length of the longest string
///
///@param[in] pl_fasta A page-locked fasta
///
///@returns The length of the longest string
size_t get_max_length(const PageLockedFasta &pl_fasta);

///@brief Given a page-locked fasta, return the length of the longest string
///
///@param[in] vector_of_strings A page-locked fasta
///@param[in] range Range of strings to consider
///
///@returns The length of the longest string
size_t get_max_length(const PageLockedFasta &pl_fasta, const RangePair range);

///Copies the sequences indicated by `rp` from the PageLockedFasta to the beginning of the device vector
void copy_sequences_to_device_async(char *dev_ptr, const PageLockedFasta &pl_fasta, const RangePair &rp, const cudaStream_t stream);

///Copies the sequences indicated by `rp` from the PageLockedFasta to the beginning of the device vector
void copy_sequences_to_device_async(const cuda_unique_dptr<char> &dev_ptr, const PageLockedFasta &pl_fasta, const RangePair &rp, const cudaStream_t stream);

}
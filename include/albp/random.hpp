#pragma once

#include <array>
#include <random>
#include <string>

namespace albp {

template<class PRNG>
std::string random_dna_sequence(const size_t len, PRNG &gen){
  constexpr const std::array<char,4> alphabet = {'A','C','G','T'};
  std::uniform_int_distribution<int> dist(0,3);

  std::string ret;
  for(size_t n=0;n<len;n++){
    ret += alphabet[dist(gen)];
  }

  return ret;
}

}
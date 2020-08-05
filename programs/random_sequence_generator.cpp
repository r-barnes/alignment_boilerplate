#include <albp/cli_options.hpp>

#include <fstream>
#include <iostream>
#include <random>

int main(int argc, char **argv){
  CLI::App app("Random Sequence Generator");

  std::string output_fasta_filename;
  size_t num;
  size_t len;
  app.add_option("filename", output_fasta_filename, "Output FASTA Filename")->required();
  app.add_option("num", num, "Number of sequences")->required();
  app.add_option("len", len, "Length of sequences")->required();

  CLI11_PARSE(app, argc, argv);

  std::mt19937 gen;
  std::uniform_int_distribution<int> dist(0,3);

  std::ofstream fout(output_fasta_filename);

  const std::array<char,4> alphabet = {'A','C','G','T'};

  for(size_t n=0;n<num;n++){
    if(n%100000==0){
      std::cerr<<n<<"/"<<num<<std::endl;
    }
    fout<<">Seq #"<<n<<"\n";
    for(size_t l=0;l<len;l++)
      fout<<alphabet[dist(gen)];
    fout<<"\n";
  }

  return 0;
}
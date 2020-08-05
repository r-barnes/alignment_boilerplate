#include <albp/doctest.hpp>
#include <albp/reordering.hpp>

#include <algorithm>
#include <random>
#include <vector>



// TEST_CASE("sorted_ordering"){
//   std::mt19937 gen;
//   std::uniform_int_distribution<short> len_dist(0,std::numeric_limits<short>::max());
//   std::vector<short> data;

//   for(int i=0;i<3000;i++){
//     const int len = len_dist(gen);
//     data.clear();
//     for(int i=0;i<len;i++){
//       data.push_back(i);
//     }

//     //Shuffle the data
//     std::shuffle(data.begin(), data.end(), gen);

//     //Get the sorted ordering
//     const auto comp = [&](const size_t ai, const size_t bi){
//       return data[ai]<data[bi];
//     };

//     const auto ordering = albp::sorted_ordering(data.size(), comp);

//     albp::forward_reorder(data, ordering);

//     CHECK(std::is_sorted(data.begin(), data.end()));
//   }
// }



TEST_CASE("reorder"){
  std::mt19937 gen;
  std::uniform_int_distribution<short> value_dist(0,std::numeric_limits<short>::max());
  std::vector<short> data;
  std::vector<short> ordering;

  for(int i=0;i<3000;i++){
    const int len = value_dist(gen);
    data.clear();
    ordering.clear();
    for(int i=0;i<len;i++){
      data.push_back(value_dist(gen));
      ordering.push_back(i);
    }

    const auto original = data;

    std::shuffle(ordering.begin(), ordering.end(), gen);

    albp::forward_reorder(data, ordering);
    albp::backward_reorder(data.data(), ordering);

    CHECK(original==data);
  }
}

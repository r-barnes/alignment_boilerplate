#include <albp/doctest.hpp>
#include <albp/doctest_vectors.hpp>
#include <albp/simple_sw.hpp>

using namespace albp;

TEST_CASE("Matrix Init"){
  Matrix2D<int> mat = {
    {1,  2,  3,  4},
    {5,  6,  7,  8},
    {9, 10, 11, 12}
  };

  CHECK(mat.width  == 4);
  CHECK(mat.height == 3);
  CHECK(mat(0,0) ==  1);
  CHECK(mat(1,2) ==  7);
  CHECK(mat(2,1) == 10);
  CHECK(mat(2,3) == 12);

  CHECK(extract_diagonal(mat, -2)==std::vector<int>{});
  CHECK(extract_diagonal(mat, -1)==std::vector<int>{});
  CHECK(extract_diagonal(mat, 0)==std::vector<int>{1});
  CHECK(extract_diagonal(mat, 1)==std::vector<int>{5,2});
  CHECK(extract_diagonal(mat, 2)==std::vector<int>{9,6,3});
  CHECK(extract_diagonal(mat, 3)==std::vector<int>{10,7,4});
  CHECK(extract_diagonal(mat, 4)==std::vector<int>{11,8});
  CHECK(extract_diagonal(mat, 5)==std::vector<int>{12});
  CHECK_THROWS_AS(extract_diagonal(mat, 6), std::runtime_error);
}
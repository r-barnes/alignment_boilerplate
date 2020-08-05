#include <albp/doctest.hpp>
#include <albp/page_locked_string.hpp>

#include <stdexcept>

using namespace albp;

TEST_CASE("PageLockedString"){
  PageLockedString pls(20);
  CHECK(pls.capacity()==20);
  CHECK(pls.empty());
  CHECK(!pls.full());
  CHECK(pls.size()==0);
  CHECK(pls.size_left()==20);
  CHECK(pls.str()=="");

  pls += "testing";
  CHECK(pls.capacity()==20);
  CHECK(!pls.empty());
  CHECK(!pls.full());
  CHECK(pls.size()==7);
  CHECK(pls.size_left()==13);
  CHECK(pls.str()=="testing");

  pls += "andagain";
  CHECK(pls.capacity()==20);
  CHECK(!pls.empty());
  CHECK(!pls.full());
  CHECK(pls.size()==15);
  CHECK(pls.size_left()==5);
  CHECK(pls.str()=="testingandagain");

  CHECK_THROWS_AS(pls += "toolong", std::runtime_error);
  CHECK(pls.capacity()==20);
  CHECK(!pls.empty());
  CHECK(!pls.full());
  CHECK(pls.size()==15);
  CHECK(pls.size_left()==5);
  CHECK(pls.str()=="testingandagain");

  pls += "done.";
  CHECK(pls.capacity()==20);
  CHECK(!pls.empty());
  CHECK(pls.full());
  CHECK(pls.size()==20);
  CHECK(pls.size_left()==0);
  CHECK(pls.str()=="testingandagaindone.");
  CHECK(pls[2]=='s');

  pls.clear();
  CHECK(pls.capacity()==20);
  CHECK(pls.empty());
  CHECK(!pls.full());
  CHECK(pls.size()==0);
  CHECK(pls.size_left()==20);
  CHECK(pls.str()=="");

  pls += 'a';
  CHECK(pls.capacity()==20);
  CHECK(!pls.empty());
  CHECK(!pls.full());
  CHECK(pls.size()==1);
  CHECK(pls.size_left()==19);
  CHECK(pls.str()=="a");
}
#include <albp/doctest.hpp>
#include <albp/timer.hpp>

#include <chrono>
#include <thread>

using namespace albp;

TEST_CASE("Timer Start Stop"){
  Timer timer;
  timer.start();
  std::this_thread::sleep_for(std::chrono::milliseconds(200));
  timer.stop();

  CHECK(std::abs(timer.getSeconds()-0.2)<0.01);

  timer.start();
  std::this_thread::sleep_for(std::chrono::milliseconds(200));
  timer.stop();

  CHECK(std::abs(timer.getSeconds()-0.4)<0.02);

  timer.restart();
  std::this_thread::sleep_for(std::chrono::milliseconds(200));
  timer.stop();

  CHECK(std::abs(timer.getSeconds()-0.2)<0.01);

  timer.clear();
  CHECK(timer.getSeconds()==0);
}

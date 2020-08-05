cmake_minimum_required(VERSION 3.13)

project(albp LANGUAGES CUDA CXX)

find_package(OpenMP REQUIRED)
find_package(CUDA REQUIRED)

#TODO: Probably remove this before release since a standard user won't recompile
#many times
find_program(CCACHE_FOUND ccache)
if(CCACHE_FOUND)
    set_property(GLOBAL PROPERTY RULE_LAUNCH_COMPILE ccache)
    set_property(GLOBAL PROPERTY RULE_LAUNCH_LINK ccache) # Less useful to do it for linking, see edit2
endif(CCACHE_FOUND)

if(MSVC)
  add_compile_options(/W4 /WX)
else()
  add_compile_options(
    $<$<COMPILE_LANGUAGE:CXX>:-Wall>
    $<$<COMPILE_LANGUAGE:CXX>:-pedantic>
    $<$<COMPILE_LANGUAGE:CXX>:-Wextra>
  )
endif()

add_library(albp
  src/page_locked_fasta.cpp
  src/page_locked_string.cpp
  src/ranges.cpp
  src/read_fasta.cpp
  src/simple_sw.cpp
  src/stream_manager.cpp
)
target_include_directories(albp PUBLIC include ${CUDA_INCLUDE_DIRS})
target_link_libraries(albp PUBLIC ${CUDA_LIBRARIES})
# set_target_properties(gasal2 PROPERTIES CUDA_RESOLVE_DEVICE_SYMBOLS ON)
# set_target_properties(gasal2 PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

if(DEFINED GPU_SM_ARCH)
  target_compile_options(albp PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-arch ${GPU_SM_ARCH}>)
endif()

if(DEFINED GPU_COMPUTE_ARCH)
  target_compile_options(albp PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-arch ${GPU_COMPUTE_ARCH}>)
endif()

set_target_properties(albp PROPERTIES
  POSITION_INDEPENDENT_CODE ON
  CXX_STANDARD 14
  CXX_STANDARD_REQUIRED YES
)



# add_executable(test_prog test_prog/test_prog.cpp)

# target_link_libraries(test_prog PRIVATE OpenMP::OpenMP_CXX gasal2)

# file(COPY test_prog/run_tests.sh DESTINATION ${CMAKE_BINARY_DIR} FILE_PERMISSIONS OWNER_READ OWNER_WRITE OWNER_EXECUTE)

execute_process(
  COMMAND ${CMAKE_COMMAND} -E tar xzf ${CMAKE_CURRENT_SOURCE_DIR}/test_data/query_batch.fasta.tar.gz
  WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/
)

execute_process(
  COMMAND ${CMAKE_COMMAND} -E tar xzf ${CMAKE_CURRENT_SOURCE_DIR}/test_data/target_batch.fasta.tar.gz
  WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/
)



add_executable(unittests_albp
  unittests/test_main.cpp
  unittests/test_page_locked_fasta.cu
  unittests/test_page_locked_string.cu
  unittests/test_ranges.cpp
  unittests/test_read_fasta.cpp
  unittests/test_reordering.cpp
  unittests/test_simple_sw.cpp
  unittests/test_timer.cpp
)
target_link_libraries(unittests_albp PRIVATE albp)

add_executable(random_sequence_generator
  programs/random_sequence_generator.cpp
)
target_link_libraries(random_sequence_generator PUBLIC albp)

# configure_file(unittests/test.fasta ${CMAKE_BINARY_DIR}/test.fasta COPYONLY)
# configure_file(unittests/test_target.fasta ${CMAKE_BINARY_DIR}/test_target.fasta COPYONLY)

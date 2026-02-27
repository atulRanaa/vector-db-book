cmake_minimum_required(VERSION 3.14)
project(VectorDBBook_Advanced_C++)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find Apache Arrow and Parquet
find_package(Arrow REQUIRED)
find_package(Parquet REQUIRED)
find_package(ArrowDataset REQUIRED)

add_executable(test_vector_db 
    test_vector_db.cpp 
    ../distances.cpp)

target_include_directories(test_vector_db PRIVATE ..)
target_link_libraries(test_vector_db PRIVATE arrow_shared parquet_shared arrow_dataset_shared)

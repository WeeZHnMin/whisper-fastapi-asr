cmake_minimum_required(VERSION 3.16)
project(asr_minimal CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Keep the dependency build minimal for this demo app.
set(WHISPER_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(WHISPER_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(WHISPER_BUILD_SERVER OFF CACHE BOOL "" FORCE)

add_subdirectory(${CMAKE_CURRENT_LIST_DIR}/../whisper.cpp ${CMAKE_CURRENT_BINARY_DIR}/whispercpp EXCLUDE_FROM_ALL)

add_executable(asr_minimal main.cpp)
target_link_libraries(asr_minimal PRIVATE whisper)
target_include_directories(asr_minimal PRIVATE ${CMAKE_CURRENT_LIST_DIR}/../whisper.cpp/include)

# For MSVC multi-config builds, copy runtime DLLs next to the demo executable.
add_custom_command(TARGET asr_minimal POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different $<TARGET_FILE:whisper> $<TARGET_FILE_DIR:asr_minimal>
    COMMAND ${CMAKE_COMMAND} -E copy_if_different $<TARGET_FILE:ggml> $<TARGET_FILE_DIR:asr_minimal>
    COMMAND ${CMAKE_COMMAND} -E copy_if_different $<TARGET_FILE:ggml-base> $<TARGET_FILE_DIR:asr_minimal>
    COMMAND ${CMAKE_COMMAND} -E copy_if_different $<TARGET_FILE:ggml-cpu> $<TARGET_FILE_DIR:asr_minimal>
)

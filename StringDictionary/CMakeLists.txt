cmake_minimum_required(VERSION 2.8)
#set(CMAKE_CXX_FLAGS "-std=c++11 -Wall -Wno-attributes -O3")

add_library(StringDictionary StringDictionary.cpp StringDictionaryProxy.cpp)
target_link_libraries(StringDictionary Utils ${Glog_LIBRARIES})

set(string_dictionary_service_thrift_files
    ${CMAKE_CURRENT_BINARY_DIR}/gen-cpp/RemoteStringDictionary.cpp
    ${CMAKE_CURRENT_BINARY_DIR}/gen-cpp/string_dictionary_constants.cpp
    ${CMAKE_CURRENT_BINARY_DIR}/gen-cpp/string_dictionary_types.cpp)

add_custom_command(
    DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/string_dictionary.thrift
    OUTPUT ${string_dictionary_service_thrift_files}
    COMMAND ${Thrift_EXECUTABLE}
    ARGS -gen cpp -o ${CMAKE_CURRENT_BINARY_DIR} ${CMAKE_CURRENT_SOURCE_DIR}/string_dictionary.thrift)
list(APPEND ADDITIONAL_MAKE_CLEAN_FILES ${CMAKE_CURRENT_BINARY_DIR}/gen-cpp/)
include_directories(${CMAKE_CURRENT_BINARY_DIR})

add_executable(StringDictionaryServer StringDictionaryServer.cpp ${string_dictionary_service_thrift_files})
target_link_libraries(StringDictionaryServer StringDictionary ${Thrift_LIBRARIES})

#install(TARGETS StringDictionaryServer DESTINATION bin)

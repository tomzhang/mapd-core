set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Dlinux -DSIZEOF_LONG_INT=8")

set(odbc_source_files
    Common/QSUtilities_Unix.cpp
    Core/QSConnection.cpp
    Core/QSDriver.cpp
    Core/QSEnvironment.cpp
    Core/QSStatement.cpp
    DataEngine/QSDataEngine.cpp
    DataEngine/QSMetadataHelper.cpp
    DataEngine/QSTable.cpp
    DataEngine/QSTypeInfoMetadataSource.cpp
    Main_Unix.cpp)

add_custom_command(
    DEPENDS ${CMAKE_SOURCE_DIR}/mapd.thrift
    OUTPUT ${CMAKE_BINARY_DIR}/gen-cpp/MapD.cpp
    COMMAND thrift
    ARGS -gen cpp -o ${CMAKE_BINARY_DIR} ${CMAKE_SOURCE_DIR}/mapd.thrift)

set_source_files_properties(${CMAKE_BINARY_DIR}/gen-cpp/MapD.cpp GENERATED)
set_source_files_properties(${CMAKE_BINARY_DIR}/gen-cpp/mapd_constants.cpp GENERATED)
set_source_files_properties(${CMAKE_BINARY_DIR}/gen-cpp/mapd_types.cpp GENERATED)

include_directories(${CMAKE_SOURCE_DIR}/ODBC ${CMAKE_SOURCE_DIR}/ODBC/Common ${CMAKE_SOURCE_DIR}/ODBC/Core ${CMAKE_SOURCE_DIR}/ODBC/DataEngine $ENV{SIMBAENGINE_THIRDPARTY_DIR}/odbcheaders $ENV{SIMBAENGINE_DIR}/Include/DSI $ENV{SIMBAENGINE_DIR}/Include/DSI/Client $ENV{SIMBAENGINE_DIR}/Include/Support $ENV{SIMBAENGINE_DIR}/Include/Support/Exceptions $ENV{SIMBAENGINE_DIR}/Include/Support/TypedDataWrapper $ENV{SIMBAENGINE_DIR}/Include/SQLEngine $ENV{SIMBAENGINE_DIR}/Include/SQLEngine/AETree $ENV{SIMBAENGINE_DIR}/Include/SQLEngine/DSIExt $ENV{SIMBAENGINE_DIR}/Include/Server)

add_library(ODBC ${odbc_source_files} ${CMAKE_BINARY_DIR}/gen-cpp/MapD.cpp ${CMAKE_BINARY_DIR}/gen-cpp/mapd_constants.cpp ${CMAKE_BINARY_DIR}/gen-cpp/mapd_types.cpp)

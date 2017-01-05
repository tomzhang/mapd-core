set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Dlinux -fPIC -DSIZEOF_LONG_INT=8")

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

include_directories(${CMAKE_SOURCE_DIR}/ODBC ${CMAKE_SOURCE_DIR}/ODBC/Common ${CMAKE_SOURCE_DIR}/ODBC/Core ${CMAKE_SOURCE_DIR}/ODBC/DataEngine $ENV{SIMBAENGINE_THIRDPARTY_DIR}/odbcheaders $ENV{SIMBAENGINE_DIR}/Include/DSI $ENV{SIMBAENGINE_DIR}/Include/DSI/Client $ENV{SIMBAENGINE_DIR}/Include/Support $ENV{SIMBAENGINE_DIR}/Include/Support/Exceptions $ENV{SIMBAENGINE_DIR}/Include/Support/TypedDataWrapper $ENV{SIMBAENGINE_DIR}/Include/SQLEngine $ENV{SIMBAENGINE_DIR}/Include/SQLEngine/AETree $ENV{SIMBAENGINE_DIR}/Include/SQLEngine/DSIExt $ENV{SIMBAENGINE_DIR}/Include/Server)

add_library(ODBC SHARED ${odbc_source_files})
target_link_libraries(ODBC mapd_thrift "-Wl,--no-undefined -Wl,--whole-archive,$ENV{SIMBAENGINE_DIR}/Lib/Linux_x8664/libSimbaDSI_debug.a,$ENV{SIMBAENGINE_DIR}/Lib/Linux_x8664/libSimbaSupport_debug.a,$ENV{SIMBAENGINE_DIR}/Lib/Linux_x8664/libAEProcessor_debug.a,$ENV{SIMBAENGINE_DIR}/Lib/Linux_x8664/libCore_debug.a,$ENV{SIMBAENGINE_DIR}/Lib/Linux_x8664/libDSIExt_debug.a,$ENV{SIMBAENGINE_DIR}/Lib/Linux_x8664/libExecutor_debug.a,$ENV{SIMBAENGINE_DIR}/Lib/Linux_x8664/libParser_debug.a,$ENV{SIMBAENGINE_DIR}/Lib/Linux_x8664/libSimbaODBC_debug.a -Wl,--no-whole-archive -L$ENV{SIMBAENGINE_THIRDPARTY_DIR}/icu/53.1/centos5/gcc4_4/release64/lib -licudata_sb64 -licui18n_sb64 -licuuc_sb64 -ldl -lcrypto -lthrift")

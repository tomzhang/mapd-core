if(ENABLE_LICENSING_AWS)
  find_package(CURL REQUIRED)
  find_package(OpenSSL REQUIRED)

  configure_file("${CMAKE_CURRENT_SOURCE_DIR}/AWSBillingProducts.in.h"
    "${CMAKE_BINARY_DIR}/Licensing/AWSBillingProducts.h" @ONLY)
  add_library(AWSMarketplace AWSMarketplace.cpp AWSMarketplace.h ${CMAKE_BINARY_DIR}/Licensing/AWSBillingProducts.h)
  target_link_libraries(AWSMarketplace cpr ${OPENSSL_LIBRARIES} ${Glog_LIBRARIES})
endif()

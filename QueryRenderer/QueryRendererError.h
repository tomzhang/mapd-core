#ifndef QUERY_RENDERER_ERROR_H_
#define QUERY_RENDERER_ERROR_H_

#include <glog/logging.h>
#include <stdexcept>
#include <string>

// #define RUNTIME_LOG(condition, severity) !(condition) ? (void) 0 :
// inline bool RUNTIME_LOG(bool condition, int severity) {
// }

#define RUNTIME_EX_ASSERT(condition, errstr)                                                         \
  (condition) ? (void)0 : ([&]() {                                                                   \
    LOG(ERROR) << errstr;                                                                            \
    throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + " " + errstr); \
                          }())

#define THROW_RUNTIME_EX(errstr) \
  LOG(ERROR) << errstr;          \
  throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + " " + errstr);

#endif  // QUERY_RENDERER_ERROR_H_

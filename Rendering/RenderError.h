#ifndef QUERY_RENDERER_ERROR_H_
#define QUERY_RENDERER_ERROR_H_

#include <glog/logging.h>
#include <stdexcept>
#include <string>

namespace Rendering {

class RenderError : public std::runtime_error {
 public:
  RenderError(const std::string& details = "") : std::runtime_error(details) {}
};

class OutOfGpuMemoryError : public RenderError {
 public:
  OutOfGpuMemoryError(const std::string& details = "")
      : RenderError("OutOfGpuMemoryError" + (details.length() ? ": " + details : "")) {}
};

}  // namespace Rendering

#define RUNTIME_EX_ASSERT(condition, errstr)                                                               \
  (condition) ? (void)0 : ([&]() {                                                                         \
    LOG(ERROR) << errstr;                                                                                  \
    throw ::Rendering::RenderError(std::string(__FILE__) + ":" + std::to_string(__LINE__) + " " + errstr); \
                          }())

#define THROW_RUNTIME_EX(errstr) \
  LOG(ERROR) << errstr;          \
  throw ::Rendering::RenderError(std::string(__FILE__) + ":" + std::to_string(__LINE__) + " " + errstr);

#endif  // QUERY_RENDERER_ERROR_H_

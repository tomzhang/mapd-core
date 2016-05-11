#include "Utils.h"
#include <algorithm>

namespace QueryRenderer {

std::chrono::milliseconds getCurrentTimeMS() {
  return std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch());
}

std::string makeLowerCase(const std::string& str) {
  std::string rtn(str);
  std::transform(rtn.begin(), rtn.end(), rtn.begin(), ::tolower);

  return rtn;
}

std::string makeUpperCase(const std::string& str) {
  std::string rtn(str);
  std::transform(rtn.begin(), rtn.end(), rtn.begin(), ::toupper);

  return rtn;
}

}  // namespace QueryRenderer

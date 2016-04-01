#include "Utilities.h"

namespace QueryRenderer {

std::chrono::milliseconds getCurrentTimeMS() {
  return std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch());
}

}  // namespace QueryRenderer

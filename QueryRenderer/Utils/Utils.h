#ifndef QUERYRENDERER_UTILITIES_H_
#define QUERYRENDERER_UTILITIES_H_

#include <chrono>
#include <string>

namespace QueryRenderer {

std::chrono::milliseconds getCurrentTimeMS();

std::string makeLowerCase(const std::string& str);
std::string makeUpperCase(const std::string& str);
}

#endif  // QUERYRENDERER_UTILITIES_H_

#ifndef QUERYRENDERER_UTILS_SHADERUTILS_H_
#define QUERYRENDERER_UTILS_SHADERUTILS_H_

#include "../Marks/Types.h"
#include <boost/algorithm/string/replace.hpp>
#include <boost/algorithm/string/regex.hpp>

namespace QueryRenderer {

struct ShaderUtils {
  typedef std::string::iterator str_itr;
  typedef boost::iterator_range<str_itr> str_itr_range;

  static str_itr_range getGLSLFunctionBounds(std::string& codeStr, const std::string& funcName);
  static void setRenderPropertyTypeInShaderSrc(const BaseRenderProperty& prop, std::string& shaderSrc);
  static void setRenderPropertyAttrTypeInShaderSrc(const BaseRenderProperty& prop,
                                                   std::string& shaderSrc,
                                                   bool isUniform);
  static std::string addShaderExtensionAndPreprocessorInfo(const std::string& shaderSrc);
};

}  // namespace QueryRenderer

#endif  // QUERYRENDERER_UTILS_SHADERUTILS_H_

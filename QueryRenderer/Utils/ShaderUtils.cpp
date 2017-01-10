#include "ShaderUtils.h"
#include "../Marks/RenderProperty.h"
#include <Rendering/Renderer/GL/GLRenderer.h>

#include <sstream>

namespace QueryRenderer {

ShaderUtils::str_itr_range ShaderUtils::getGLSLFunctionBounds(std::string& codeStr, const std::string& funcName) {
  std::string regexStr = "\\h*\\w+\\h+" + funcName + "\\h*\\([\\w\\h\\v,]*\\)\\h*\\v*\\{";
  boost::regex funcSignatureRegex(regexStr);

  str_itr_range signature_range = boost::find_regex(codeStr, funcSignatureRegex);

  if (signature_range.empty()) {
    return signature_range;
  }

  str_itr lastItr = signature_range.end();
  std::vector<str_itr> scopeStack = {lastItr - 1};

  size_t curr_pos = signature_range.end() - codeStr.begin();
  while ((curr_pos = codeStr.find_first_of("{}", curr_pos)) != std::string::npos) {
    if (codeStr[curr_pos] == '{') {
      scopeStack.push_back(codeStr.begin() + curr_pos);
    } else {
      // found a '}'
      scopeStack.pop_back();
      if (scopeStack.empty()) {
        lastItr = codeStr.begin() + curr_pos + 1;
        break;
      }
    }
  }

  if (!scopeStack.empty()) {
    // return an empty range
    return str_itr_range();
  }

  return str_itr_range(signature_range.begin(), lastItr);
}

void ShaderUtils::setRenderPropertyTypeInShaderSrc(const BaseRenderProperty& prop, std::string& shaderSrc) {
  std::string inname = prop.getInGLSLName();
  {
    std::string intype = prop.getInGLSLType();
    std::ostringstream in_ss;
    in_ss << "<" << inname << "Type"
          << ">";
    boost::replace_first(shaderSrc, in_ss.str(), intype);
  }

  {
    auto& inGLType = prop.getInTypeGL();
    std::ostringstream in_ss;
    in_ss << "<" << inname << "Enum"
          << ">";
    boost::replace_first(shaderSrc, in_ss.str(), std::to_string(inGLType->glslGLType()));
  }

  std::string outname = prop.getOutGLSLName();
  {
    std::string outtype = prop.getOutGLSLType();
    std::ostringstream out_ss;
    out_ss << "<" << outname << "Type"
           << ">";
    boost::replace_first(shaderSrc, out_ss.str(), outtype);
  }

  {
    auto& outGLType = prop.getOutTypeGL();
    std::ostringstream out_ss;
    out_ss << "<" << outname << "Enum"
           << ">";
    boost::replace_first(shaderSrc, out_ss.str(), std::to_string(outGLType->glslGLType()));
  }
}

void ShaderUtils::setRenderPropertyAttrTypeInShaderSrc(const BaseRenderProperty& prop,
                                                       std::string& shaderSrc,
                                                       bool isUniform) {
  std::ostringstream ss;

  std::string name = prop.getName();

  ss << "<useU" << name << ">";
  boost::replace_first(shaderSrc, ss.str(), (isUniform ? "1" : "0"));
}

std::string ShaderUtils::addShaderExtensionAndPreprocessorInfo(const std::string& shaderSrc) {
  // add the type defines at the head of vert shaders
  RUNTIME_EX_ASSERT(
      ::Rendering::GL::GLRenderer::getCurrentThreadRenderer(),
      "A renderer needs to be active to call BaseTypeGL::getExtensionStr & BaseTypeGL::getTypeDefinesMacroForShader()");

  std::regex versionRegex("#version\\s+\\d+\\s+core\\s*");
  std::smatch versionMatch;

  size_t idx = 0;
  if (std::regex_search(shaderSrc, versionMatch, versionRegex)) {
    idx = versionMatch.position() + versionMatch.length();
  }

  std::string rtn = shaderSrc;
  rtn.insert(idx,
             ::Rendering::GL::BaseTypeGL::getExtensionStr() + "\n" +
                 ::Rendering::GL::BaseTypeGL::getTypeDefinesMacroForShader());
  return rtn;
}

}  // namespace QueryRenderer

#include "QueryRendererError.h"
#include "ShaderUtils.h"
#include <boost/filesystem.hpp>
#include <glog/logging.h>
#include <fstream>

using namespace MapD_Renderer;

std::string MapD_Renderer::getShaderCodeFromFile(const std::string& shaderFilename) {
  boost::filesystem::path path(shaderFilename);
  RUNTIME_EX_ASSERT(boost::filesystem::exists(path), "Shader \"" + shaderFilename + "\" does not exist.");
  RUNTIME_EX_ASSERT(boost::filesystem::is_regular_file(path),
                    "Shader \"" + shaderFilename + "\" is not an appropriate shader file.");

  // Read the Vertex Shader code from the file
  std::string shaderCode;
  std::ifstream shaderStream(shaderFilename, std::ios::in);

  if (shaderStream.is_open()) {
    std::string line = "";
    while (getline(shaderStream, line)) {
      shaderCode += "\n" + line;
    }
    shaderStream.close();
  } else {
    THROW_RUNTIME_EX("Cannot open shader file \"" + shaderFilename + "\".");
  }

  return shaderCode;
}

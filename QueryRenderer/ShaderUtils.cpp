#include "ShaderUtils.h"
#include <boost/filesystem.hpp>
#include <glog/logging.h>
#include <fstream>

using namespace MapD_Renderer;

std::string MapD_Renderer::getShaderCodeFromFile(const std::string& shaderFilename) {
  boost::filesystem::path path(shaderFilename);
  CHECK(boost::filesystem::exists(path) && boost::filesystem::is_regular_file(path));

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
    printf("Impossible to open %s. Are you in the right directory ? Don't forget to read the FAQ !\n",
           shaderFilename.c_str());
    CHECK(false);
  }

  return shaderCode;
}

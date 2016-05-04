#include "../MapDGL.h"
#include "GLShader.h"
#include "GLShaderBlockLayout.h"
#include "../GLResourceManager.h"
#include <boost/algorithm/string/predicate.hpp>

namespace Rendering {
namespace GL {
namespace Resources {

static detail::UniformAttrInfo* createUniformAttrInfoPtr(GLint type, GLint size, GLuint location);

namespace detail {

struct Uniform1uiAttr : UniformAttrInfo {
  Uniform1uiAttr(GLint t, GLint s, GLuint l) : UniformAttrInfo(t, s, l) {}
  void setAttr(const void* data) {
    MAPD_CHECK_GL_ERROR(glUniform1uiv(location, size, static_cast<const GLuint*>(data)));
  }
};

struct Uniform2uiAttr : UniformAttrInfo {
  Uniform2uiAttr(GLint t, GLint s, GLuint l) : UniformAttrInfo(t, s, l) {}
  void setAttr(const void* data) {
    MAPD_CHECK_GL_ERROR(glUniform2uiv(location, size, static_cast<const GLuint*>(data)));
  }
};

struct Uniform3uiAttr : UniformAttrInfo {
  Uniform3uiAttr(GLint t, GLint s, GLuint l) : UniformAttrInfo(t, s, l) {}
  void setAttr(const void* data) {
    MAPD_CHECK_GL_ERROR(glUniform3uiv(location, size, static_cast<const GLuint*>(data)));
  }
};

struct Uniform4uiAttr : UniformAttrInfo {
  Uniform4uiAttr(GLint t, GLint s, GLuint l) : UniformAttrInfo(t, s, l) {}
  void setAttr(const void* data) {
    MAPD_CHECK_GL_ERROR(glUniform4uiv(location, size, static_cast<const GLuint*>(data)));
  }
};

struct Uniform1iAttr : UniformAttrInfo {
  Uniform1iAttr(GLint t, GLint s, GLuint l) : UniformAttrInfo(t, s, l) {}
  void setAttr(const void* data) { MAPD_CHECK_GL_ERROR(glUniform1iv(location, size, static_cast<const GLint*>(data))); }
};

struct Uniform2iAttr : UniformAttrInfo {
  Uniform2iAttr(GLint t, GLint s, GLuint l) : UniformAttrInfo(t, s, l) {}
  void setAttr(const void* data) { MAPD_CHECK_GL_ERROR(glUniform2iv(location, size, static_cast<const GLint*>(data))); }
};

struct Uniform3iAttr : UniformAttrInfo {
  Uniform3iAttr(GLint t, GLint s, GLuint l) : UniformAttrInfo(t, s, l) {}
  void setAttr(const void* data) { MAPD_CHECK_GL_ERROR(glUniform3iv(location, size, static_cast<const GLint*>(data))); }
};

struct Uniform4iAttr : UniformAttrInfo {
  Uniform4iAttr(GLint t, GLint s, GLuint l) : UniformAttrInfo(t, s, l) {}
  void setAttr(const void* data) { MAPD_CHECK_GL_ERROR(glUniform4iv(location, size, static_cast<const GLint*>(data))); }
};

struct Uniform1fAttr : UniformAttrInfo {
  Uniform1fAttr(GLint t, GLint s, GLuint l) : UniformAttrInfo(t, s, l) {}
  void setAttr(const void* data) {
    MAPD_CHECK_GL_ERROR(glUniform1fv(location, size, static_cast<const GLfloat*>(data)));
  }
};

struct Uniform2fAttr : UniformAttrInfo {
  Uniform2fAttr(GLint t, GLint s, GLuint l) : UniformAttrInfo(t, s, l) {}
  void setAttr(const void* data) {
    MAPD_CHECK_GL_ERROR(glUniform2fv(location, size, static_cast<const GLfloat*>(data)));
  }
};

struct Uniform3fAttr : UniformAttrInfo {
  Uniform3fAttr(GLint t, GLint s, GLuint l) : UniformAttrInfo(t, s, l) {}
  void setAttr(const void* data) {
    MAPD_CHECK_GL_ERROR(glUniform3fv(location, size, static_cast<const GLfloat*>(data)));
  }
};

struct Uniform4fAttr : UniformAttrInfo {
  Uniform4fAttr(GLint t, GLint s, GLuint l) : UniformAttrInfo(t, s, l) {}
  void setAttr(const void* data) {
    MAPD_CHECK_GL_ERROR(glUniform4fv(location, size, static_cast<const GLfloat*>(data)));
  }
};

struct Uniform1dAttr : UniformAttrInfo {
  Uniform1dAttr(GLint t, GLint s, GLuint l) : UniformAttrInfo(t, s, l) {}
  void setAttr(const void* data) {
    MAPD_CHECK_GL_ERROR(glUniform1dv(location, size, static_cast<const GLdouble*>(data)));
  }
};

struct Uniform2dAttr : UniformAttrInfo {
  Uniform2dAttr(GLint t, GLint s, GLuint l) : UniformAttrInfo(t, s, l) {}
  void setAttr(const void* data) {
    MAPD_CHECK_GL_ERROR(glUniform2dv(location, size, static_cast<const GLdouble*>(data)));
  }
};

struct Uniform3dAttr : UniformAttrInfo {
  Uniform3dAttr(GLint t, GLint s, GLuint l) : UniformAttrInfo(t, s, l) {}
  void setAttr(const void* data) {
    MAPD_CHECK_GL_ERROR(glUniform3dv(location, size, static_cast<const GLdouble*>(data)));
  }
};

struct Uniform4dAttr : UniformAttrInfo {
  Uniform4dAttr(GLint t, GLint s, GLuint l) : UniformAttrInfo(t, s, l) {}
  void setAttr(const void* data) {
    MAPD_CHECK_GL_ERROR(glUniform4dv(location, size, static_cast<const GLdouble*>(data)));
  }
};

UniformSamplerAttr::UniformSamplerAttr(GLint t, GLint s, GLuint l, GLenum target, GLint startTxImgUnit)
    : UniformAttrInfo(t, s, l), target(target), startTexImgUnit(startTxImgUnit) {
}

void UniformSamplerAttr::setAttr(const void* data) {
  // TODO(croot): throw an warning for 2 samplers bound to the same texture unit?
  RUNTIME_EX_ASSERT(startTexImgUnit >= GL_TEXTURE0,
                    "Uniform sampler2d has not been properly initialized with a texture image unit.");

  const GLuint* textureIds = static_cast<const GLuint*>(data);
  for (int i = 0; i < size; ++i) {
    // TODO(croot): should I always set the binding point?
    // i.e. glUniform1i(location, startTexImgUnit + i);
    // or will doing that once always keep it set for the shader?
    MAPD_CHECK_GL_ERROR(glActiveTexture(startTexImgUnit + i));
    MAPD_CHECK_GL_ERROR(glBindTexture(target, textureIds[i]));
  }
}

void UniformSamplerAttr::setTexImgUnit(GLint texImgUnit) {
  GLint maxTexImgUnits;
  MAPD_CHECK_GL_ERROR(glGetIntegerv(GL_MAX_COMBINED_TEXTURE_IMAGE_UNITS, &maxTexImgUnits));

  RUNTIME_EX_ASSERT(texImgUnit >= GL_TEXTURE0 && texImgUnit + size <= GL_TEXTURE0 + maxTexImgUnits,
                    "Invalid start texture image unit set for uniform sampler attr. Start texture image unit: " +
                        std::to_string(texImgUnit) + " + number of samplers: " + std::to_string(size) +
                        " is not in the texture image unit range: [" + std::to_string(GL_TEXTURE0) + ", " +
                        std::to_string(GL_TEXTURE0 + maxTexImgUnits) + "]");

  for (int i = 0; i < size; ++i) {
    // TODO(croot): use glBindTextures​(GLuint first​, GLsizei count​, const GLuint *textures​) instead as
    // descrived here: https://www.opengl.org/wiki/Sampler_(GLSL)#Multibind_and_textures
    MAPD_CHECK_GL_ERROR(glUniform1i(location, texImgUnit + i - GL_TEXTURE0));
  }

  startTexImgUnit = texImgUnit;
}

UniformBlockAttrInfo::UniformBlockAttrInfo(const GLShaderShPtr& shaderPtr,
                                           const std::string& blockName,
                                           GLint blockIndex,
                                           GLint bufferSize,
                                           GLint bufferBindingIndex,
                                           ShaderBlockLayoutType layoutType)
    : blockName(blockName),
      blockIndex(blockIndex),
      bufferSize(bufferSize),
      bufferBindingIndex(bufferBindingIndex),
      blockLayoutPtr(new GLShaderBlockLayout(layoutType, shaderPtr, bufferSize)) {
}

void UniformBlockAttrInfo::setBufferBinding(GLuint programId, GLint bindingIndex) {
  // TODO(croot): check that this binding index isn't in use elsewhere?
  // What happens in that case?
  MAPD_CHECK_GL_ERROR(glUniformBlockBinding(programId, blockIndex, bindingIndex));
  bufferBindingIndex = bindingIndex;
}

void UniformBlockAttrInfo::bindBuffer(GLuint bufferId) {
  // TODO(croot): check that the buffer is valid...
  // TODO(croot): what about buffer size?
  MAPD_CHECK_GL_ERROR(glBindBufferBase(GL_UNIFORM_BUFFER, bufferBindingIndex, bufferId));
}

void UniformBlockAttrInfo::bindBuffer(GLuint bufferId, size_t offsetBytes, size_t sizeBytes) {
  // TODO(croot): check that the buffer is valid...
  // TODO(croot): what about buffer size?
  MAPD_CHECK_GL_ERROR(glBindBufferRange(GL_UNIFORM_BUFFER, bufferBindingIndex, bufferId, offsetBytes, sizeBytes));
}

void UniformBlockAttrInfo::addActiveAttr(const std::string& attrName, GLint type, GLint size, GLuint idx) {
  activeAttrs.insert(make_pair(attrName, std::unique_ptr<UniformAttrInfo>(createUniformAttrInfoPtr(type, size, -1))));

  ShaderBlockLayoutType layoutType = blockLayoutPtr->getLayoutType();
  bool usePublicAddAttr = false;
  if (layoutType == ShaderBlockLayoutType::STD140 || layoutType == ShaderBlockLayoutType::STD430) {
    usePublicAddAttr = true;
  }

  switch (type) {
    case GL_UNSIGNED_INT:
      if (usePublicAddAttr) {
        blockLayoutPtr->addAttribute<unsigned int>(attrName);
      } else {
        blockLayoutPtr->addAttribute<unsigned int>(attrName, idx);
      }
      break;
    case GL_UNSIGNED_INT_VEC2:
      if (usePublicAddAttr) {
        blockLayoutPtr->addAttribute<unsigned int, 2>(attrName);
      } else {
        blockLayoutPtr->addAttribute<unsigned int, 2>(attrName, idx);
      }
      break;
    case GL_UNSIGNED_INT_VEC3:
      if (usePublicAddAttr) {
        blockLayoutPtr->addAttribute<unsigned int, 3>(attrName);
      } else {
        blockLayoutPtr->addAttribute<unsigned int, 3>(attrName, idx);
      }
      break;
    case GL_UNSIGNED_INT_VEC4:
      if (usePublicAddAttr) {
        blockLayoutPtr->addAttribute<unsigned int, 4>(attrName);
      } else {
        blockLayoutPtr->addAttribute<unsigned int, 4>(attrName, idx);
      }
      break;

    case GL_INT:
      if (usePublicAddAttr) {
        blockLayoutPtr->addAttribute<int>(attrName);
      } else {
        blockLayoutPtr->addAttribute<int>(attrName, idx);
      }
      break;
    case GL_INT_VEC2:
      if (usePublicAddAttr) {
        blockLayoutPtr->addAttribute<int, 2>(attrName);
      } else {
        blockLayoutPtr->addAttribute<int, 2>(attrName, idx);
      }
      break;
    case GL_INT_VEC3:
      if (usePublicAddAttr) {
        blockLayoutPtr->addAttribute<int, 3>(attrName);
      } else {
        blockLayoutPtr->addAttribute<int, 3>(attrName, idx);
      }
      break;
    case GL_INT_VEC4:
      if (usePublicAddAttr) {
        blockLayoutPtr->addAttribute<int, 4>(attrName);
      } else {
        blockLayoutPtr->addAttribute<int, 4>(attrName, idx);
      }
      break;

    case GL_FLOAT:
      if (usePublicAddAttr) {
        blockLayoutPtr->addAttribute<float>(attrName);
      } else {
        blockLayoutPtr->addAttribute<float>(attrName, idx);
      }
      break;
    case GL_FLOAT_VEC2:
      if (usePublicAddAttr) {
        blockLayoutPtr->addAttribute<float, 2>(attrName);
      } else {
        blockLayoutPtr->addAttribute<float, 2>(attrName, idx);
      }
      break;
    case GL_FLOAT_VEC3:
      if (usePublicAddAttr) {
        blockLayoutPtr->addAttribute<float, 3>(attrName);
      } else {
        blockLayoutPtr->addAttribute<float, 3>(attrName, idx);
      }
      break;
    case GL_FLOAT_VEC4:
      if (usePublicAddAttr) {
        blockLayoutPtr->addAttribute<float, 4>(attrName);
      } else {
        blockLayoutPtr->addAttribute<float, 4>(attrName, idx);
      }
      break;

    case GL_DOUBLE:
      if (usePublicAddAttr) {
        blockLayoutPtr->addAttribute<double>(attrName);
      } else {
        blockLayoutPtr->addAttribute<double>(attrName, idx);
      }
      break;
    case GL_DOUBLE_VEC2:
      if (usePublicAddAttr) {
        blockLayoutPtr->addAttribute<double, 2>(attrName);
      } else {
        blockLayoutPtr->addAttribute<double, 2>(attrName, idx);
      }
      break;
    case GL_DOUBLE_VEC3:
      if (usePublicAddAttr) {
        blockLayoutPtr->addAttribute<double, 3>(attrName);
      } else {
        blockLayoutPtr->addAttribute<double, 3>(attrName, idx);
      }
      break;
    case GL_DOUBLE_VEC4:
      if (usePublicAddAttr) {
        blockLayoutPtr->addAttribute<double, 4>(attrName);
      } else {
        blockLayoutPtr->addAttribute<double, 4>(attrName, idx);
      }
      break;

    // case GL_SAMPLER_1D:
    // case GL_SAMPLER_1D_ARRAY:

    // case GL_SAMPLER_1D_SHADOW:
    // case GL_SAMPLER_1D_ARRAY_SHADOW:

    // case GL_INT_SAMPLER_1D:
    // case GL_INT_SAMPLER_1D_ARRAY:

    // case GL_UNSIGNED_INT_SAMPLER_1D:
    // case GL_UNSIGNED_INT_SAMPLER_1D_ARRAY:

    // TODO(croot): for samplers, the texture image unit can be set according to
    // https://www.opengl.org/wiki/Sampler_(GLSL)#Version_4.20_binding
    // i.e. layout(binding=0) uniform sampler2D diffuseTex;
    // but I can't determine how to find out what that binding is with any
    // of the opengl program introspection methods. So, I may need to do
    // a scan of the shader source myself to determine that.
    // case GL_SAMPLER_2D:
    // case GL_SAMPLER_2D_ARRAY:
    // case GL_SAMPLER_2D_MULTISAMPLE:
    // case GL_SAMPLER_2D_MULTISAMPLE_ARRAY:

    // case GL_SAMPLER_2D_SHADOW:
    // case GL_SAMPLER_2D_ARRAY_SHADOW:

    // case GL_INT_SAMPLER_2D:
    // case GL_INT_SAMPLER_2D_ARRAY:
    // case GL_INT_SAMPLER_2D_MULTISAMPLE:
    // case GL_INT_SAMPLER_2D_MULTISAMPLE_ARRAY:

    // case GL_UNSIGNED_INT_SAMPLER_2D:
    // case GL_UNSIGNED_INT_SAMPLER_2D_ARRAY:
    // case GL_UNSIGNED_INT_SAMPLER_2D_MULTISAMPLE:
    // case GL_UNSIGNED_INT_SAMPLER_2D_MULTISAMPLE_ARRAY:

    // case GL_SAMPLER_3D:
    // case GL_INT_SAMPLER_3D:
    // case GL_UNSIGNED_INT_SAMPLER_3D:

    // case GL_SAMPLER_CUBE:
    // case GL_SAMPLER_CUBE_SHADOW:
    // case GL_INT_SAMPLER_CUBE:
    // case GL_UNSIGNED_INT_SAMPLER_CUBE:

    // case GL_SAMPLER_BUFFER:
    // case GL_INT_SAMPLER_BUFFER:
    // case GL_UNSIGNED_INT_SAMPLER_BUFFER:

    // case GL_SAMPLER_2D_RECT:
    // case GL_SAMPLER_2D_RECT_SHADOW:
    // case GL_INT_SAMPLER_2D_RECT:
    // case GL_UNSIGNED_INT_SAMPLER_2D_RECT:

    default:
      // TODO(croot): throw an error instead?
      LOG(WARNING) << "GL type: " << type << " is not a currently supported type for buffer-backed uniform block.";
      break;
  }
}

}  // namespace detail

using detail::AttrInfo;
using detail::UniformAttrInfo;
using detail::Uniform1uiAttr;
using detail::Uniform2uiAttr;
using detail::Uniform3uiAttr;
using detail::Uniform4uiAttr;
using detail::Uniform1iAttr;
using detail::Uniform2iAttr;
using detail::Uniform3iAttr;
using detail::Uniform4iAttr;
using detail::Uniform1fAttr;
using detail::Uniform2fAttr;
using detail::Uniform3fAttr;
using detail::Uniform4fAttr;
using detail::Uniform1dAttr;
using detail::Uniform2dAttr;
using detail::Uniform3dAttr;
using detail::Uniform4dAttr;
using detail::UniformSamplerAttr;
using detail::UniformBlockAttrInfo;

static GLint compileShader(const GLuint& shaderId, const std::string& shaderSrc, std::string& errStr) {
  GLint compiled;
  const GLchar* shaderSrcCode = shaderSrc.c_str();

  MAPD_CHECK_GL_ERROR(glShaderSource(shaderId, 1, &shaderSrcCode, NULL));
  MAPD_CHECK_GL_ERROR(glCompileShader(shaderId));
  MAPD_CHECK_GL_ERROR(glGetShaderiv(shaderId, GL_COMPILE_STATUS, &compiled));
  if (!compiled) {
    GLchar errLog[1024];
    MAPD_CHECK_GL_ERROR(glGetShaderInfoLog(shaderId, 1024, NULL, errLog));
    errStr.assign(std::string(errLog));

    // std::ofstream shadersrcstream;
    // shadersrcstream.open("shadersource.vert");
    // shadersrcstream << shadersrc;
    // shadersrcstream.close();
  }

  return compiled;
}

static std::string getShaderSource(const GLuint& shaderId) {
  if (!shaderId) {
    return std::string();
  }

  GLint sourceLen;
  MAPD_CHECK_GL_ERROR(glGetShaderiv(shaderId, GL_SHADER_SOURCE_LENGTH, &sourceLen));

  std::shared_ptr<GLchar> source(new GLchar[sourceLen], std::default_delete<GLchar[]>());
  MAPD_CHECK_GL_ERROR(glGetShaderSource(shaderId, sourceLen, NULL, source.get()));

  return std::string(source.get());
}

static GLint linkProgram(const GLuint& programId, std::string& errStr) {
  GLint linked;

  MAPD_CHECK_GL_ERROR(glLinkProgram(programId));
  MAPD_CHECK_GL_ERROR(glGetProgramiv(programId, GL_LINK_STATUS, &linked));
  if (!linked) {
    GLchar errLog[1024];
    MAPD_CHECK_GL_ERROR(glGetProgramInfoLog(programId, 1024, NULL, errLog));
    errStr.assign(std::string(errLog));
  }

  return linked;
}

static UniformAttrInfo* createUniformAttrInfoPtr(GLint type, GLint size, GLuint location) {
  UniformAttrInfo* rtn = NULL;

  switch (type) {
    case GL_UNSIGNED_INT:
      rtn = new Uniform1uiAttr(type, size, location);
      break;
    case GL_UNSIGNED_INT_VEC2:
      rtn = new Uniform2uiAttr(type, size, location);
      break;
    case GL_UNSIGNED_INT_VEC3:
      rtn = new Uniform3uiAttr(type, size, location);
      break;
    case GL_UNSIGNED_INT_VEC4:
      rtn = new Uniform4uiAttr(type, size, location);
      break;

    case GL_INT:
      rtn = new Uniform1iAttr(type, size, location);
      break;
    case GL_INT_VEC2:
      rtn = new Uniform2iAttr(type, size, location);
      break;
    case GL_INT_VEC3:
      rtn = new Uniform3iAttr(type, size, location);
      break;
    case GL_INT_VEC4:
      rtn = new Uniform4iAttr(type, size, location);
      break;

    case GL_FLOAT:
      rtn = new Uniform1fAttr(type, size, location);
      break;
    case GL_FLOAT_VEC2:
      rtn = new Uniform2fAttr(type, size, location);
      break;
    case GL_FLOAT_VEC3:
      rtn = new Uniform3fAttr(type, size, location);
      break;
    case GL_FLOAT_VEC4:
      rtn = new Uniform4fAttr(type, size, location);
      break;

    case GL_DOUBLE:
      rtn = new Uniform1dAttr(type, size, location);
      break;
    case GL_DOUBLE_VEC2:
      rtn = new Uniform2dAttr(type, size, location);
      break;
    case GL_DOUBLE_VEC3:
      rtn = new Uniform3dAttr(type, size, location);
      break;
    case GL_DOUBLE_VEC4:
      rtn = new Uniform4dAttr(type, size, location);
      break;

    // case GL_SAMPLER_1D:
    // case GL_SAMPLER_1D_ARRAY:

    // case GL_SAMPLER_1D_SHADOW:
    // case GL_SAMPLER_1D_ARRAY_SHADOW:

    // case GL_INT_SAMPLER_1D:
    // case GL_INT_SAMPLER_1D_ARRAY:

    // case GL_UNSIGNED_INT_SAMPLER_1D:
    // case GL_UNSIGNED_INT_SAMPLER_1D_ARRAY:

    // TODO(croot): for samplers, the texture image unit can be set according to
    // https://www.opengl.org/wiki/Sampler_(GLSL)#Version_4.20_binding
    // i.e. layout(binding=0) uniform sampler2D diffuseTex;
    // but I can't determine how to find out what that binding is with any
    // of the opengl program introspection methods. So, I may need to do
    // a scan of the shader source myself to determine that.
    case GL_SAMPLER_2D:
      rtn = new UniformSamplerAttr(type, size, location, GL_TEXTURE_2D);
      break;
    case GL_SAMPLER_2D_ARRAY:
      rtn = new UniformSamplerAttr(type, size, location, GL_TEXTURE_2D_ARRAY);
      break;
    case GL_SAMPLER_2D_MULTISAMPLE:
      rtn = new UniformSamplerAttr(type, size, location, GL_TEXTURE_2D_MULTISAMPLE);
      break;
    case GL_SAMPLER_2D_MULTISAMPLE_ARRAY:
      rtn = new UniformSamplerAttr(type, size, location, GL_TEXTURE_2D_MULTISAMPLE_ARRAY);
      break;

    // case GL_SAMPLER_2D_SHADOW:
    // case GL_SAMPLER_2D_ARRAY_SHADOW:

    // case GL_INT_SAMPLER_2D:
    // case GL_INT_SAMPLER_2D_ARRAY:
    // case GL_INT_SAMPLER_2D_MULTISAMPLE:
    // case GL_INT_SAMPLER_2D_MULTISAMPLE_ARRAY:

    case GL_UNSIGNED_INT_SAMPLER_2D:
      rtn = new UniformSamplerAttr(type, size, location, GL_TEXTURE_2D);
      break;
    case GL_UNSIGNED_INT_SAMPLER_2D_ARRAY:
      rtn = new UniformSamplerAttr(type, size, location, GL_TEXTURE_2D_ARRAY);
      break;
    case GL_UNSIGNED_INT_SAMPLER_2D_MULTISAMPLE:
      rtn = new UniformSamplerAttr(type, size, location, GL_TEXTURE_2D_MULTISAMPLE);
      break;
    case GL_UNSIGNED_INT_SAMPLER_2D_MULTISAMPLE_ARRAY:
      rtn = new UniformSamplerAttr(type, size, location, GL_TEXTURE_2D_MULTISAMPLE_ARRAY);
      break;

    // case GL_SAMPLER_3D:
    // case GL_INT_SAMPLER_3D:
    // case GL_UNSIGNED_INT_SAMPLER_3D:

    // case GL_SAMPLER_CUBE:
    // case GL_SAMPLER_CUBE_SHADOW:
    // case GL_INT_SAMPLER_CUBE:
    // case GL_UNSIGNED_INT_SAMPLER_CUBE:

    // case GL_SAMPLER_BUFFER:
    // case GL_INT_SAMPLER_BUFFER:
    // case GL_UNSIGNED_INT_SAMPLER_BUFFER:

    // case GL_SAMPLER_2D_RECT:
    // case GL_SAMPLER_2D_RECT_SHADOW:
    // case GL_INT_SAMPLER_2D_RECT:
    // case GL_UNSIGNED_INT_SAMPLER_2D_RECT:

    default:
      THROW_RUNTIME_EX("createUniformAttrPtr(): GL type " + std::to_string(type) + " is not yet a supported type.");
      break;
  }

  return rtn;
}

// TODO(croot): move strsplit() functions into a string utility somewhere
static std::vector<std::string>& strsplit(const std::string& s, char delim, std::vector<std::string>& elems) {
  std::stringstream ss(s);
  std::string item;
  while (std::getline(ss, item, delim)) {
    if (item.length()) {
      elems.push_back(item);
    }
  }
  return elems;
}

static std::vector<std::string> strsplit(const std::string& s, char delim) {
  std::vector<std::string> elems;
  strsplit(s, delim, elems);
  return elems;
}

GLShader::GLShader(const RendererWkPtr& rendererPtr)
    : GLResource(rendererPtr, GLResourceType::SHADER), _vertShaderId(0), _fragShaderId(0), _programId(0) {
}

GLShader::~GLShader() {
  cleanupResource();
}

void GLShader::_initResource(const std::string& vertSrc, const std::string& fragSrc) {
  std::string errStr;

  validateRenderer(__FILE__, __LINE__);

  // build and compile the vertex shader
  MAPD_CHECK_GL_ERROR((_vertShaderId = glCreateShader(GL_VERTEX_SHADER)));

  if (!compileShader(_vertShaderId, vertSrc, errStr)) {
    _cleanupResource();
    THROW_RUNTIME_EX("Error compiling vertex shader: " + errStr + ".\n\nVertex shader src:\n\n" + vertSrc);
  }

  MAPD_CHECK_GL_ERROR((_fragShaderId = glCreateShader(GL_FRAGMENT_SHADER)));
  if (!compileShader(_fragShaderId, fragSrc, errStr)) {
    _cleanupResource();
    THROW_RUNTIME_EX("Error compiling fragment shader: " + errStr + ".\n\nFragment shader src:\n\n" + fragSrc);
  }

  MAPD_CHECK_GL_ERROR((_programId = glCreateProgram()));
  RUNTIME_EX_ASSERT(_programId != 0, "An error occured trying to create a shader program");

  MAPD_CHECK_GL_ERROR(glAttachShader(_programId, _vertShaderId));
  MAPD_CHECK_GL_ERROR(glAttachShader(_programId, _fragShaderId));
  if (!linkProgram(_programId, errStr)) {
    // clean out the shader references
    _cleanupResource();
    THROW_RUNTIME_EX("Error linking the shader: " + errStr);
  }

  GLint numAttrs;
  GLchar attrName[512];
  GLint attrType;
  // GLint uAttrType;
  GLint attrSz;
  GLuint attrLoc;

  // NOTE: This needs to be improved to handle basic array types, structs,
  // arrays of structs, interface blocks (uniform & shader storage blocks),
  // subroutines, atomic counters, etc.
  // See: https://www.opengl.org/wiki/Program_Introspection

  GLRenderer* renderer = getGLRenderer();
  CHECK(renderer);

  GLResourceManagerShPtr rsrcMgr = renderer->getResourceManager();
  GLShaderShPtr myRsrc = std::dynamic_pointer_cast<GLShader>(rsrcMgr->getResourcePtr(this));
  CHECK(myRsrc && myRsrc.get() == this);

  // setup the uniform attributes
  _uniformBlockAttrs.clear();
  MAPD_CHECK_GL_ERROR(glGetProgramiv(_programId, GL_ACTIVE_UNIFORM_BLOCKS, &numAttrs));
  std::set<GLuint> uniformBlockAttrIndices;
  for (GLint i = 0; i < numAttrs; ++i) {
    MAPD_CHECK_GL_ERROR(glGetActiveUniformBlockName(_programId, i, 512, NULL, attrName));
    std::string attrBlockName(attrName);

    GLint bufSz;
    MAPD_CHECK_GL_ERROR(glGetActiveUniformBlockiv(_programId, i, GL_UNIFORM_BLOCK_DATA_SIZE, &bufSz));

    GLint activeBlockUniforms;
    MAPD_CHECK_GL_ERROR(
        glGetActiveUniformBlockiv(_programId, i, GL_UNIFORM_BLOCK_ACTIVE_UNIFORMS, &activeBlockUniforms));

    GLint activeBlockUniformIndices[activeBlockUniforms];
    MAPD_CHECK_GL_ERROR(
        glGetActiveUniformBlockiv(_programId, i, GL_UNIFORM_BLOCK_ACTIVE_UNIFORM_INDICES, activeBlockUniformIndices));

    GLint bindingIndex;
    MAPD_CHECK_GL_ERROR(glGetActiveUniformBlockiv(_programId, i, GL_UNIFORM_BLOCK_BINDING, &bindingIndex));

    // TODO(croot): support different layouts for shader blocks. I don't think there's
    // a way to query this using the program introspection API, so we'll need to parse the
    // shader's source code. Right now, hard-coding STD140 for QueryRenderer

    // TODO(croot): I think we'll need to add the order the attributes are added to the
    // shader block layout by parsing the block as well, so we'll need to manually parse
    // the block code
    auto blockItr = _uniformBlockAttrs.insert(
        make_pair(attrBlockName,
                  std::make_unique<UniformBlockAttrInfo>(
                      myRsrc, attrBlockName, i, bufSz, bindingIndex, ShaderBlockLayoutType::STD140)));

    blockItr.first->second->blockLayoutPtr->beginAddingAttrs();

    for (GLint j = 0; j < activeBlockUniforms; ++j) {
      uniformBlockAttrIndices.insert(activeBlockUniformIndices[j]);
    }
  }

  _uniformAttrs.clear();
  MAPD_CHECK_GL_ERROR(glGetProgramiv(_programId, GL_ACTIVE_UNIFORMS, &numAttrs));
  for (GLuint i = 0; i < static_cast<GLuint>(numAttrs); ++i) {
    MAPD_CHECK_GL_ERROR(glGetActiveUniformName(_programId, i, 512, NULL, attrName));
    std::string attrNameStr(attrName);

    MAPD_CHECK_GL_ERROR(glGetActiveUniformsiv(_programId, 1, &i, GL_UNIFORM_TYPE, &attrType));
    MAPD_CHECK_GL_ERROR(glGetActiveUniformsiv(_programId, 1, &i, GL_UNIFORM_SIZE, &attrSz));

    if (uniformBlockAttrIndices.erase(i) > 0) {
      std::vector<std::string> dotSepNames = strsplit(attrNameStr, '.');

      // TODO(croot): handle blocks within blocks?
      RUNTIME_EX_ASSERT(dotSepNames.size() == 2, "Currently not supporting shader blocks within shader blocks.");

      auto blockItr = _uniformBlockAttrs.find(dotSepNames[0]);
      RUNTIME_EX_ASSERT(blockItr != _uniformBlockAttrs.end(),
                        "Could not find uniform block " + dotSepNames[0] + " in shader.");

      // TODO(croot): it turns out adding attrs to the block in their location order is
      // not the right order the attrs are defined in the block
      // we may need to parse the block in the src line-by-line to get the right order
      blockItr->second->addActiveAttr(dotSepNames[1], attrType, attrSz, i);
    } else {
      attrLoc = MAPD_CHECK_GL_ERROR(glGetUniformLocation(_programId, attrName));

      if (boost::algorithm::ends_with(attrNameStr, "[0]")) {
        attrNameStr.erase(attrNameStr.size() - 3, 3);
      }

      _uniformAttrs.insert(make_pair(
          attrNameStr, std::unique_ptr<UniformAttrInfo>(createUniformAttrInfoPtr(attrType, attrSz, attrLoc))));
    }
  }

  // verify that the shader block was build properly
  for (const auto& uniformBlockItr : _uniformBlockAttrs) {
    uniformBlockItr.second->blockLayoutPtr->endAddingAttrs();
    CHECK(static_cast<size_t>(uniformBlockItr.second->bufferSize) ==
          uniformBlockItr.second->blockLayoutPtr->getNumBytesInBlock())
        << uniformBlockItr.second->blockName << ": " << uniformBlockItr.second->bufferSize
        << " != " << uniformBlockItr.second->blockLayoutPtr->getNumBytesInBlock();
  }

  // now setup the vertex attributes
  _vertexAttrs.clear();
  MAPD_CHECK_GL_ERROR(glGetProgramiv(_programId, GL_ACTIVE_ATTRIBUTES, &numAttrs));
  for (int i = 0; i < numAttrs; ++i) {
    MAPD_CHECK_GL_ERROR(glGetActiveAttrib(_programId, i, 512, NULL, &attrSz, (GLenum*)&attrType, attrName));
    MAPD_CHECK_GL_ERROR((attrLoc = glGetAttribLocation(_programId, attrName)));

    _vertexAttrs.insert(
        std::make_pair(std::string(attrName), std::unique_ptr<AttrInfo>(new AttrInfo(attrType, attrSz, attrLoc))));
  }

  setUsable();
}

void GLShader::_cleanupResource() {
  if (_vertShaderId) {
    MAPD_CHECK_GL_ERROR(glDeleteShader(_vertShaderId));
  }

  if (_fragShaderId) {
    MAPD_CHECK_GL_ERROR(glDeleteShader(_fragShaderId));
  }

  if (_programId) {
    MAPD_CHECK_GL_ERROR(glDeleteProgram(_programId));
  }

  _makeEmpty();
}

void GLShader::_makeEmpty() {
  _uniformAttrs.clear();
  _vertexAttrs.clear();
  _vertShaderId = 0;
  _fragShaderId = 0;
  _programId = 0;
}

UniformAttrInfo* GLShader::_validateAttr(const std::string& attrName) {
  validateUsability(__FILE__, __LINE__);

  auto itr = _uniformAttrs.find(attrName);

  // TODO(croot): check if bound
  RUNTIME_EX_ASSERT(itr != _uniformAttrs.end(), "Uniform attribute \"" + attrName + "\" is not defined in the shader.");

  return itr->second.get();
}

UniformSamplerAttr* GLShader::_validateSamplerAttr(const std::string& attrName) {
  UniformAttrInfo* info = _validateAttr(attrName);
  UniformSamplerAttr* samplerAttr = dynamic_cast<UniformSamplerAttr*>(info);

  RUNTIME_EX_ASSERT(samplerAttr != nullptr, "Uniform attribute: " + attrName + " is not a sampler attribute.");

  return samplerAttr;
}

UniformBlockAttrInfo* GLShader::_validateBlockAttr(const std::string& blockName,
                                                   const GLUniformBufferShPtr& ubo,
                                                   size_t idx) {
  validateUsability(__FILE__, __LINE__);

  auto itr = _uniformBlockAttrs.find(blockName);

  // TODO(croot): check if bound
  RUNTIME_EX_ASSERT(itr != _uniformBlockAttrs.end(),
                    "Uniform block \"" + blockName + "\" is not defined in the shader.");

  RUNTIME_EX_ASSERT(ubo != nullptr, "Uniform buffer object is null. Cannot bind it to the block " + blockName + ".");

  GLRenderer* renderer = GLRenderer::getCurrentThreadRenderer();
  // this should've already passed via the validateUsability() method above.
  CHECK(renderer);
  RUNTIME_EX_ASSERT(renderer->getBoundUniformBuffer() == ubo,
                    "The uniform buffer object is not currently bound to the active renderer. Bind the UBO first.");

  RUNTIME_EX_ASSERT(
      idx < ubo->numItems(),
      "Index " + std::to_string(idx) + " out of bounds. Ubo has only " + std::to_string(ubo->numItems()) + " blocks.");

  return itr->second.get();
}

std::string GLShader::getVertexSource() const {
  validateUsability(__FILE__, __LINE__);
  return getShaderSource(_vertShaderId);
}

std::string GLShader::getFragmentSource() const {
  validateUsability(__FILE__, __LINE__);
  return getShaderSource(_fragShaderId);
}

bool GLShader::hasUniformAttribute(const std::string& attrName) {
  UniformAttrMap::const_iterator iter = _uniformAttrs.find(attrName);
  return (iter != _uniformAttrs.end());
}

GLint GLShader::getUniformAttributeGLType(const std::string& attrName) {
  UniformAttrMap::const_iterator iter = _uniformAttrs.find(attrName);

  RUNTIME_EX_ASSERT(
      iter != _uniformAttrs.end(),
      "GLShader::getUniformAttributeGLType(): uniform attribute \"" + attrName + "\" does not exist in shader.");

  return iter->second->type;
}

GLShaderBlockLayoutShPtr GLShader::getBlockLayout(const std::string& blockName) const {
  const auto itr = _uniformBlockAttrs.find(blockName);

  RUNTIME_EX_ASSERT(itr != _uniformBlockAttrs.end(),
                    "GLShader::getBlockLayout(): uniform block \"" + blockName + "\" does not exist in shader.");

  return itr->second->blockLayoutPtr;
}

void GLShader::setSamplerAttribute(const std::string& attrName, const GLResourceShPtr& rsrc) {
  switch (rsrc->getResourceType()) {
    case GLResourceType::TEXTURE_2D:
    case GLResourceType::TEXTURE_2D_ARRAY: {
      UniformSamplerAttr* samplerAttr = _validateSamplerAttr(attrName);
      RUNTIME_EX_ASSERT(samplerAttr->target == rsrc->getTarget(),
                        "Attr mismatch. Sampler expects a " + std::to_string(samplerAttr->target) +
                            " but the texture is a " + std::to_string(rsrc->getTarget()));
      GLuint id = rsrc->getId();
      samplerAttr->setAttr((void*)(&id));
    } break;
    default:
      THROW_RUNTIME_EX("Attr mismatch. Invalid resource type: " + to_string(rsrc->getResourceType()));
  }
}

void GLShader::setSamplerTextureImageUnit(const std::string& attrName, GLenum startTexImageUnit) {
  UniformSamplerAttr* samplerAttr = _validateSamplerAttr(attrName);
  samplerAttr->setTexImgUnit(startTexImageUnit);
}

bool GLShader::hasUniformBlockAttribute(const std::string& attrName) {
  for (auto& itr : _uniformBlockAttrs) {
    if (itr.second->activeAttrs.find(attrName) != itr.second->activeAttrs.end()) {
      return true;
    }
  }

  return false;
}

void GLShader::bindUniformBufferToBlock(const std::string& blockName, const GLUniformBufferShPtr& ubo, int idx) {
  UniformBlockAttrInfo* blockAttr = _validateBlockAttr(blockName, ubo, idx);
  if (idx < 0) {
    blockAttr->bindBuffer(ubo->getId());
  } else {
    size_t numBytes = ubo->getNumBytesPerItem();
    blockAttr->bindBuffer(ubo->getId(), idx * numBytes, numBytes);
  }
}

bool GLShader::hasVertexAttribute(const std::string& attrName) const {
  auto iter = _vertexAttrs.find(attrName);
  return (iter != _vertexAttrs.end());
}

GLuint GLShader::getVertexAttributeLocation(const std::string& attrName) const {
  AttrMap::const_iterator iter = _vertexAttrs.find(attrName);

  RUNTIME_EX_ASSERT(iter != _vertexAttrs.end(),
                    "Attribute \"" + attrName + "\" does not exist in shader. Cannot get attribute location.");

  AttrInfo* info = iter->second.get();

  return info->location;
}

}  // namespace Resources
}  // namespace GL
}  // namespace Rendering

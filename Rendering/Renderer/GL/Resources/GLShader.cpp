#include "../MapDGL.h"
#include "GLShader.h"
#include "GLShaderBlockLayout.h"
#include "../GLResourceManager.h"
#include <boost/algorithm/string/predicate.hpp>

namespace Rendering {
namespace GL {
namespace Resources {

static std::unique_ptr<detail::UniformAttrInfo> createUniformAttrInfoPtr(GLint type, GLint size, GLuint location);

namespace detail {

struct Uniform1uiAttr : UniformAttrInfo {
  Uniform1uiAttr(GLint t, GLint s, GLuint l) : UniformAttrInfo(t, s, l) {}
  void setAttr(const void* data, GLint attrSize) {
    CHECK(attrSize <= size);
    MAPD_CHECK_GL_ERROR(glUniform1uiv(location, attrSize, static_cast<const GLuint*>(data)));
  }
};

struct Uniform2uiAttr : UniformAttrInfo {
  Uniform2uiAttr(GLint t, GLint s, GLuint l) : UniformAttrInfo(t, s, l) {}
  void setAttr(const void* data, GLint attrSize) {
    CHECK(attrSize <= size);
    MAPD_CHECK_GL_ERROR(glUniform2uiv(location, attrSize, static_cast<const GLuint*>(data)));
  }
};

struct Uniform3uiAttr : UniformAttrInfo {
  Uniform3uiAttr(GLint t, GLint s, GLuint l) : UniformAttrInfo(t, s, l) {}
  void setAttr(const void* data, GLint attrSize) {
    CHECK(attrSize <= size);
    MAPD_CHECK_GL_ERROR(glUniform3uiv(location, attrSize, static_cast<const GLuint*>(data)));
  }
};

struct Uniform4uiAttr : UniformAttrInfo {
  Uniform4uiAttr(GLint t, GLint s, GLuint l) : UniformAttrInfo(t, s, l) {}
  void setAttr(const void* data, GLint attrSize) {
    CHECK(attrSize <= size);
    MAPD_CHECK_GL_ERROR(glUniform4uiv(location, attrSize, static_cast<const GLuint*>(data)));
  }
};

struct Uniform1iAttr : UniformAttrInfo {
  Uniform1iAttr(GLint t, GLint s, GLuint l) : UniformAttrInfo(t, s, l) {}
  void setAttr(const void* data, GLint attrSize) {
    CHECK(attrSize <= size);
    MAPD_CHECK_GL_ERROR(glUniform1iv(location, attrSize, static_cast<const GLint*>(data)));
  }
};

struct Uniform2iAttr : UniformAttrInfo {
  Uniform2iAttr(GLint t, GLint s, GLuint l) : UniformAttrInfo(t, s, l) {}
  void setAttr(const void* data, GLint attrSize) {
    CHECK(attrSize <= size);
    MAPD_CHECK_GL_ERROR(glUniform2iv(location, attrSize, static_cast<const GLint*>(data)));
  }
};

struct Uniform3iAttr : UniformAttrInfo {
  Uniform3iAttr(GLint t, GLint s, GLuint l) : UniformAttrInfo(t, s, l) {}
  void setAttr(const void* data, GLint attrSize) {
    CHECK(attrSize <= size);
    MAPD_CHECK_GL_ERROR(glUniform3iv(location, attrSize, static_cast<const GLint*>(data)));
  }
};

struct Uniform4iAttr : UniformAttrInfo {
  Uniform4iAttr(GLint t, GLint s, GLuint l) : UniformAttrInfo(t, s, l) {}
  void setAttr(const void* data, GLint attrSize) {
    CHECK(attrSize <= size);
    MAPD_CHECK_GL_ERROR(glUniform4iv(location, attrSize, static_cast<const GLint*>(data)));
  }
};

struct Uniform1ui64Attr : UniformAttrInfo {
  Uniform1ui64Attr(GLint t, GLint s, GLuint l) : UniformAttrInfo(t, s, l) {}
  void setAttr(const void* data, GLint attrSize) {
    CHECK(attrSize <= size && GLEW_NV_vertex_attrib_integer_64bit);
    MAPD_CHECK_GL_ERROR(glUniform1ui64vNV(location, attrSize, static_cast<const GLuint64EXT*>(data)));
  }
};

struct Uniform2ui64Attr : UniformAttrInfo {
  Uniform2ui64Attr(GLint t, GLint s, GLuint l) : UniformAttrInfo(t, s, l) {}
  void setAttr(const void* data, GLint attrSize) {
    CHECK(attrSize <= size && GLEW_NV_vertex_attrib_integer_64bit);
    MAPD_CHECK_GL_ERROR(glUniform2ui64vNV(location, attrSize, static_cast<const GLuint64EXT*>(data)));
  }
};

struct Uniform3ui64Attr : UniformAttrInfo {
  Uniform3ui64Attr(GLint t, GLint s, GLuint l) : UniformAttrInfo(t, s, l) {}
  void setAttr(const void* data, GLint attrSize) {
    CHECK(attrSize <= size && GLEW_NV_vertex_attrib_integer_64bit);
    MAPD_CHECK_GL_ERROR(glUniform3ui64vNV(location, attrSize, static_cast<const GLuint64EXT*>(data)));
  }
};

struct Uniform4ui64Attr : UniformAttrInfo {
  Uniform4ui64Attr(GLint t, GLint s, GLuint l) : UniformAttrInfo(t, s, l) {}
  void setAttr(const void* data, GLint attrSize) {
    CHECK(attrSize <= size && GLEW_NV_vertex_attrib_integer_64bit);
    MAPD_CHECK_GL_ERROR(glUniform4ui64vNV(location, attrSize, static_cast<const GLuint64EXT*>(data)));
  }
};

struct Uniform1i64Attr : UniformAttrInfo {
  Uniform1i64Attr(GLint t, GLint s, GLuint l) : UniformAttrInfo(t, s, l) {}
  void setAttr(const void* data, GLint attrSize) {
    CHECK(attrSize <= size && GLEW_NV_vertex_attrib_integer_64bit);
    MAPD_CHECK_GL_ERROR(glUniform1i64vNV(location, attrSize, static_cast<const GLint64EXT*>(data)));
  }
};

struct Uniform2i64Attr : UniformAttrInfo {
  Uniform2i64Attr(GLint t, GLint s, GLuint l) : UniformAttrInfo(t, s, l) {}
  void setAttr(const void* data, GLint attrSize) {
    CHECK(attrSize <= size && GLEW_NV_vertex_attrib_integer_64bit);
    MAPD_CHECK_GL_ERROR(glUniform2i64vNV(location, attrSize, static_cast<const GLint64EXT*>(data)));
  }
};

struct Uniform3i64Attr : UniformAttrInfo {
  Uniform3i64Attr(GLint t, GLint s, GLuint l) : UniformAttrInfo(t, s, l) {}
  void setAttr(const void* data, GLint attrSize) {
    CHECK(attrSize <= size && GLEW_NV_vertex_attrib_integer_64bit);
    MAPD_CHECK_GL_ERROR(glUniform3i64vNV(location, attrSize, static_cast<const GLint64EXT*>(data)));
  }
};

struct Uniform4i64Attr : UniformAttrInfo {
  Uniform4i64Attr(GLint t, GLint s, GLuint l) : UniformAttrInfo(t, s, l) {}
  void setAttr(const void* data, GLint attrSize) {
    CHECK(attrSize <= size && GLEW_NV_vertex_attrib_integer_64bit);
    MAPD_CHECK_GL_ERROR(glUniform4i64vNV(location, attrSize, static_cast<const GLint64EXT*>(data)));
  }
};

struct Uniform1fAttr : UniformAttrInfo {
  Uniform1fAttr(GLint t, GLint s, GLuint l) : UniformAttrInfo(t, s, l) {}
  void setAttr(const void* data, GLint attrSize) {
    CHECK(attrSize <= size);
    MAPD_CHECK_GL_ERROR(glUniform1fv(location, attrSize, static_cast<const GLfloat*>(data)));
  }
};

struct Uniform2fAttr : UniformAttrInfo {
  Uniform2fAttr(GLint t, GLint s, GLuint l) : UniformAttrInfo(t, s, l) {}
  void setAttr(const void* data, GLint attrSize) {
    CHECK(attrSize <= size);
    MAPD_CHECK_GL_ERROR(glUniform2fv(location, attrSize, static_cast<const GLfloat*>(data)));
  }
};

struct Uniform3fAttr : UniformAttrInfo {
  Uniform3fAttr(GLint t, GLint s, GLuint l) : UniformAttrInfo(t, s, l) {}
  void setAttr(const void* data, GLint attrSize) {
    CHECK(attrSize <= size);
    MAPD_CHECK_GL_ERROR(glUniform3fv(location, attrSize, static_cast<const GLfloat*>(data)));
  }
};

struct Uniform4fAttr : UniformAttrInfo {
  Uniform4fAttr(GLint t, GLint s, GLuint l) : UniformAttrInfo(t, s, l) {}
  void setAttr(const void* data, GLint attrSize) {
    CHECK(attrSize <= size);
    MAPD_CHECK_GL_ERROR(glUniform4fv(location, attrSize, static_cast<const GLfloat*>(data)));
  }
};

struct Uniform1dAttr : UniformAttrInfo {
  Uniform1dAttr(GLint t, GLint s, GLuint l) : UniformAttrInfo(t, s, l) {}
  void setAttr(const void* data, GLint attrSize) {
    CHECK(attrSize <= size);
    MAPD_CHECK_GL_ERROR(glUniform1dv(location, attrSize, static_cast<const GLdouble*>(data)));
  }
};

struct Uniform2dAttr : UniformAttrInfo {
  Uniform2dAttr(GLint t, GLint s, GLuint l) : UniformAttrInfo(t, s, l) {}
  void setAttr(const void* data, GLint attrSize) {
    CHECK(attrSize <= size);
    MAPD_CHECK_GL_ERROR(glUniform2dv(location, attrSize, static_cast<const GLdouble*>(data)));
  }
};

struct Uniform3dAttr : UniformAttrInfo {
  Uniform3dAttr(GLint t, GLint s, GLuint l) : UniformAttrInfo(t, s, l) {}
  void setAttr(const void* data, GLint attrSize) {
    CHECK(attrSize <= size);
    MAPD_CHECK_GL_ERROR(glUniform3dv(location, attrSize, static_cast<const GLdouble*>(data)));
  }
};

struct Uniform4dAttr : UniformAttrInfo {
  Uniform4dAttr(GLint t, GLint s, GLuint l) : UniformAttrInfo(t, s, l) {}
  void setAttr(const void* data, GLint attrSize) {
    CHECK(attrSize <= size);
    MAPD_CHECK_GL_ERROR(glUniform4dv(location, attrSize, static_cast<const GLdouble*>(data)));
  }
};

UniformSamplerAttr::UniformSamplerAttr(GLint t, GLint s, GLuint l, GLenum target, GLint startTxImgUnit)
    : UniformAttrInfo(t, s, l), target(target), startTexImgUnit(startTxImgUnit) {}

void UniformSamplerAttr::setAttr(const void* data, GLint attrSize) {
  CHECK(attrSize <= size);

  // TODO(croot): throw an warning for 2 samplers bound to the same texture unit?
  RUNTIME_EX_ASSERT(startTexImgUnit >= GL_TEXTURE0,
                    "Uniform sampler2d has not been properly initialized with a texture image unit.");

  const GLuint* textureIds = static_cast<const GLuint*>(data);
  for (int i = 0; i < attrSize; ++i) {
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
    // described here: https://www.opengl.org/wiki/Sampler_(GLSL)#Multibind_and_textures
    MAPD_CHECK_GL_ERROR(glUniform1i(location, texImgUnit + i - GL_TEXTURE0));
  }

  startTexImgUnit = texImgUnit;
}

UniformImageLoadStoreAttr::UniformImageLoadStoreAttr(GLint t,
                                                     GLint s,
                                                     GLuint l,
                                                     GLenum format,
                                                     GLenum access,
                                                     bool isMultisampled,
                                                     GLint startImgUnit)
    : UniformAttrInfo(t, s, l),
      format(format),
      access(access),
      isMultisampled(isMultisampled),
      startImgUnit(startImgUnit) {}

void UniformImageLoadStoreAttr::setAttr(const void* data, GLint attrSize) {
  setAttr(data, attrSize, false, 0);
}

void UniformImageLoadStoreAttr::setAttr(const void* data, GLint attrSize, bool layered, int layerIdx) {
  CHECK(attrSize <= size);

  // TODO(croot): throw an warning for 2 samplers bound to the same texture unit?
  RUNTIME_EX_ASSERT(startImgUnit >= 0,
                    "Uniform image load store has not been properly initialized with an image unit.");

  const GLuint* textureIds = static_cast<const GLuint*>(data);
  for (int i = 0; i < attrSize; ++i) {
    // TODO(croot): should I always set the binding point?
    // i.e. glUniform1i(location, startImgUnit + i);
    // or will doing that once always keep it set for the shader?

    // TODO(croot): support mipmaps
    MAPD_CHECK_GL_ERROR(glBindImageTexture(startImgUnit + i, textureIds[i], 0, layered, layerIdx, access, format));
  }
}

void UniformImageLoadStoreAttr::setImgUnit(GLint imgUnit, GLint attrSize) {
  GLint maxImgUnits;
  // TODO(croot): check the max number of image units in the shader stage?
  // i.e. glGetIntegerv(GL_MAX_VERTEX_IMAGE_UNIFORMS, ...)
  // See: https://www.opengl.org/wiki/Image_Load_Store#Images_in_the_context

  CHECK(attrSize <= size);

  MAPD_CHECK_GL_ERROR(glGetIntegerv(GL_MAX_IMAGE_UNITS, &maxImgUnits));

  RUNTIME_EX_ASSERT(imgUnit >= 0 && imgUnit + size <= maxImgUnits,
                    "Invalid start image unit set for uniform image load store attr. Start image unit: " +
                        std::to_string(imgUnit) + " + number of image load stores in attr: " + std::to_string(size) +
                        " is not in the image unit range: [0, " + std::to_string(maxImgUnits) + "]");

  for (int i = 0; i < attrSize; ++i) {
    // TODO(croot): use glBindTextures​(GLuint first​, GLsizei count​, const GLuint *textures​) instead as
    // described here: https://www.opengl.org/wiki/Sampler_(GLSL)#Multibind_and_textures
    MAPD_CHECK_GL_ERROR(glUniform1i(location, imgUnit + i));
  }

  startImgUnit = imgUnit;
}

UniformBlockAttrInfo::UniformBlockAttrInfo(const std::set<std::string>& supportedExtensions,
                                           const GLShaderShPtr& shaderPtr,
                                           const std::string& blockName,
                                           GLint blockIndex,
                                           GLint bufferSize,
                                           GLint bufferBindingIndex,
                                           ShaderBlockLayoutType layoutType)
    : blockName(blockName),
      blockIndex(blockIndex),
      bufferSize(bufferSize),
      bufferBindingIndex(bufferBindingIndex),
      blockLayoutPtr(new GLShaderBlockLayout(supportedExtensions, layoutType, shaderPtr, bufferSize)) {}

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
  activeAttrs.insert(make_pair(attrName, createUniformAttrInfoPtr(type, size, -1)));

  ShaderBlockLayoutType layoutType = blockLayoutPtr->getLayoutType();
  bool usePublicAddAttr = false;
  if (layoutType == ShaderBlockLayoutType::STD140 || layoutType == ShaderBlockLayoutType::STD430) {
    usePublicAddAttr = true;
  }

  if (GLEW_NV_vertex_attrib_integer_64bit) {
    switch (type) {
      case GL_INT64_NV:
        if (usePublicAddAttr) {
          blockLayoutPtr->addAttribute<int64_t>(attrName);
        } else {
          blockLayoutPtr->addAttribute<int64_t>(attrName, idx);
        }
        break;
      case GL_INT64_VEC2_NV:
        if (usePublicAddAttr) {
          blockLayoutPtr->addAttribute<int64_t, 2>(attrName);
        } else {
          blockLayoutPtr->addAttribute<int64_t, 2>(attrName, idx);
        }
        break;
      case GL_INT64_VEC3_NV:
        if (usePublicAddAttr) {
          blockLayoutPtr->addAttribute<int64_t, 3>(attrName);
        } else {
          blockLayoutPtr->addAttribute<int64_t, 3>(attrName, idx);
        }
        break;
      case GL_INT64_VEC4_NV:
        if (usePublicAddAttr) {
          blockLayoutPtr->addAttribute<int64_t, 4>(attrName);
        } else {
          blockLayoutPtr->addAttribute<int64_t, 4>(attrName, idx);
        }
        break;
      case GL_UNSIGNED_INT64_NV:
        if (usePublicAddAttr) {
          blockLayoutPtr->addAttribute<uint64_t>(attrName);
        } else {
          blockLayoutPtr->addAttribute<uint64_t>(attrName, idx);
        }
        break;
      case GL_UNSIGNED_INT64_VEC2_NV:
        if (usePublicAddAttr) {
          blockLayoutPtr->addAttribute<uint64_t, 2>(attrName);
        } else {
          blockLayoutPtr->addAttribute<uint64_t, 2>(attrName, idx);
        }
        break;
      case GL_UNSIGNED_INT64_VEC3_NV:
        if (usePublicAddAttr) {
          blockLayoutPtr->addAttribute<uint64_t, 3>(attrName);
        } else {
          blockLayoutPtr->addAttribute<uint64_t, 3>(attrName, idx);
        }
        break;
      case GL_UNSIGNED_INT64_VEC4_NV:
        if (usePublicAddAttr) {
          blockLayoutPtr->addAttribute<uint64_t, 4>(attrName);
        } else {
          blockLayoutPtr->addAttribute<uint64_t, 4>(attrName, idx);
        }
        break;
    }
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

static GLenum getSubroutineType(GLenum stage) {
  switch (stage) {
    case GL_VERTEX_SHADER:
      return GL_VERTEX_SUBROUTINE;
    case GL_FRAGMENT_SHADER:
      return GL_FRAGMENT_SUBROUTINE;
    case GL_GEOMETRY_SHADER:
      return GL_GEOMETRY_SUBROUTINE;
    default:
      THROW_RUNTIME_EX("Subroutines are not currently supported in shader stage " + std::to_string(stage));
  }
}

UniformSubroutineAttrInfo::UniformSubroutineAttrInfo(const std::string& subroutineName,
                                                     GLint sz,
                                                     GLuint loc,
                                                     GLint index,
                                                     GLenum shaderStage,
                                                     std::unordered_map<std::string, GLuint> compatibleSubroutines)
    : UniformAttrInfo(getSubroutineType(shaderStage), sz, loc),
      subroutineName(subroutineName),
      index(index),
      shaderStage(shaderStage),
      compatibleSubroutines(compatibleSubroutines) {}

void UniformSubroutineAttrInfo::setAttr(const void* data, GLint attrSize) {
  THROW_RUNTIME_EX("UniformSubroutineAttrInfo::setAttr() is not implemented yet.");
}

GLuint UniformSubroutineAttrInfo::getCompatibleSubroutineIndex(const std::string& compatibleSubroutine) {
  auto itr = compatibleSubroutines.find(compatibleSubroutine);
  RUNTIME_EX_ASSERT(itr != compatibleSubroutines.end(),
                    "The GLSL function/subroutine \"" + compatibleSubroutine +
                        "\" either does not exist or is not compatible with the uniform subroutine attr \"" +
                        subroutineName + "\".");

  // TODO(croot): As indicated here:
  // https://www.opengl.org/wiki/Shader_Subroutine#Runtime_selection
  // The subroutine selection is not actually part of the program but a
  // context state attribute -- given that, should we move this to
  // GLBindState somewhere? That's not as logical given the
  // "shader" object here, but it would ensure things are in
  // sync in that way, particularly if there's ever a case where
  // the state is saved/restored.
  // MAPD_CHECK_GL_ERROR(glUniformSubroutinesuiv(shaderStage, size, &(itr->second)));
  return itr->second;
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
using detail::Uniform1ui64Attr;
using detail::Uniform2ui64Attr;
using detail::Uniform3ui64Attr;
using detail::Uniform4ui64Attr;
using detail::Uniform1i64Attr;
using detail::Uniform2i64Attr;
using detail::Uniform3i64Attr;
using detail::Uniform4i64Attr;
using detail::Uniform1fAttr;
using detail::Uniform2fAttr;
using detail::Uniform3fAttr;
using detail::Uniform4fAttr;
using detail::Uniform1dAttr;
using detail::Uniform2dAttr;
using detail::Uniform3dAttr;
using detail::Uniform4dAttr;
using detail::UniformSamplerAttr;
using detail::UniformImageLoadStoreAttr;
using detail::UniformBlockAttrInfo;
using detail::UniformSubroutineAttrInfo;

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

static std::unique_ptr<UniformAttrInfo> createUniformAttrInfoPtr(GLint type, GLint size, GLuint location) {
  if (GLEW_NV_vertex_attrib_integer_64bit) {
    switch (type) {
      case GL_INT64_NV:
        return std::make_unique<Uniform1i64Attr>(type, size, location);
      case GL_INT64_VEC2_NV:
        return std::make_unique<Uniform2i64Attr>(type, size, location);
      case GL_INT64_VEC3_NV:
        return std::make_unique<Uniform3i64Attr>(type, size, location);
      case GL_INT64_VEC4_NV:
        return std::make_unique<Uniform4i64Attr>(type, size, location);
      case GL_UNSIGNED_INT64_NV:
        return std::make_unique<Uniform1ui64Attr>(type, size, location);
      case GL_UNSIGNED_INT64_VEC2_NV:
        return std::make_unique<Uniform2ui64Attr>(type, size, location);
      case GL_UNSIGNED_INT64_VEC3_NV:
        return std::make_unique<Uniform3ui64Attr>(type, size, location);
      case GL_UNSIGNED_INT64_VEC4_NV:
        return std::make_unique<Uniform4ui64Attr>(type, size, location);
    }
  }

  switch (type) {
    // NOTE: uniform booleans can be set using
    // any of the ui/i/f constructs. It does
    // not have one of its own.
    case GL_BOOL:
      return std::make_unique<Uniform1iAttr>(type, size, location);
    case GL_BOOL_VEC2:
      return std::make_unique<Uniform2iAttr>(type, size, location);
    case GL_BOOL_VEC3:
      return std::make_unique<Uniform3iAttr>(type, size, location);
    case GL_BOOL_VEC4:
      return std::make_unique<Uniform4iAttr>(type, size, location);

    case GL_UNSIGNED_INT:
      return std::make_unique<Uniform1uiAttr>(type, size, location);
    case GL_UNSIGNED_INT_VEC2:
      return std::make_unique<Uniform2uiAttr>(type, size, location);
    case GL_UNSIGNED_INT_VEC3:
      return std::make_unique<Uniform3uiAttr>(type, size, location);
    case GL_UNSIGNED_INT_VEC4:
      return std::make_unique<Uniform4uiAttr>(type, size, location);

    case GL_INT:
      return std::make_unique<Uniform1iAttr>(type, size, location);
    case GL_INT_VEC2:
      return std::make_unique<Uniform2iAttr>(type, size, location);
    case GL_INT_VEC3:
      return std::make_unique<Uniform3iAttr>(type, size, location);
    case GL_INT_VEC4:
      return std::make_unique<Uniform4iAttr>(type, size, location);

    case GL_FLOAT:
      return std::make_unique<Uniform1fAttr>(type, size, location);
    case GL_FLOAT_VEC2:
      return std::make_unique<Uniform2fAttr>(type, size, location);
    case GL_FLOAT_VEC3:
      return std::make_unique<Uniform3fAttr>(type, size, location);
    case GL_FLOAT_VEC4:
      return std::make_unique<Uniform4fAttr>(type, size, location);

    case GL_DOUBLE:
      return std::make_unique<Uniform1dAttr>(type, size, location);
    case GL_DOUBLE_VEC2:
      return std::make_unique<Uniform2dAttr>(type, size, location);
    case GL_DOUBLE_VEC3:
      return std::make_unique<Uniform3dAttr>(type, size, location);
    case GL_DOUBLE_VEC4:
      return std::make_unique<Uniform4dAttr>(type, size, location);

    // case GL_SAMPLER_1D:
    // case GL_IMAGE_1D:
    // case GL_SAMPLER_1D_ARRAY:
    // case GL_IMAGE_1D_ARRAY:

    // case GL_SAMPLER_1D_SHADOW:
    // case GL_SAMPLER_1D_ARRAY_SHADOW:

    // case GL_INT_SAMPLER_1D:
    // case GL_INT_IMAGE_1D:
    // case GL_INT_SAMPLER_1D_ARRAY:
    // case GL_INT_IMAGE_1D_ARRAY:

    // case GL_UNSIGNED_INT_SAMPLER_1D:
    // case GL_UNSIGNED_INT_IMAGE_1D:
    // case GL_UNSIGNED_INT_SAMPLER_1D_ARRAY:
    // case GL_UNSIGNED_INT_IMAGE_1D_ARRAY:

    // TODO(croot): for samplers, the texture image unit can be set according to
    // https://www.opengl.org/wiki/Sampler_(GLSL)#Version_4.20_binding
    // i.e. layout(binding=0) uniform sampler2D diffuseTex;
    // but I can't determine how to find out what that binding is with any
    // of the opengl program introspection methods. So, I may need to do
    // a scan of the shader source myself to determine that.

    // TODO(croot): for image load store attrs, the format, access type, and
    // the image unit can be set according to
    // https://www.opengl.org/wiki/Image_Load_Store#Images_in_the_context
    // i.e. layout(r32ui, binding=0) uniform coherent image2D image;
    // but I can't determine how to find out what the format, access type,
    // or binding is with any of the opengl program introspection methods.
    // So, I may need to do a scan of the shader source myself to determine that.
    //
    // One possible way to do that is pass in the line from the shader
    // src that defines this attr and do the scan internally for the
    // attr types that require it.

    case GL_SAMPLER_2D:
      return std::make_unique<UniformSamplerAttr>(type, size, location, GL_TEXTURE_2D);
    // case GL_IMAGE_2D:
    //   break;
    case GL_SAMPLER_2D_ARRAY:
      return std::make_unique<UniformSamplerAttr>(type, size, location, GL_TEXTURE_2D_ARRAY);
    // case GL_IMAGE_2D_ARRAY:
    //   break;
    case GL_SAMPLER_2D_MULTISAMPLE:
      return std::make_unique<UniformSamplerAttr>(type, size, location, GL_TEXTURE_2D_MULTISAMPLE);
    // case GL_IMAGE_2D_MULTISAMPLE:
    //   break;
    case GL_SAMPLER_2D_MULTISAMPLE_ARRAY:
      return std::make_unique<UniformSamplerAttr>(type, size, location, GL_TEXTURE_2D_MULTISAMPLE_ARRAY);
    // case GL_IMAGE_2D_MULTISAMPLE_ARRAY:
    //   break;

    // case GL_SAMPLER_2D_SHADOW:
    // case GL_SAMPLER_2D_ARRAY_SHADOW:

    case GL_INT_SAMPLER_2D:
      return std::make_unique<UniformSamplerAttr>(type, size, location, GL_TEXTURE_2D);
    case GL_INT_IMAGE_2D:
      // TODO(croot): need to un-hardcode the GL_RG32I/GL_READ_WRITE/multisample/binding attrs
      return std::make_unique<UniformImageLoadStoreAttr>(type, size, location, GL_R32I, GL_READ_WRITE, false, 0);
    case GL_INT_SAMPLER_2D_ARRAY:
      return std::make_unique<UniformSamplerAttr>(type, size, location, GL_TEXTURE_2D_ARRAY);
    case GL_INT_IMAGE_2D_ARRAY:
      // TODO(croot): need to un-hardcode the GL_RG32I/GL_READ_WRITE/multisample/binding attrs
      return std::make_unique<UniformImageLoadStoreAttr>(type, size, location, GL_R32I, GL_READ_WRITE, false, 0);
    case GL_INT_SAMPLER_2D_MULTISAMPLE:
      return std::make_unique<UniformSamplerAttr>(type, size, location, GL_TEXTURE_2D_MULTISAMPLE);
    case GL_INT_IMAGE_2D_MULTISAMPLE:
      break;
    case GL_INT_SAMPLER_2D_MULTISAMPLE_ARRAY:
      return std::make_unique<UniformSamplerAttr>(type, size, location, GL_TEXTURE_2D_MULTISAMPLE_ARRAY);
    case GL_INT_IMAGE_2D_MULTISAMPLE_ARRAY:
      break;

    case GL_UNSIGNED_INT_SAMPLER_2D:
      return std::make_unique<UniformSamplerAttr>(type, size, location, GL_TEXTURE_2D);
    case GL_UNSIGNED_INT_IMAGE_2D:
      // TODO(croot): need to un-hardcode the GL_RG32UI/GL_READ_WRITE/multisample/binding attrs
      return std::make_unique<UniformImageLoadStoreAttr>(type, size, location, GL_R32UI, GL_READ_WRITE, false, 0);
    case GL_UNSIGNED_INT_SAMPLER_2D_ARRAY:
      return std::make_unique<UniformSamplerAttr>(type, size, location, GL_TEXTURE_2D_ARRAY);
    case GL_UNSIGNED_INT_IMAGE_2D_ARRAY:
      // TODO(croot): need to un-hardcode the GL_RG32UI/GL_READ_WRITE/multisample/binding attrs
      return std::make_unique<UniformImageLoadStoreAttr>(type, size, location, GL_R32UI, GL_READ_WRITE, false, 0);
    case GL_UNSIGNED_INT_SAMPLER_2D_MULTISAMPLE:
      return std::make_unique<UniformSamplerAttr>(type, size, location, GL_TEXTURE_2D_MULTISAMPLE);
    case GL_UNSIGNED_INT_IMAGE_2D_MULTISAMPLE:
      break;
    case GL_UNSIGNED_INT_SAMPLER_2D_MULTISAMPLE_ARRAY:
      return std::make_unique<UniformSamplerAttr>(type, size, location, GL_TEXTURE_2D_MULTISAMPLE_ARRAY);
    case GL_UNSIGNED_INT_IMAGE_2D_MULTISAMPLE_ARRAY:
      break;

    // case GL_SAMPLER_3D:
    // case GL_IMAGE_3D:
    // case GL_INT_SAMPLER_3D:
    // case GL_INT_IMAGE_3D:
    // case GL_UNSIGNED_INT_SAMPLER_3D:
    // case GL_UNSIGNED_INT_IMAGE_3D:

    // case GL_SAMPLER_CUBE:
    // case GL_IMAGE_CUBE:
    // case GL_SAMPLER_CUBE_SHADOW:
    // case GL_INT_SAMPLER_CUBE:
    // case GL_INT_IMAGE_CUBE:
    // case GL_UNSIGNED_INT_SAMPLER_CUBE:
    // case GL_UNSIGNED_INT_IMAGE_CUBE:

    // case GL_SAMPLER_BUFFER:
    // case GL_INT_SAMPLER_BUFFER:
    // case GL_UNSIGNED_INT_SAMPLER_BUFFER:

    // case GL_SAMPLER_2D_RECT:
    // case GL_IMAGE_2D_RECT:
    // case GL_SAMPLER_2D_RECT_SHADOW:
    // case GL_INT_SAMPLER_2D_RECT:
    // case GL_INT_IMAGE_2D_RECT:
    // case GL_UNSIGNED_INT_SAMPLER_2D_RECT:
    // case GL_UNSIGNED_INT_IMAGE_2D_RECT:

    // case GL_IMAGE_BUFFER:
    // case GL_INT_IMAGE_BUFFER:
    // case GL_UNSIGNED_INT_IMAGE_BUFFER:
    // case GL_IMAGE_CUBE_MAP_ARRAY:
    // case GL_INT_IMAGE_CUBE_MAP_ARRAY:
    // case GL_UNSIGNED_INT_IMAGE_CUBE_MAP_ARRAY:

    default:
      THROW_RUNTIME_EX("createUniformAttrPtr(): GL type " + std::to_string(type) + " is not yet a supported type.");
      break;
  }

  return nullptr;
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
    : GLResource(rendererPtr, GLResourceType::SHADER),
      _vertShaderId(0),
      _geomShaderId(0),
      _fragShaderId(0),
      _programId(0) {}

GLShader::~GLShader() {
  cleanupResource();
}

void GLShader::_initResource(const std::string& vertSrc, const std::string& fragSrc, const std::string& geomSrc) {
  std::string errStr;

  validateRenderer(__FILE__, __LINE__);

  // build and compile the vertex shader
  MAPD_CHECK_GL_ERROR((_vertShaderId = glCreateShader(GL_VERTEX_SHADER)));

  if (!compileShader(_vertShaderId, vertSrc, errStr)) {
    _cleanupResource();
    THROW_RUNTIME_EX("Error compiling vertex shader: " + errStr + ".\n\nVertex shader src:\n\n" + vertSrc);
  }

  if (geomSrc.length()) {
    MAPD_CHECK_GL_ERROR((_geomShaderId = glCreateShader(GL_GEOMETRY_SHADER)));
    if (!compileShader(_geomShaderId, geomSrc, errStr)) {
      _cleanupResource();
      THROW_RUNTIME_EX("Error compiling geometry shader: " + errStr + ".\n\nGeometry shader src:\n\n" + geomSrc);
    }
  }

  MAPD_CHECK_GL_ERROR((_fragShaderId = glCreateShader(GL_FRAGMENT_SHADER)));
  if (!compileShader(_fragShaderId, fragSrc, errStr)) {
    _cleanupResource();
    THROW_RUNTIME_EX("Error compiling fragment shader: " + errStr + ".\n\nFragment shader src:\n\n" + fragSrc);
  }

  MAPD_CHECK_GL_ERROR((_programId = glCreateProgram()));
  RUNTIME_EX_ASSERT(_programId != 0, "An error occured trying to create a shader program");

  MAPD_CHECK_GL_ERROR(glAttachShader(_programId, _vertShaderId));
  if (_geomShaderId) {
    MAPD_CHECK_GL_ERROR(glAttachShader(_programId, _geomShaderId));
  }
  MAPD_CHECK_GL_ERROR(glAttachShader(_programId, _fragShaderId));
  if (!linkProgram(_programId, errStr)) {
    // clean out the shader references
    _cleanupResource();
    THROW_RUNTIME_EX("Error linking the shader: " + errStr + "\n\nVertex Shader Src:\n" + vertSrc +
                     "\n\nFragment shader src:\n" + fragSrc);
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

  auto myRsrc = getSharedResourceFromTypePtr(this);
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
                  std::make_unique<UniformBlockAttrInfo>(getGLRenderer()->getSupportedExtensions(),
                                                         myRsrc,
                                                         attrBlockName,
                                                         i,
                                                         bufSz,
                                                         bindingIndex,
                                                         ShaderBlockLayoutType::STD140)));

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

      // TODO(croot): appropriately handle the offset of the uniform in blocks
      // and verify that attrs added to a GLShaderBlockLayout have the same
      // offset after adding. One approach.
      // GLint attrOffset;
      // MAPD_CHECK_GL_ERROR(glGetActiveUniformsiv(_programId, 1, &i, GL_UNIFORM_OFFSET, &attrOffset));

      // TODO(croot): it turns out adding attrs to the block in their location order is
      // not the right order the attrs are defined in the block
      // we may need to parse the block in the src line-by-line to get the right order
      blockItr->second->addActiveAttr(dotSepNames[1], attrType, attrSz, i);
    } else {
      attrLoc = MAPD_CHECK_GL_ERROR(glGetUniformLocation(_programId, attrName));

      if (boost::algorithm::ends_with(attrNameStr, "[0]")) {
        attrNameStr.erase(attrNameStr.size() - 3, 3);
      }

      _uniformAttrs.insert(make_pair(attrNameStr, createUniformAttrInfoPtr(attrType, attrSz, attrLoc)));
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

  std::vector<GLenum> supportedStages = {GL_VERTEX_SHADER, GL_FRAGMENT_SHADER};
  if (geomSrc.size() > 0) {
    supportedStages.push_back(GL_GEOMETRY_SHADER);
  }

  GLint numSubroutines, numCompatibleSubroutines;
  std::unordered_map<GLenum, std::vector<GLuint>>::iterator itr;
  for (auto stage : supportedStages) {
    MAPD_CHECK_GL_ERROR(glGetProgramStageiv(_programId, stage, GL_ACTIVE_SUBROUTINE_UNIFORMS, &numSubroutines));

    if (numSubroutines > 0) {
      itr = _activeSubroutines.emplace(std::make_pair(GLenum(stage), std::vector<GLuint>(numSubroutines))).first;
    }

    for (int i = 0; i < numSubroutines; ++i) {
      MAPD_CHECK_GL_ERROR(glGetActiveSubroutineUniformiv(
          _programId, stage, i, GL_NUM_COMPATIBLE_SUBROUTINES, &numCompatibleSubroutines));
      MAPD_CHECK_GL_ERROR(glGetActiveSubroutineUniformiv(_programId, stage, i, GL_UNIFORM_SIZE, &attrSz));

      MAPD_CHECK_GL_ERROR(glGetActiveSubroutineUniformName(_programId, stage, i, 512, NULL, attrName));
      std::string attrNameStr(attrName);

      attrLoc = MAPD_CHECK_GL_ERROR(glGetSubroutineUniformLocation(_programId, stage, attrName));

      std::vector<GLint> compatibleSubroutinesIdxs(numCompatibleSubroutines);
      std::unordered_map<std::string, GLuint> compatibleSubroutines;

      RUNTIME_EX_ASSERT(
          numCompatibleSubroutines > 0,
          "Invalid subroutine attribute \"" + attrNameStr + "\". It does not have any compatible subroutines.");

      MAPD_CHECK_GL_ERROR(glGetActiveSubroutineUniformiv(
          _programId, stage, i, GL_COMPATIBLE_SUBROUTINES, &compatibleSubroutinesIdxs[0]));

      for (int j = 0; j < numCompatibleSubroutines; ++j) {
        MAPD_CHECK_GL_ERROR(
            glGetActiveSubroutineName(_programId, stage, compatibleSubroutinesIdxs[j], 512, NULL, attrName));
        std::string subroutineName(attrName);
        compatibleSubroutines.emplace(subroutineName, compatibleSubroutinesIdxs[j]);
        if (j == 0) {
          itr->second[attrLoc] = compatibleSubroutinesIdxs[j];
        }
      }

      _uniformSubroutineAttrs.emplace(attrNameStr,
                                      std::unique_ptr<UniformSubroutineAttrInfo>(new UniformSubroutineAttrInfo(
                                          attrName, attrSz, attrLoc, i, stage, compatibleSubroutines)));
    }

    // MAPD_CHECK_GL_ERROR(glGetProgramStageiv(_programId, stage, GL_ACTIVE_SUBROUTINES, &numSubroutines));

    // for (int i = 0; i < numSubroutines; ++i) {
    //   index = MAPD_CHECK_GL_ERROR(glGetSubroutineIndex(_programId, stage, ));
    // }
  }

  setUsable();
}

void GLShader::_cleanupResource() {
  if (_vertShaderId) {
    MAPD_CHECK_GL_ERROR(glDeleteShader(_vertShaderId));
  }

  if (_geomShaderId) {
    MAPD_CHECK_GL_ERROR(glDeleteShader(_geomShaderId));
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
  _geomShaderId = 0;
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

UniformImageLoadStoreAttr* GLShader::_validateImageLoadStoreAttr(const std::string& attrName) {
  UniformAttrInfo* info = _validateAttr(attrName);
  UniformImageLoadStoreAttr* imgAttr = dynamic_cast<UniformImageLoadStoreAttr*>(info);

  RUNTIME_EX_ASSERT(imgAttr != nullptr, "Uniform attribute: " + attrName + " is not an image load store attribute.");

  return imgAttr;
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

UniformSubroutineAttrInfo* GLShader::_validateSubroutineAttr(const std::string& attrName) {
  validateUsability(__FILE__, __LINE__);

  auto itr = _uniformSubroutineAttrs.find(attrName);

  // TODO(croot): check if bound
  RUNTIME_EX_ASSERT(
      itr != _uniformSubroutineAttrs.end(),
      "Uniform subroutine \"" + attrName + "\" is not defined in the shader, or does not have compatible subroutines.");

  return itr->second.get();
}

std::string GLShader::getVertexSource() const {
  validateUsability(__FILE__, __LINE__);
  return getShaderSource(_vertShaderId);
}

std::string GLShader::getGeometrySource() const {
  validateUsability(__FILE__, __LINE__);
  if (_geomShaderId) {
    return getShaderSource(_geomShaderId);
  }
  return "";
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

void GLShader::setSamplerAttribute(const std::string& attrName, const GLResource* rsrc) {
  switch (rsrc->getResourceType()) {
    case GLResourceType::TEXTURE_2D:
    case GLResourceType::TEXTURE_2D_ARRAY: {
      UniformSamplerAttr* samplerAttr = _validateSamplerAttr(attrName);

      RUNTIME_EX_ASSERT(samplerAttr->size == 1,
                        "The sampler attribute \"" + attrName + "\" is an array of size " +
                            std::to_string(samplerAttr->size) +
                            " and must be set using a vector, not a single texture resource");

      RUNTIME_EX_ASSERT(samplerAttr->target == rsrc->getTarget(),
                        "Attr mismatch. Sampler expects a " + std::to_string(samplerAttr->target) +
                            " but the texture is a " + std::to_string(rsrc->getTarget()));
      GLuint id = rsrc->getId();
      samplerAttr->setAttr((void*)(&id), 1);
    } break;
    default:
      THROW_RUNTIME_EX("Attr mismatch. Invalid resource type: " + to_string(rsrc->getResourceType()));
  }
}

void GLShader::setSamplerAttribute(const std::string& attrName, const GLResourceShPtr& rsrc) {
  setSamplerAttribute(attrName, rsrc.get());
}

void GLShader::setSamplerTextureImageUnit(const std::string& attrName, GLenum startTexImageUnit) {
  UniformSamplerAttr* samplerAttr = _validateSamplerAttr(attrName);
  samplerAttr->setTexImgUnit(startTexImageUnit);
}

void GLShader::setImageLoadStoreAttribute(const std::string& attrName, const GLResource* rsrc, int layerIdx) {
  UniformImageLoadStoreAttr* imgAttr = _validateImageLoadStoreAttr(attrName);

  // TODO(croot): validate compatability between the texture/texture array/texture cube/texture rect
  // object's internal format and num samples
  switch (rsrc->getResourceType()) {
    case GLResourceType::TEXTURE_2D_ARRAY: {
      auto tex2dArrayRsrc = dynamic_cast<const GLTexture2dArray*>(rsrc);
      CHECK(tex2dArrayRsrc);

      RUNTIME_EX_ASSERT(imgAttr->size == 1,
                        "The image load store attribute \"" + attrName + "\" is an array of size " +
                            std::to_string(imgAttr->size) +
                            " and must be set using a vector, not a single texture resource");

      RUNTIME_EX_ASSERT(imgAttr->format == tex2dArrayRsrc->getInternalFormat(),
                        "Attr mismatch. Image load store attr \"" + attrName + "\" expects a " +
                            std::to_string(imgAttr->format) + " formatted image but the image rsrc has a " +
                            std::to_string(tex2dArrayRsrc->getInternalFormat()) + " format.");

      GLuint id = rsrc->getId();
      imgAttr->setAttr((void*)(&id), 1, true, layerIdx);
    } break;
    case GLResourceType::TEXTURE_2D: {
      auto tex2dRsrc = dynamic_cast<const GLTexture2d*>(rsrc);
      CHECK(tex2dRsrc);

      RUNTIME_EX_ASSERT(imgAttr->size == 1,
                        "The image load store attribute \"" + attrName + "\" is an array of size " +
                            std::to_string(imgAttr->size) +
                            " and must be set using a vector, not a single texture resource");

      RUNTIME_EX_ASSERT(imgAttr->format == tex2dRsrc->getInternalFormat(),
                        "Attr mismatch. Image load store attr \"" + attrName + "\" expects a " +
                            std::to_string(imgAttr->format) + " formatted image but the image rsrc has a " +
                            std::to_string(tex2dRsrc->getInternalFormat()) + " format.");

      GLuint id = rsrc->getId();
      imgAttr->setAttr((void*)(&id), 1, false, layerIdx);
    } break;
    default:
      THROW_RUNTIME_EX("Attr mismatch. Invalid resource type: " + to_string(rsrc->getResourceType()));
  }
}

void GLShader::setImageLoadStoreAttribute(const std::string& attrName, const GLResourceShPtr& rsrc, int layerIdx) {
  setImageLoadStoreAttribute(attrName, rsrc.get(), layerIdx);
}

void GLShader::setImageLoadStoreAttribute(const std::string& attrName, const std::vector<GLTexture2dShPtr>& rsrcs) {
  UniformImageLoadStoreAttr* imgAttr = _validateImageLoadStoreAttr(attrName);

  // TODO(croot): validate compatability between the texture/texture array/texture cube/texture rect
  // object's internal format and num samples

  RUNTIME_EX_ASSERT(imgAttr->size == static_cast<int>(rsrcs.size()),
                    "Array sizes do not match. Cannot set an array of image load store attributes. The \"" + attrName +
                        "\" shader attribute is size " + std::to_string(imgAttr->size) +
                        " and the size of the vector of textures is size " + std::to_string(rsrcs.size()));

  std::vector<GLuint> ids;
  for (auto& texPtr : rsrcs) {
    RUNTIME_EX_ASSERT(imgAttr->format == texPtr->getInternalFormat(),
                      "Attr mismatch. Image load store attr \"" + attrName + "\" expects a " +
                          std::to_string(imgAttr->format) + " formatted image but the image rsrc has a " +
                          std::to_string(texPtr->getInternalFormat()) + " format.");

    ids.push_back(texPtr->getId());
  }

  imgAttr->setAttr((void*)(&ids[0]), rsrcs.size(), false, 0);
}

void GLShader::setImageLoadStoreImageUnit(const std::string& attrName, int startImgUnit) {
  UniformImageLoadStoreAttr* imgAttr = _validateImageLoadStoreAttr(attrName);
  imgAttr->setImgUnit(startImgUnit, imgAttr->size);
}

void GLShader::setSubroutine(const std::string& subroutineAttrName, const std::string& compatibleSubroutineName) {
  auto renderer = getGLRenderer();
  auto boundShader = renderer->getBoundShader();
  RUNTIME_EX_ASSERT(boundShader != nullptr && boundShader.get() == this,
                    "Cannot set subroutine uniforms for shader. The shader must be bound to the renderer first.");

  UniformSubroutineAttrInfo* subAttr = _validateSubroutineAttr(subroutineAttrName);
  RUNTIME_EX_ASSERT(subAttr->size == 1,
                    "Subroutine array sizes do not match. Cannot set an array of  subroutine attributes. The \"" +
                        subroutineAttrName + "\" shader attribute is size " + std::to_string(subAttr->size) +
                        " and trying to set a subroutine for only 1.");

  auto shaderStage = subAttr->shaderStage;
  auto attrLoc = subAttr->location;
  auto subIdx = subAttr->getCompatibleSubroutineIndex(compatibleSubroutineName);

  auto itr = _activeSubroutines.find(shaderStage);
  CHECK(itr != _activeSubroutines.end() && itr->second.size() > attrLoc);
  itr->second[attrLoc] = subIdx;

  // TODO(croot): As indicated here:
  // https://www.opengl.org/wiki/Shader_Subroutine#Runtime_selection
  // The subroutine selection is not actually part of the program but a
  // context state attribute -- given that, should we move this to
  // GLBindState somewhere? That's not as logical given the
  // "shader" object here, but it would ensure things are in
  // sync in that way, particularly if there's ever a case where
  // the state is saved/restored.

  MAPD_CHECK_GL_ERROR(glUniformSubroutinesuiv(shaderStage, itr->second.size(), &(itr->second[0])));
}

void GLShader::setSubroutines(const std::unordered_map<std::string, std::string>& subroutineVals) {
  auto renderer = getGLRenderer();
  auto boundShader = renderer->getBoundShader();
  RUNTIME_EX_ASSERT(boundShader != nullptr && boundShader.get() == this,
                    "Cannot set subroutine uniforms for shader. The shader must be bound to the renderer first.");

  UniformSubroutineAttrInfo* subAttr;

  size_t cnt = 0;
  for (auto itr : _activeSubroutines) {
    cnt += itr.second.size();
  }

  RUNTIME_EX_ASSERT(subroutineVals.size() == cnt,
                    "setSubroutines() requires all subroutine uniform attributes to be set for the entire shader "
                    "program. The user supplied " +
                        std::to_string(subroutineVals.size()) + " subroutines to set but there are " +
                        std::to_string(cnt) + " uniform subroutines to set in the shader.");

  for (auto itr : subroutineVals) {
    subAttr = _validateSubroutineAttr(itr.first);
    RUNTIME_EX_ASSERT(subAttr->size == 1,
                      "Subroutine array sizes do not match. Cannot set an array of  subroutine attributes. The \"" +
                          itr.first + "\" shader attribute is size " + std::to_string(subAttr->size) +
                          " and trying to set a subroutine for only 1.");

    auto shaderStage = subAttr->shaderStage;
    auto attrLoc = subAttr->location;
    auto subIdx = subAttr->getCompatibleSubroutineIndex(itr.second);

    auto subitr = _activeSubroutines.find(shaderStage);
    CHECK(subitr != _activeSubroutines.end() && subitr->second.size() > attrLoc);
    subitr->second[attrLoc] = subIdx;
  }

  // TODO(croot): As indicated here:
  // https://www.opengl.org/wiki/Shader_Subroutine#Runtime_selection
  // The subroutine selection is not actually part of the program but a
  // context state attribute -- given that, should we move this to
  // GLBindState somewhere? That's not as logical given the
  // "shader" object here, but it would ensure things are in
  // sync in that way, particularly if there's ever a case where
  // the state is saved/restored.

  for (auto itr : _activeSubroutines) {
    MAPD_CHECK_GL_ERROR(glUniformSubroutinesuiv(itr.first, itr.second.size(), &(itr.second[0])));
  }
}

bool GLShader::hasUniformBlockAttribute(const std::string& attrName) {
  for (auto& itr : _uniformBlockAttrs) {
    if (itr.second->activeAttrs.find(attrName) != itr.second->activeAttrs.end()) {
      return true;
    }
  }

  return false;
}

void GLShader::bindUniformBufferToBlock(const std::string& blockName,
                                        const GLUniformBufferShPtr& ubo,
                                        int idx,
                                        const GLShaderBlockLayoutShPtr& layoutPtr) {
  UniformBlockAttrInfo* blockAttr = _validateBlockAttr(blockName, ubo, idx);
  if (idx < 0) {
    blockAttr->bindBuffer(ubo->getId());
  } else {
    size_t numBytes = ubo->getNumBytesPerItem(layoutPtr);
    size_t offset = (layoutPtr ? ubo->getBufferLayoutData(layoutPtr).second : 0);
    blockAttr->bindBuffer(ubo->getId(), offset + idx * numBytes, numBytes);
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

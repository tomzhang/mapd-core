#include "TypeGL.h"
#include "MapDGL.h"
#include "GLRenderer.h"
#include "../../RenderError.h"

namespace Rendering {
namespace GL {

static bool supports64Ints() {
  CHECK(GLRenderer::getCurrentThreadRenderer());
  return GLEW_NV_vertex_attrib_integer_64bit;
}

std::string gl_type_to_string(const int glType) {
  switch (glType) {
    case GL_BOOL:
      return "GL_BOOL";
    case GL_BOOL_VEC2:
      return "GL_BOOL_VEC2";
    case GL_BOOL_VEC3:
      return "GL_BOOL_VEC3";
    case GL_BOOL_VEC4:
      return "GL_BOOL_VEC4";
    case GL_INT:
      return "GL_INT";
    case GL_INT_VEC2:
      return "GL_INT_VEC2";
    case GL_INT_VEC3:
      return "GL_INT_VEC3";
    case GL_INT_VEC4:
      return "GL_INT_VEC4";
    case GL_FLOAT:
      return "GL_FLOAT";
    case GL_FLOAT_VEC2:
      return "GL_FLOAT_VEC2";
    case GL_FLOAT_VEC3:
      return "GL_FLOAT_VEC3";
    case GL_FLOAT_VEC4:
      return "GL_FLOAT_VEC4";
    case GL_DOUBLE:
      return "GL_DOUBLE";
    case GL_DOUBLE_VEC2:
      return "GL_DOUBLE_VEC2";
    case GL_DOUBLE_VEC3:
      return "GL_DOUBLE_VEC3";
    case GL_DOUBLE_VEC4:
      return "GL_DOUBLE_VEC4";
    case GL_UNSIGNED_INT:
      return "GL_UNSIGNED_INT";
    case GL_UNSIGNED_INT_VEC2:
      return "GL_UNSIGNED_INT_VEC2";
    case GL_UNSIGNED_INT_VEC3:
      return "GL_UNSIGNED_INT_VEC3";
    case GL_UNSIGNED_INT_VEC4:
      return "GL_UNSIGNED_INT_VEC4";
    case GL_UNSIGNED_INT64_NV:
      return "GL_UNSIGNED_INT64_NV";
    case GL_UNSIGNED_INT64_VEC2_NV:
      return "GL_UNSIGNED_INT64_VEC2_NV";
    case GL_UNSIGNED_INT64_VEC3_NV:
      return "GL_UNSIGNED_INT64_VEC3_NV";
    case GL_UNSIGNED_INT64_VEC4_NV:
      return "GL_UNSIGNED_INT64_VEC4_NV";
    case GL_INT64_NV:
      return "GL_INT64_NV";
    case GL_INT64_VEC2_NV:
      return "GL_INT64_VEC2_NV";
    case GL_INT64_VEC3_NV:
      return "GL_INT64_VEC3_NV";
    case GL_INT64_VEC4_NV:
      return "GL_INT64_VEC4_NV";
  }

  return "undefined gl type: " + std::to_string(glType);
}

std::string BaseTypeGL::getTypeDefinesMacroForShader() {
  // creates all the macros for the GL types to be used
  // in shaders that need pre-processing of types in #if/#elif
  // This must be called with a gl context active
  // NOTE: can't use GL_ as the prefix as those are considered
  // reserved words, so using _GL_ as the prefix

  std::string rtn =
      ""
      "#define BOOL " +
      std::to_string(GL_BOOL) +
      "\n"
      "#define BOOL_VEC2 " +
      std::to_string(GL_BOOL_VEC2) +
      "\n"
      "#define BOOL_VEC3 " +
      std::to_string(GL_BOOL_VEC3) +
      "\n"
      "#define BOOL_VEC4 " +
      std::to_string(GL_BOOL_VEC4) +
      "\n"
      "#define INT " +
      std::to_string(GL_INT) +
      "\n"
      "#define INT_VEC2 " +
      std::to_string(GL_INT_VEC2) +
      "\n"
      "#define INT_VEC3 " +
      std::to_string(GL_INT_VEC3) +
      "\n"
      "#define INT_VEC4 " +
      std::to_string(GL_INT_VEC4) +
      "\n"
      "#define FLOAT " +
      std::to_string(GL_FLOAT) +
      "\n"
      "#define FLOAT_VEC2 " +
      std::to_string(GL_FLOAT_VEC2) +
      "\n"
      "#define FLOAT_VEC3 " +
      std::to_string(GL_FLOAT_VEC3) +
      "\n"
      "#define FLOAT_VEC4 " +
      std::to_string(GL_FLOAT_VEC4) +
      "\n"
      "#define DOUBLE " +
      std::to_string(GL_DOUBLE) +
      "\n"
      "#define DOUBLE_VEC2 " +
      std::to_string(GL_DOUBLE_VEC2) +
      "\n"
      "#define DOUBLE_VEC3 " +
      std::to_string(GL_DOUBLE_VEC3) +
      "\n"
      "#define DOUBLE_VEC4 " +
      std::to_string(GL_DOUBLE_VEC4) +
      "\n"
      "#define UNSIGNED_INT " +
      std::to_string(GL_UNSIGNED_INT) +
      "\n"
      "#define UNSIGNED_INT_VEC2 " +
      std::to_string(GL_UNSIGNED_INT_VEC2) +
      "\n"
      "#define UNSIGNED_INT_VEC3 " +
      std::to_string(GL_UNSIGNED_INT_VEC3) +
      "\n"
      "#define UNSIGNED_INT_VEC4 " +
      std::to_string(GL_UNSIGNED_INT_VEC4) + "\n";
  if (supports64Ints()) {
    rtn +=
        ""
        "#define UNSIGNED_INT64_NV " +
        std::to_string(GL_UNSIGNED_INT64_NV) +
        "\n"
        "#define UNSIGNED_INT64_VEC2_NV " +
        std::to_string(GL_UNSIGNED_INT64_VEC2_NV) +
        "\n"
        "#define UNSIGNED_INT64_VEC3_NV " +
        std::to_string(GL_UNSIGNED_INT64_VEC3_NV) +
        "\n"
        "#define UNSIGNED_INT64_VEC4_NV " +
        std::to_string(GL_UNSIGNED_INT64_VEC4_NV) +
        "\n"
        "#define INT64_NV " +
        std::to_string(GL_INT64_NV) +
        "\n"
        "#define INT64_VEC2_NV " +
        std::to_string(GL_INT64_VEC2_NV) +
        "\n"
        "#define INT64_VEC3_NV " +
        std::to_string(GL_INT64_VEC3_NV) +
        "\n"
        "#define INT64_VEC4_NV " +
        std::to_string(GL_INT64_VEC4_NV) + "\n";
  }
  return rtn + "\n";
}

std::string BaseTypeGL::getExtensionStr() {
  if (GLEW_NV_gpu_shader5) {
    return "#extension GL_NV_gpu_shader5 : enable\n#define EXT_GL_NV_gpu_shader5 1\n";
  } else if (GLEW_AMD_gpu_shader_int64) {
    return "#extension GL_AMD_gpu_shader_int64 : enable\n#define EXT_GL_AMD_gpu_shader_int64 1\n";
  }

  return "";
}

template <>
TypeGLMiddleLayer<float>::TypeGLMiddleLayer(const int baseGLType,
                                            const int glslGLType,
                                            const std::string& glslType,
                                            const bool useAsFloat,
                                            const bool normalize)
    : BaseTypeGL(baseGLType, glslGLType, glslType, true, false) {}

template <>
TypeGLMiddleLayer<double>::VertexAttribPtrFunc TypeGLMiddleLayer<double>::_getAltVertexAttribPointerFunc() const {
  return glVertexAttribLPointer;
}

template <>
TypeGLMiddleLayer<uint64_t>::VertexAttribPtrFunc TypeGLMiddleLayer<uint64_t>::_getAltVertexAttribPointerFunc() const {
  if (_baseGLType == GL_UNSIGNED_INT64_NV) {
    return glVertexAttribLPointer;
  } else {
    return glVertexAttribIPointer;
  }
}

template <>
TypeGLMiddleLayer<int64_t>::VertexAttribPtrFunc TypeGLMiddleLayer<int64_t>::_getAltVertexAttribPointerFunc() const {
  if (_baseGLType == GL_INT64_NV) {
    return glVertexAttribLPointer;
  } else {
    return glVertexAttribIPointer;
  }
}

/*****************
 * BOOL
 *****************/

template <>
TypeGL<bool, 1>::TypeGL(const std::set<std::string>& supportedExtensions)
    : TypeGLMiddleLayer<bool>(GL_BOOL, GL_BOOL, "bool") {}

template <>
TypeGL<bool, 2>::TypeGL(const std::set<std::string>& supportedExtensions)
    : TypeGLMiddleLayer<bool>(GL_BOOL, GL_BOOL_VEC2, "bvec2") {}

template <>
TypeGL<bool, 3>::TypeGL(const std::set<std::string>& supportedExtensions)
    : TypeGLMiddleLayer<bool>(GL_BOOL, GL_BOOL_VEC3, "bvec3") {}

template <>
TypeGL<bool, 4>::TypeGL(const std::set<std::string>& supportedExtensions)
    : TypeGLMiddleLayer<bool>(GL_BOOL, GL_BOOL_VEC4, "bvec4") {}

/*****************
 * UNSIGNED INT
 *****************/

template <>
TypeGL<unsigned int, 1>::TypeGL(const std::set<std::string>& supportedExtensions, bool useAsFloat, bool normalize)
    : TypeGLMiddleLayer<unsigned int>(GL_UNSIGNED_INT, GL_UNSIGNED_INT, "uint", useAsFloat, normalize) {}

template <>
TypeGL<unsigned int, 2>::TypeGL(const std::set<std::string>& supportedExtensions, bool useAsFloat, bool normalize)
    : TypeGLMiddleLayer<unsigned int>(GL_UNSIGNED_INT, GL_UNSIGNED_INT_VEC2, "uvec2", useAsFloat, normalize) {}

template <>
TypeGL<unsigned int, 3>::TypeGL(const std::set<std::string>& supportedExtensions, bool useAsFloat, bool normalize)
    : TypeGLMiddleLayer<unsigned int>(GL_UNSIGNED_INT, GL_UNSIGNED_INT_VEC3, "uvec3", useAsFloat, normalize) {}

template <>
TypeGL<unsigned int, 4>::TypeGL(const std::set<std::string>& supportedExtensions, bool useAsFloat, bool normalize)
    : TypeGLMiddleLayer<unsigned int>(GL_UNSIGNED_INT, GL_UNSIGNED_INT_VEC4, "uvec4", useAsFloat, normalize) {}

/*****************
 * INT
 *****************/

template <>
TypeGL<int, 1>::TypeGL(const std::set<std::string>& supportedExtensions, bool useAsFloat, bool normalize)
    : TypeGLMiddleLayer<int>(GL_INT, GL_INT, "int", useAsFloat, normalize) {}

template <>
TypeGL<int, 2>::TypeGL(const std::set<std::string>& supportedExtensions, bool useAsFloat, bool normalize)
    : TypeGLMiddleLayer<int>(GL_INT, GL_INT_VEC2, "ivec2", useAsFloat, normalize) {}

template <>
TypeGL<int, 3>::TypeGL(const std::set<std::string>& supportedExtensions, bool useAsFloat, bool normalize)
    : TypeGLMiddleLayer<int>(GL_INT, GL_INT_VEC3, "ivec3", useAsFloat, normalize) {}

template <>
TypeGL<int, 4>::TypeGL(const std::set<std::string>& supportedExtensions, bool useAsFloat, bool normalize)
    : TypeGLMiddleLayer<int>(GL_INT, GL_INT_VEC4, "ivec4", useAsFloat, normalize) {}

/*****************
 * FLOAT
 *****************/

template <>
TypeGL<float, 1>::TypeGL(const std::set<std::string>& supportedExtensions)
    : TypeGLMiddleLayer<float>(GL_FLOAT, GL_FLOAT, "float") {}

template <>
TypeGL<float, 2>::TypeGL(const std::set<std::string>& supportedExtensions)
    : TypeGLMiddleLayer<float>(GL_FLOAT, GL_FLOAT_VEC2, "vec2") {}

template <>
TypeGL<float, 3>::TypeGL(const std::set<std::string>& supportedExtensions)
    : TypeGLMiddleLayer<float>(GL_FLOAT, GL_FLOAT_VEC3, "vec3") {}

template <>
TypeGL<float, 4>::TypeGL(const std::set<std::string>& supportedExtensions)
    : TypeGLMiddleLayer<float>(GL_FLOAT, GL_FLOAT_VEC4, "vec4") {}

/*****************
 * DOUBLE
 *****************/

template <>
TypeGL<double, 1>::TypeGL(const std::set<std::string>& supportedExtensions, bool useAsFloat, bool normalize)
    : TypeGLMiddleLayer<double>(GL_DOUBLE, GL_DOUBLE, "double", useAsFloat, normalize) {}

template <>
TypeGL<double, 2>::TypeGL(const std::set<std::string>& supportedExtensions, bool useAsFloat, bool normalize)
    : TypeGLMiddleLayer<double>(GL_DOUBLE, GL_DOUBLE_VEC2, "dvec2", useAsFloat, normalize) {}

template <>
TypeGL<double, 3>::TypeGL(const std::set<std::string>& supportedExtensions, bool useAsFloat, bool normalize)
    : TypeGLMiddleLayer<double>(GL_DOUBLE, GL_DOUBLE_VEC3, "dvec3", useAsFloat, normalize) {}

template <>
TypeGL<double, 4>::TypeGL(const std::set<std::string>& supportedExtensions, bool useAsFloat, bool normalize)
    : TypeGLMiddleLayer<double>(GL_DOUBLE, GL_DOUBLE_VEC4, "dvec4", useAsFloat, normalize) {}

/*****************
 * UINT64
 *****************/

static bool supports64Ints(const std::set<std::string>& supportedExtensions) {
  return supportedExtensions.find("GL_NV_vertex_attrib_integer_64bit") != supportedExtensions.end();
}

template <>
TypeGL<uint64_t, 1>::TypeGL(const std::set<std::string>& supportedExtensions)
    : TypeGLMiddleLayer<uint64_t>(GL_UNSIGNED_INT, GL_UNSIGNED_INT, "uint") {
  if (supports64Ints(supportedExtensions)) {
    _baseGLType = GL_UNSIGNED_INT64_NV;
    _glslGLType = GL_UNSIGNED_INT64_NV;
    _glslType = "uint64_t";
  }
}

template <>
TypeGL<uint64_t, 2>::TypeGL(const std::set<std::string>& supportedExtensions)
    : TypeGLMiddleLayer<uint64_t>(GL_UNSIGNED_INT, GL_UNSIGNED_INT_VEC2, "uvec2") {
  if (supports64Ints(supportedExtensions)) {
    _baseGLType = GL_UNSIGNED_INT64_NV;
    _glslGLType = GL_UNSIGNED_INT64_VEC2_NV;
    _glslType = "u64vec2";
  }
}

template <>
TypeGL<uint64_t, 3>::TypeGL(const std::set<std::string>& supportedExtensions)
    : TypeGLMiddleLayer<uint64_t>(GL_UNSIGNED_INT, GL_UNSIGNED_INT_VEC3, "uvec3") {
  if (supports64Ints(supportedExtensions)) {
    _baseGLType = GL_UNSIGNED_INT64_NV;
    _glslGLType = GL_UNSIGNED_INT64_VEC3_NV;
    _glslType = "u64vec3";
  }
}

template <>
TypeGL<uint64_t, 4>::TypeGL(const std::set<std::string>& supportedExtensions)
    : TypeGLMiddleLayer<uint64_t>(GL_UNSIGNED_INT, GL_UNSIGNED_INT_VEC4, "uvec4") {
  if (supports64Ints(supportedExtensions)) {
    _baseGLType = GL_UNSIGNED_INT64_NV;
    _glslGLType = GL_UNSIGNED_INT64_VEC4_NV;
    _glslType = "u64vec4";
  }
}

/*****************
 * INT64
 *****************/

template <>
TypeGL<int64_t, 1>::TypeGL(const std::set<std::string>& supportedExtensions)
    : TypeGLMiddleLayer<int64_t>(GL_INT, GL_INT, "int") {
  if (supports64Ints(supportedExtensions)) {
    _baseGLType = GL_INT64_NV;
    _glslGLType = GL_INT64_NV;
    _glslType = "int64_t";
  }
}

template <>
TypeGL<int64_t, 2>::TypeGL(const std::set<std::string>& supportedExtensions)
    : TypeGLMiddleLayer<int64_t>(GL_INT, GL_INT_VEC2, "ivec2") {
  if (supports64Ints(supportedExtensions)) {
    _baseGLType = GL_INT64_NV;
    _glslGLType = GL_INT64_VEC2_NV;
    _glslType = "i64vec2";
  }
}

template <>
TypeGL<int64_t, 3>::TypeGL(const std::set<std::string>& supportedExtensions)
    : TypeGLMiddleLayer<int64_t>(GL_INT, GL_INT_VEC3, "ivec3") {
  if (supports64Ints(supportedExtensions)) {
    _baseGLType = GL_INT64_NV;
    _glslGLType = GL_INT64_VEC3_NV;
    _glslType = "i64vec3";
  }
}

template <>
TypeGL<int64_t, 4>::TypeGL(const std::set<std::string>& supportedExtensions)
    : TypeGLMiddleLayer<int64_t>(GL_INT, GL_INT_VEC4, "ivec4") {
  if (supports64Ints(supportedExtensions)) {
    _baseGLType = GL_INT64_NV;
    _glslGLType = GL_INT64_VEC4_NV;
    _glslType = "i64vec4";
  }
}

}  // namespace GL
}  // namespace Rendering

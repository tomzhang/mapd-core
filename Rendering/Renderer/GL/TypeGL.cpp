#include "TypeGL.h"
#include "MapDGL.h"

namespace Rendering {
namespace GL {

/*****************
 * BOOL
 *****************/

template <>
std::string TypeGL<bool, 1>::glslType() const {
  return "bool";
}

template <>
int TypeGL<bool, 1>::baseGLType() const {
  return GL_BOOL;
}

template <>
std::string TypeGL<bool, 2>::glslType() const {
  return "bvec2";
}

template <>
int TypeGL<bool, 2>::baseGLType() const {
  return GL_BOOL;
}

template <>
std::string TypeGL<bool, 3>::glslType() const {
  return "bvec3";
}

template <>
int TypeGL<bool, 3>::baseGLType() const {
  return GL_BOOL;
}

template <>
std::string TypeGL<bool, 4>::glslType() const {
  return "bvec4";
}

template <>
int TypeGL<bool, 4>::baseGLType() const {
  return GL_BOOL;
}

/*****************
 * UNSIGNED INT
 *****************/

template <>
std::string TypeGL<unsigned int, 1>::glslType() const {
  return "uint";
}

template <>
int TypeGL<unsigned int, 1>::baseGLType() const {
  return GL_UNSIGNED_INT;
}

/*****************
 * INT
 *****************/

template <>
std::string TypeGL<int, 1>::glslType() const {
  return "int";
}

template <>
int TypeGL<int, 1>::baseGLType() const {
  return GL_INT;
}

template <>
std::string TypeGL<int, 2>::glslType() const {
  return (_useAsFloat ? "vec2" : "ivec2");
}

template <>
int TypeGL<int, 2>::baseGLType() const {
  return GL_INT;
}

template <>
std::string TypeGL<int, 3>::glslType() const {
  return (_useAsFloat ? "vec3" : "ivec3");
}

template <>
int TypeGL<int, 3>::baseGLType() const {
  return GL_INT;
}

template <>
std::string TypeGL<int, 4>::glslType() const {
  return (_useAsFloat ? "vec4" : "ivec4");
}

template <>
int TypeGL<int, 4>::baseGLType() const {
  return GL_INT;
}

/*****************
 * FLOAT
 *****************/

template <>
std::string TypeGL<float, 1>::glslType() const {
  return "float";
}

template <>
int TypeGL<float, 1>::baseGLType() const {
  return GL_FLOAT;
}

template <>
void TypeGL<float, 1>::setUseAsFloat(bool bindAsFloat, bool normalize) {
  _useAsFloat = true;
  _normalize = false;
}

template <>
std::string TypeGL<float, 2>::glslType() const {
  return "vec2";
}

template <>
int TypeGL<float, 2>::baseGLType() const {
  return GL_FLOAT;
}

template <>
void TypeGL<float, 2>::setUseAsFloat(bool bindAsFloat, bool normalize) {
  _useAsFloat = true;
  _normalize = false;
  // _bindFunc = std::bind(glVertexAttribPointer, std::placeholders::_1, 2, GL_FLOAT, GL_FALSE, std::placeholders::_2,
  // std::placeholders::_3);
}

template <>
std::string TypeGL<float, 3>::glslType() const {
  return "vec3";
}

template <>
int TypeGL<float, 3>::baseGLType() const {
  return GL_FLOAT;
}

template <>
void TypeGL<float, 3>::setUseAsFloat(bool bindAsFloat, bool normalize) {
  _useAsFloat = true;
  _normalize = false;
  // _bindFunc = std::bind(glVertexAttribPointer, std::placeholders::_1, 3, GL_FLOAT, GL_FALSE, std::placeholders::_2,
  // std::placeholders::_3);
}

template <>
std::string TypeGL<float, 4>::glslType() const {
  return "vec4";
}

template <>
int TypeGL<float, 4>::baseGLType() const {
  return GL_FLOAT;
}

template <>
void TypeGL<float, 4>::setUseAsFloat(bool bindAsFloat, bool normalize) {
  _useAsFloat = true;
  _normalize = false;
}

/*****************
 * DOUBLE
 *****************/

template <>
std::string TypeGL<double, 1>::glslType() const {
  return "double";
}

template <>
int TypeGL<double, 1>::baseGLType() const {
  return GL_DOUBLE;
}

template <>
TypeGL<double, 1>::VertexAttribPtrFunc TypeGL<double, 1>::_getAltVertexAttribPointerFunc() const {
  return glVertexAttribLPointer;
}

template <>
std::string TypeGL<double, 2>::glslType() const {
  return (_useAsFloat ? "vec2" : "dvec2");
}

template <>
int TypeGL<double, 2>::baseGLType() const {
  return GL_DOUBLE;
}

template <>
TypeGL<double, 2>::VertexAttribPtrFunc TypeGL<double, 2>::_getAltVertexAttribPointerFunc() const {
  return glVertexAttribLPointer;
}

template <>
std::string TypeGL<double, 3>::glslType() const {
  return (_useAsFloat ? "vec3" : "dvec3");
}

template <>
int TypeGL<double, 3>::baseGLType() const {
  return GL_DOUBLE;
}

template <>
TypeGL<double, 3>::VertexAttribPtrFunc TypeGL<double, 3>::_getAltVertexAttribPointerFunc() const {
  return glVertexAttribLPointer;
}

template <>
std::string TypeGL<double, 4>::glslType() const {
  return (_useAsFloat ? "vec4" : "dvec4");
}

template <>
int TypeGL<double, 4>::baseGLType() const {
  return GL_DOUBLE;
}

template <>
TypeGL<double, 4>::VertexAttribPtrFunc TypeGL<double, 4>::_getAltVertexAttribPointerFunc() const {
  return glVertexAttribLPointer;
}

/*****************
 * UINT8
 *****************/

template <>
std::string TypeGL<uint8_t, 1>::glslType() const {
  return "uint";
}

template <>
int TypeGL<uint8_t, 1>::baseGLType() const {
  return GL_UNSIGNED_BYTE;
}

template <>
std::string TypeGL<uint8_t, 2>::glslType() const {
  return (_useAsFloat ? "vec2" : "uvec2");
}

template <>
int TypeGL<uint8_t, 2>::baseGLType() const {
  return GL_UNSIGNED_BYTE;
}

template <>
std::string TypeGL<uint8_t, 3>::glslType() const {
  return (_useAsFloat ? "vec3" : "uvec3");
}

template <>
int TypeGL<uint8_t, 3>::baseGLType() const {
  return GL_UNSIGNED_BYTE;
}
template <>
std::string TypeGL<uint8_t, 4>::glslType() const {
  return (_useAsFloat ? "vec4" : "uvec4");
}

template <>
int TypeGL<uint8_t, 4>::baseGLType() const {
  return GL_UNSIGNED_BYTE;
}

/*****************
 * UINT64
 *****************/

template <>
std::string TypeGL<uint64_t, 1>::glslType() const {
  // TODO(croot): make a specific class
  // for these types and check for the
  // existence of the extension at construction,
  // or better yet, make it a static const
  // member variable, if possible to reduce the
  // if/else ? Not sure this is doable. Even
  // with the static const variable, you still
  // would need to do an if/else unless we're
  // able to create an additional template
  // specialization somehow, but i'm not sure
  // how to do that.
  if (GLEW_NV_vertex_attrib_integer_64bit) {
    return "uint64_t";
  } else {
    return "uint";
  }
}

template <>
int TypeGL<uint64_t, 1>::baseGLType() const {
  if (GLEW_NV_vertex_attrib_integer_64bit) {
    return GL_UNSIGNED_INT64_NV;
  } else {
    return GL_UNSIGNED_INT;
  }
}

template <>
std::string TypeGL<uint64_t, 2>::glslType() const {
  if (GLEW_NV_vertex_attrib_integer_64bit) {
    return "u64vec2";
  } else {
    return "uvec2";
  }
}

template <>
int TypeGL<uint64_t, 2>::baseGLType() const {
  if (GLEW_NV_vertex_attrib_integer_64bit) {
    return GL_UNSIGNED_INT64_NV;
  } else {
    return GL_UNSIGNED_INT;
  }
}

template <>
std::string TypeGL<uint64_t, 3>::glslType() const {
  if (GLEW_NV_vertex_attrib_integer_64bit) {
    return "u64vec3";
  } else {
    return "uvec3";
  }
}

template <>
int TypeGL<uint64_t, 3>::baseGLType() const {
  if (GLEW_NV_vertex_attrib_integer_64bit) {
    return GL_UNSIGNED_INT64_NV;
  } else {
    return GL_UNSIGNED_INT;
  }
}

template <>
std::string TypeGL<uint64_t, 4>::glslType() const {
  if (GLEW_NV_vertex_attrib_integer_64bit) {
    return "u64vec4";
  } else {
    return "uvec4";
  }
}

template <>
int TypeGL<uint64_t, 4>::baseGLType() const {
  if (GLEW_NV_vertex_attrib_integer_64bit) {
    return GL_UNSIGNED_INT64_NV;
  } else {
    return GL_UNSIGNED_INT;
  }
}

/*****************
 * INT64
 *****************/

template <>
std::string TypeGL<int64_t, 1>::glslType() const {
  // TODO(croot): make a specific class
  // for these types and check for the
  // existence of the extension at construction,
  // or better yet, make it a static const
  // member variable, if possible to reduce the
  // if/else ? Not sure this is doable. Even
  // with the static const variable, you still
  // would need to do an if/else unless we're
  // able to create an additional template
  // specialization somehow, but i'm not sure
  // how to do that.
  if (GLEW_NV_vertex_attrib_integer_64bit) {
    return "int64_t";
  } else {
    return "int";
  }
}

template <>
int TypeGL<int64_t, 1>::baseGLType() const {
  if (GLEW_NV_vertex_attrib_integer_64bit) {
    return GL_INT64_NV;
  } else {
    return GL_INT;
  }
}

template <>
std::string TypeGL<int64_t, 2>::glslType() const {
  if (GLEW_NV_vertex_attrib_integer_64bit) {
    return "i64vec2";
  } else {
    return "ivec2";
  }
}

template <>
int TypeGL<int64_t, 2>::baseGLType() const {
  if (GLEW_NV_vertex_attrib_integer_64bit) {
    return GL_INT64_NV;
  } else {
    return GL_INT;
  }
}

template <>
std::string TypeGL<int64_t, 3>::glslType() const {
  if (GLEW_NV_vertex_attrib_integer_64bit) {
    return "i64vec3";
  } else {
    return "ivec3";
  }
}

template <>
int TypeGL<int64_t, 3>::baseGLType() const {
  if (GLEW_NV_vertex_attrib_integer_64bit) {
    return GL_INT64_NV;
  } else {
    return GL_INT;
  }
}

template <>
std::string TypeGL<int64_t, 4>::glslType() const {
  if (GLEW_NV_vertex_attrib_integer_64bit) {
    return "i64vec4";
  } else {
    return "ivec4";
  }
}

template <>
int TypeGL<int64_t, 4>::baseGLType() const {
  if (GLEW_NV_vertex_attrib_integer_64bit) {
    return GL_INT64_NV;
  } else {
    return GL_INT;
  }
}

}  // namespace GL
}  // namespace Rendering

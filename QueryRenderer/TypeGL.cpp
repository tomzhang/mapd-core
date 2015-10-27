#include "TypeGL.h"

using namespace MapD_Renderer;

/*****************
 * UNSIGNED INT
 *****************/

template <>
std::string TypeGL<unsigned int, 1>::glslType() {
  return "uint";
}

template <>
int TypeGL<unsigned int, 1>::baseGLType() {
  return GL_UNSIGNED_INT;
}

/*****************
 * INT
 *****************/

template <>
std::string TypeGL<int, 1>::glslType() {
  return "int";
}

template <>
int TypeGL<int, 1>::baseGLType() {
  return GL_INT;
}

template <>
std::string TypeGL<int, 2>::glslType() {
  return (_useAsFloat ? "vec2" : "ivec2");
}

template <>
int TypeGL<int, 2>::baseGLType() {
  return GL_INT;
}

template <>
std::string TypeGL<int, 3>::glslType() {
  return (_useAsFloat ? "vec3" : "ivec3");
}

template <>
int TypeGL<int, 3>::baseGLType() {
  return GL_INT;
}

template <>
std::string TypeGL<int, 4>::glslType() {
  return (_useAsFloat ? "vec4" : "ivec4");
}

template <>
int TypeGL<int, 4>::baseGLType() {
  return GL_INT;
}

/*****************
 * FLOAT
 *****************/

template <>
std::string TypeGL<float, 1>::glslType() {
  return "float";
}

template <>
int TypeGL<float, 1>::baseGLType() {
  return GL_FLOAT;
}

template <>
void TypeGL<float, 1>::setUseAsFloat(bool bindAsFloat, bool normalize) {
  _useAsFloat = true;
  _normalize = false;
}

template <>
std::string TypeGL<float, 2>::glslType() {
  return "vec2";
}

template <>
int TypeGL<float, 2>::baseGLType() {
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
std::string TypeGL<float, 3>::glslType() {
  return "vec3";
}

template <>
int TypeGL<float, 3>::baseGLType() {
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
std::string TypeGL<float, 4>::glslType() {
  return "vec4";
}

template <>
int TypeGL<float, 4>::baseGLType() {
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
std::string TypeGL<double, 1>::glslType() {
  return "double";
}

template <>
int TypeGL<double, 1>::baseGLType() {
  return GL_DOUBLE;
}

template <>
TypeGL<double, 1>::VertexAttribPtrFunc TypeGL<double, 1>::_getAltVertexAttribPointerFunc() {
  return glVertexAttribLPointer;
}

template <>
std::string TypeGL<double, 2>::glslType() {
  return (_useAsFloat ? "vec2" : "dvec2");
}

template <>
int TypeGL<double, 2>::baseGLType() {
  return GL_DOUBLE;
}

template <>
TypeGL<double, 2>::VertexAttribPtrFunc TypeGL<double, 2>::_getAltVertexAttribPointerFunc() {
  return glVertexAttribLPointer;
}

template <>
std::string TypeGL<double, 3>::glslType() {
  return (_useAsFloat ? "vec3" : "dvec3");
}

template <>
int TypeGL<double, 3>::baseGLType() {
  return GL_DOUBLE;
}

template <>
TypeGL<double, 3>::VertexAttribPtrFunc TypeGL<double, 3>::_getAltVertexAttribPointerFunc() {
  return glVertexAttribLPointer;
}

template <>
std::string TypeGL<double, 4>::glslType() {
  return (_useAsFloat ? "vec4" : "dvec4");
}

template <>
int TypeGL<double, 4>::baseGLType() {
  return GL_DOUBLE;
}

template <>
TypeGL<double, 4>::VertexAttribPtrFunc TypeGL<double, 4>::_getAltVertexAttribPointerFunc() {
  return glVertexAttribLPointer;
}

/*****************
 * UINT8
 *****************/

template <>
std::string TypeGL<uint8_t, 1>::glslType() {
  return "uint";
}

template <>
int TypeGL<uint8_t, 1>::baseGLType() {
  return GL_UNSIGNED_BYTE;
}

template <>
std::string TypeGL<uint8_t, 2>::glslType() {
  return (_useAsFloat ? "vec2" : "uvec2");
}

template <>
int TypeGL<uint8_t, 2>::baseGLType() {
  return GL_UNSIGNED_BYTE;
}

template <>
std::string TypeGL<uint8_t, 3>::glslType() {
  return (_useAsFloat ? "vec3" : "uvec3");
}

template <>
int TypeGL<uint8_t, 3>::baseGLType() {
  return GL_UNSIGNED_BYTE;
}
template <>
std::string TypeGL<uint8_t, 4>::glslType() {
  return (_useAsFloat ? "vec4" : "uvec4");
}

template <>
int TypeGL<uint8_t, 4>::baseGLType() {
  return GL_UNSIGNED_BYTE;
}

/*****************
 * UINT64
 *****************/

template <>
std::string TypeGL<uint64_t, 1>::glslType() {
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
int TypeGL<uint64_t, 1>::baseGLType() {
  if (GLEW_NV_vertex_attrib_integer_64bit) {
    return GL_UNSIGNED_INT64_NV;
  } else {
    return GL_UNSIGNED_INT;
  }
}

// template <>
// int TypeGL<uint64_t, 1>::numGLSLBytes() {
//   if (GLEW_NV_vertex_attrib_integer_64bit) {
//     return numBytes();
//   } else {
//     return sizeof(unsigned int) * numComponents();
//   }
// }

template <>
std::string TypeGL<uint64_t, 2>::glslType() {
  if (GLEW_NV_vertex_attrib_integer_64bit) {
    return "u64vec2";
  } else {
    return "uvec2";
  }
}

template <>
int TypeGL<uint64_t, 2>::baseGLType() {
  if (GLEW_NV_vertex_attrib_integer_64bit) {
    return GL_UNSIGNED_INT64_NV;
  } else {
    return GL_UNSIGNED_INT;
  }
}

// template <>
// int TypeGL<uint64_t, 2>::numGLSLBytes() {
//   if (GLEW_NV_vertex_attrib_integer_64bit) {
//     return numBytes();
//   } else {
//     return sizeof(unsigned int) * numComponents();
//   }
// }

template <>
std::string TypeGL<uint64_t, 3>::glslType() {
  if (GLEW_NV_vertex_attrib_integer_64bit) {
    return "u64vec3";
  } else {
    return "uvec3";
  }
}

template <>
int TypeGL<uint64_t, 3>::baseGLType() {
  if (GLEW_NV_vertex_attrib_integer_64bit) {
    return GL_UNSIGNED_INT64_NV;
  } else {
    return GL_UNSIGNED_INT;
  }
}

// template <>
// int TypeGL<uint64_t, 3>::numGLSLBytes() {
//   if (GLEW_NV_vertex_attrib_integer_64bit) {
//     return numBytes();
//   } else {
//     return sizeof(unsigned int) * numComponents();
//   }
// }

template <>
std::string TypeGL<uint64_t, 4>::glslType() {
  if (GLEW_NV_vertex_attrib_integer_64bit) {
    return "u64vec4";
  } else {
    return "uvec4";
  }
}

template <>
int TypeGL<uint64_t, 4>::baseGLType() {
  if (GLEW_NV_vertex_attrib_integer_64bit) {
    return GL_UNSIGNED_INT64_NV;
  } else {
    return GL_UNSIGNED_INT;
  }
}

// template <>
// int TypeGL<uint64_t, 4>::numGLSLBytes() {
//   if (GLEW_NV_vertex_attrib_integer_64bit) {
//     return numBytes();
//   } else {
//     return sizeof(unsigned int) * numComponents();
//   }
// }

/*****************
 * INT64
 *****************/

template <>
std::string TypeGL<int64_t, 1>::glslType() {
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
    return "iint";
  }
}

template <>
int TypeGL<int64_t, 1>::baseGLType() {
  if (GLEW_NV_vertex_attrib_integer_64bit) {
    return GL_INT64_NV;
  } else {
    return GL_INT;
  }
}

// template <>
// int TypeGL<int64_t, 1>::numGLSLBytes() {
//   if (GLEW_NV_vertex_attrib_integer_64bit) {
//     return numBytes();
//   } else {
//     return sizeof(int) * numComponents();
//   }
// }

template <>
std::string TypeGL<int64_t, 2>::glslType() {
  if (GLEW_NV_vertex_attrib_integer_64bit) {
    return "i64vec2";
  } else {
    return "ivec2";
  }
}

template <>
int TypeGL<int64_t, 2>::baseGLType() {
  if (GLEW_NV_vertex_attrib_integer_64bit) {
    return GL_INT64_NV;
  } else {
    return GL_INT;
  }
}

// template <>
// int TypeGL<int64_t, 2>::numGLSLBytes() {
//   if (GLEW_NV_vertex_attrib_integer_64bit) {
//     return numBytes();
//   } else {
//     return sizeof(int) * numComponents();
//   }
// }

template <>
std::string TypeGL<int64_t, 3>::glslType() {
  if (GLEW_NV_vertex_attrib_integer_64bit) {
    return "i64vec3";
  } else {
    return "ivec3";
  }
}

template <>
int TypeGL<int64_t, 3>::baseGLType() {
  if (GLEW_NV_vertex_attrib_integer_64bit) {
    return GL_INT64_NV;
  } else {
    return GL_INT;
  }
}

// template <>
// int TypeGL<int64_t, 3>::numGLSLBytes() {
//   if (GLEW_NV_vertex_attrib_integer_64bit) {
//     return numBytes();
//   } else {
//     return sizeof(int) * numComponents();
//   }
// }

template <>
std::string TypeGL<int64_t, 4>::glslType() {
  if (GLEW_NV_vertex_attrib_integer_64bit) {
    return "i64vec4";
  } else {
    return "ivec4";
  }
}

template <>
int TypeGL<int64_t, 4>::baseGLType() {
  if (GLEW_NV_vertex_attrib_integer_64bit) {
    return GL_INT64_NV;
  } else {
    return GL_INT;
  }
}

// template <>
// int TypeGL<int64_t, 4>::numGLSLBytes() {
//   if (GLEW_NV_vertex_attrib_integer_64bit) {
//     return numBytes();
//   } else {
//     return sizeof(int) * numComponents();
//   }
// }

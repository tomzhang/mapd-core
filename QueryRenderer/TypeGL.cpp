#include "TypeGL.h"

using namespace MapD_Renderer;



template <>
std::string TypeGL<unsigned int, 1>::glslType() {
    return "uint";
}

template<>
int TypeGL<unsigned int, 1>::baseGLType() {
    return GL_UNSIGNED_INT;
}

template <>
std::string TypeGL<int, 1>::glslType() {
    return "int";
}

template<>
int TypeGL<int, 1>::baseGLType() {
    return GL_INT;
}

template <>
std::string TypeGL<int, 2>::glslType() {
    return (_useAsFloat ? "vec2" : "ivec2");
}

template<>
int TypeGL<int, 2>::baseGLType() {
    return GL_INT;
}

template <>
std::string TypeGL<int, 3>::glslType() {
    return (_useAsFloat ? "vec3" : "ivec3");
}

template<>
int TypeGL<int, 3>::baseGLType() {
    return GL_INT;
}

template <>
std::string TypeGL<int, 4>::glslType() {
    return (_useAsFloat ? "vec4" : "ivec4");
}

template<>
int TypeGL<int, 4>::baseGLType() {
    return GL_INT;
}





template <>
std::string TypeGL<float, 1>::glslType() {
    return "float";
}

template<>
int TypeGL<float, 1>::baseGLType() {
    return GL_FLOAT;
}

template <>
void TypeGL<float, 1>::setUseAsFloat(bool bindAsFloat, bool normalize) {
    _useAsFloat = true;
    _normalize = false;
    // _bindFunc = std::bind(glVertexAttribPointer, std::placeholders::_1, 1, GL_FLOAT, GL_FALSE, std::placeholders::_2, std::placeholders::_3);
}

template <>
std::string TypeGL<float, 2>::glslType() {
    return "vec2";
}

template<>
int TypeGL<float, 2>::baseGLType() {
    return GL_FLOAT;
}

template <>
void TypeGL<float, 2>::setUseAsFloat(bool bindAsFloat, bool normalize) {
    _useAsFloat = true;
    _normalize = false;
    // _bindFunc = std::bind(glVertexAttribPointer, std::placeholders::_1, 2, GL_FLOAT, GL_FALSE, std::placeholders::_2, std::placeholders::_3);
}

template <>
std::string TypeGL<float, 3>::glslType() {
    return "vec3";
}

template<>
int TypeGL<float, 3>::baseGLType() {
    return GL_FLOAT;
}

template <>
void TypeGL<float, 3>::setUseAsFloat(bool bindAsFloat, bool normalize) {
    _useAsFloat = true;
    _normalize = false;
    // _bindFunc = std::bind(glVertexAttribPointer, std::placeholders::_1, 3, GL_FLOAT, GL_FALSE, std::placeholders::_2, std::placeholders::_3);
}

template <>
std::string TypeGL<float, 4>::glslType() {
    return "vec4";
}

template<>
int TypeGL<float, 4>::baseGLType() {
    return GL_FLOAT;
}

template <>
void TypeGL<float, 4>::setUseAsFloat(bool bindAsFloat, bool normalize) {
    _useAsFloat = true;
    _normalize = false;
    // _bindFunc = std::bind(glVertexAttribPointer, std::placeholders::_1, 4, GL_FLOAT, GL_FALSE, std::placeholders::_2, std::placeholders::_3);
}





template <>
std::string TypeGL<double, 1>::glslType() {
    return "double";
}

template<>
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

template<>
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

template<>
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

template<>
int TypeGL<double, 4>::baseGLType() {
    return GL_DOUBLE;
}

template <>
TypeGL<double, 4>::VertexAttribPtrFunc TypeGL<double, 4>::_getAltVertexAttribPointerFunc() {
    return glVertexAttribLPointer;
}








template <>
std::string TypeGL<uint8_t, 1>::glslType() {
    return "uint";
}

template<>
int TypeGL<uint8_t, 1>::baseGLType() {
    return GL_UNSIGNED_BYTE;
}

template <>
std::string TypeGL<uint8_t, 2>::glslType() {
    return (_useAsFloat ? "vec2" : "uvec2");
}

template<>
int TypeGL<uint8_t, 2>::baseGLType() {
    return GL_UNSIGNED_BYTE;
}

template <>
std::string TypeGL<uint8_t, 3>::glslType() {
    return (_useAsFloat ? "vec3" : "uvec3");
}

template<>
int TypeGL<uint8_t, 3>::baseGLType() {
    return GL_UNSIGNED_BYTE;
}
template <>
std::string TypeGL<uint8_t, 4>::glslType() {
    return (_useAsFloat ? "vec4" : "uvec4");
}

template<>
int TypeGL<uint8_t, 4>::baseGLType() {
    return GL_UNSIGNED_BYTE;
}












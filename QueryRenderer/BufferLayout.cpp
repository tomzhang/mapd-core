#include "BufferLayout.h"

using namespace MapD_Renderer;

BufferAttrType MapD_Renderer::getBufferAttrType(unsigned int a, int numComponents) {
  RUNTIME_EX_ASSERT(numComponents == 1, "Only 1 component of unsigned ints are currently supported.");

  return BufferAttrType::UINT;
}

BufferAttrType MapD_Renderer::getBufferAttrType(int a, int numComponents) {
  switch (numComponents) {
    case 1:
      return BufferAttrType::INT;
    case 2:
      return BufferAttrType::VEC2I;
    case 3:
      return BufferAttrType::VEC3I;
    case 4:
      return BufferAttrType::VEC4I;
    default:
      THROW_RUNTIME_EX("Invalid number of components " + std::to_string(numComponents) + ". Need 1-4 components.");
  }

  return BufferAttrType::INT;
}

BufferAttrType MapD_Renderer::getBufferAttrType(float a, int numComponents) {
  switch (numComponents) {
    case 1:
      return BufferAttrType::FLOAT;
    case 2:
      return BufferAttrType::VEC2F;
    case 3:
      return BufferAttrType::VEC3F;
    case 4:
      return BufferAttrType::VEC4F;
    default:
      THROW_RUNTIME_EX("Invalid number of components " + std::to_string(numComponents) + ". Need 1-4 components.");
  }

  return BufferAttrType::FLOAT;
}

BufferAttrType MapD_Renderer::getBufferAttrType(double a, int numComponents) {
  switch (numComponents) {
    case 1:
      return BufferAttrType::DOUBLE;
    case 2:
      return BufferAttrType::VEC2D;
    case 3:
      return BufferAttrType::VEC3D;
    case 4:
      return BufferAttrType::VEC4D;
    default:
      THROW_RUNTIME_EX("Invalid number of components " + std::to_string(numComponents) + ". Need 1-4 components.");
  }

  return BufferAttrType::DOUBLE;
}

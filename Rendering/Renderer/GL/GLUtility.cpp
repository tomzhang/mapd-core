#include "GLUtility.h"
#include "../../RenderError.h"

namespace Rendering {
namespace GL {

size_t pixelTypeSize(GLenum pixelType) {
  switch (pixelType) {
    case GL_UNSIGNED_BYTE:
      return 1;
    case GL_BYTE:
      return 1;
    case GL_UNSIGNED_SHORT:
      return 2;
    case GL_SHORT:
      return 2;
    case GL_UNSIGNED_INT:
      return 4;
    case GL_INT:
      return 4;
    case GL_FLOAT:
      return 4;
    case GL_UNSIGNED_BYTE_3_3_2:
      return 8;
    case GL_UNSIGNED_BYTE_2_3_3_REV:
      return 8;
    case GL_UNSIGNED_SHORT_5_6_5:
      return 16;
    case GL_UNSIGNED_SHORT_5_6_5_REV:
      return 16;
    case GL_UNSIGNED_SHORT_4_4_4_4:
      return 16;
    case GL_UNSIGNED_SHORT_4_4_4_4_REV:
      return 16;
    case GL_UNSIGNED_SHORT_5_5_5_1:
      return 16;
    case GL_UNSIGNED_SHORT_1_5_5_5_REV:
      return 16;
    case GL_UNSIGNED_INT_8_8_8_8:
      return 32;
    case GL_UNSIGNED_INT_8_8_8_8_REV:
      return 32;
    case GL_UNSIGNED_INT_10_10_10_2:
      return 32;
    case GL_UNSIGNED_INT_2_10_10_10_REV:
      return 32;
    default:
      std::stringstream stream;
      stream << "0x" << std::hex << pixelType;
      THROW_RUNTIME_EX("Unsupported pixel type: " + stream.str());
  }

  return 0;
}

}  // namespace GL
}  // namespace Rendering

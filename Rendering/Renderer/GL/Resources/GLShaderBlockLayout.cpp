#include "GLShaderBlockLayout.h"

namespace Rendering {
namespace GL {
namespace Resources {

int GLShaderBlockLayout::uniformBufferOffsetAlignment = -1;
int GLShaderBlockLayout::maxUniformBlockSize = -1;

// all shader block layouts must be interleaved, but there are rules for
// the stride/offset for each layout type, hence the reasoning for
// a specific shader block layout.
GLShaderBlockLayout::GLShaderBlockLayout(const GLShaderShPtr& shaderPtr, size_t blockByteSize, LayoutType layoutType)
    : GLBaseBufferLayout(GLBufferLayoutType::INTERLEAVED), _shaderPtr(shaderPtr), _layoutType(layoutType) {
  _initNumAlignmentBytes();
  _initMaxUniformBlockSize();
  RUNTIME_EX_ASSERT(static_cast<int>(blockByteSize) <= maxUniformBlockSize,
                    "Block size of " + std::to_string(blockByteSize) + " bytes exceeds the maximum size of " +
                        std::to_string(maxUniformBlockSize) + " bytes.");
  _vertexByteSize = blockByteSize;
}

GLShaderBlockLayout::~GLShaderBlockLayout() {
}

void GLShaderBlockLayout::bindToShader(GLShader* activeShader,
                                       int numActiveBufferItems,
                                       const std::string& attr,
                                       const std::string& shaderAttr) {
  THROW_RUNTIME_EX(
      "A GLShaderBlockLayout cannot be bound to a shader. It is instead defined by a shader via a shader storage "
      "block.")
}

void GLShaderBlockLayout::_initNumAlignmentBytes() {
  if (uniformBufferOffsetAlignment < 0) {
    MAPD_CHECK_GL_ERROR(glGetIntegerv(GL_UNIFORM_BUFFER_OFFSET_ALIGNMENT, &uniformBufferOffsetAlignment));
    CHECK(uniformBufferOffsetAlignment > 0);
  }
}

void GLShaderBlockLayout::_initMaxUniformBlockSize() {
  if (maxUniformBlockSize < 0) {
    MAPD_CHECK_GL_ERROR(glGetIntegerv(GL_MAX_UNIFORM_BLOCK_SIZE, &maxUniformBlockSize));
  }
}

size_t GLShaderBlockLayout::getNumAlignmentBytes() {
  // TODO(croot): if this is called outside of this class, should I throw
  // a runtime error if the alignment bytes is un-initialized instead? This
  // would force that a GLShaderBlockLayout be defined first, which means
  // a context would be defined.
  _initNumAlignmentBytes();
  return uniformBufferOffsetAlignment;
}

}  // namespace Resources
}  // namespace GL
}  // namespace Rendering

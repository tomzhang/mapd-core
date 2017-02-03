#include "GLUniformBuffer.h"

namespace Rendering {
namespace GL {
namespace Resources {

static size_t calcNumBytesPerItem(size_t bytesInBlock, size_t alignmentBytes, size_t numItems) {
  if (numItems > 1) {
    size_t mod = bytesInBlock % alignmentBytes;

    if (!mod) {
      return bytesInBlock;
    } else {
      return bytesInBlock + alignmentBytes - mod;
    }
  }

  return numItems * alignmentBytes;
}

GLUniformBuffer::GLUniformBuffer(const RendererWkPtr& rendererPtr,
                                 BufferAccessType accessType,
                                 BufferAccessFreq accessFreq)
    : GLLayoutBuffer<GLShaderBlockLayout>(rendererPtr,
                                          GLResourceType::UNIFORM_BUFFER,
                                          GLBufferType::UNIFORM_BUFFER,
                                          GL_UNIFORM_BUFFER,
                                          accessType,
                                          accessFreq) {}

GLUniformBuffer::GLUniformBuffer(const RendererWkPtr& rendererPtr,
                                 size_t numBytes,
                                 BufferAccessType accessType,
                                 BufferAccessFreq accessFreq)
    : GLUniformBuffer(rendererPtr, accessType, accessFreq) {
  size_t alignBytes = getNumAlignmentBytes();
  size_t mod = numBytes % alignBytes;
  if (mod > 0) {
    // TODO(croot): should I just throw a runtime error instead?
    size_t diff = numBytes + alignBytes - mod;
    LOG(WARNING) << "Trying to allocate a uniform buffer with " + std::to_string(numBytes) +
                        " bytes, but it does not alignment requirement of " + std::to_string(alignBytes) + " bytes (" +
                        std::to_string(numBytes) + "/" + std::to_string(alignBytes) +
                        " != 0). Increasing the allocation to " + std::to_string(diff) +
                        " bytes to meet that requirement";
    numBytes = diff;
  }
  GLBaseBuffer::bufferData(nullptr, numBytes);
}

GLUniformBuffer::~GLUniformBuffer() {
  cleanupResource();
}

void GLUniformBuffer::_makeEmpty() {
  GLLayoutBuffer<GLShaderBlockLayout>::_makeEmpty();
}

void GLUniformBuffer::_validateBufferLayout(size_t numBytes,
                                            size_t offsetBytes,
                                            const GLBufferLayoutShPtr& layoutPtr,
                                            bool replaceExistingLayout,
                                            const std::string& errPrefix) {
  GLLayoutManagerBuffer::_validateBufferLayout(numBytes, offsetBytes, layoutPtr, replaceExistingLayout, errPrefix);

  auto sblayoutPtr = std::dynamic_pointer_cast<GLShaderBlockLayout>(layoutPtr);
  RUNTIME_EX_ASSERT(sblayoutPtr, "GLUniformBuffers only support GLShaderBlockLayout. Validation failed.");
}

size_t GLUniformBuffer::getNumBytesInBlock(const GLShaderBlockLayoutShPtr& layoutPtr) const {
  auto layout = _getBufferLayoutToUse(layoutPtr, "Cannot get number of bytes in uniform buffer block. ");
  return (layout ? layout->getNumBytesInBlock() : 0);
}

size_t GLUniformBuffer::getNumBytesPerItem(const GLShaderBlockLayoutShPtr& layoutPtr) const {
  return getNumBytes() / numItems(layoutPtr);
}

size_t GLUniformBuffer::getNumAlignmentBytes() const {
  return GLShaderBlockLayout::getNumAlignmentBytes();
}

void GLUniformBuffer::bufferData(void* data, const size_t numBytes, const GLShaderBlockLayoutShPtr& layoutPtr) {
  auto fullNumBytesToUse = numBytes;
  std::unique_ptr<unsigned char[]> newData = nullptr;

  if (layoutPtr) {
    auto numBytesPerItem = layoutPtr->getNumBytesInBlock();
    RUNTIME_EX_ASSERT(numBytes % numBytesPerItem == 0,
                      "Cannot allocate " + std::to_string(numBytesPerItem) +
                          " bytes of data for uniform buffer. It does not align with the " +
                          std::to_string(numBytesPerItem) + " bytes required by the shader block layout.");

    if (data && numBytesPerItem % layoutPtr->getNumAlignmentBytes() != 0) {
      auto numItems = numBytes / layoutPtr->getNumBytesInBlock();

      auto numBytesToUse =
          calcNumBytesPerItem(layoutPtr->getNumBytesInBlock(), layoutPtr->getNumAlignmentBytes(), numItems);

      fullNumBytesToUse = numBytesToUse * numItems;
      newData.reset(new unsigned char[fullNumBytesToUse]);
      unsigned char* oldRawData = reinterpret_cast<unsigned char*>(data);
      unsigned char* newRawData = newData.get();
      std::fill(&newRawData[0], &newRawData[fullNumBytesToUse], 0);

      for (size_t i = 0; i < numItems; ++i) {
        std::copy(
            &oldRawData[i * numBytesPerItem], &oldRawData[(i + 1) * numBytesPerItem], &newRawData[i * numBytesToUse]);
      }

      data = newRawData;
    }
  }

  GLLayoutBuffer<GLShaderBlockLayout>::bufferData(data, fullNumBytesToUse, layoutPtr);
}

}  // namespace Resources
}  // namespace GL
}  // namespace Rendering

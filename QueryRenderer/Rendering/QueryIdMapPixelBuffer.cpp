#include "QueryIdMapPixelBuffer.h"
#include <Rendering/Renderer/GL/GLRenderer.h>
#include <Rendering/Renderer/GL/GLResourceManager.h>

namespace QueryRenderer {

template <>
void QueryIdMapPixelBuffer<unsigned int>::_init(::Rendering::GL::GLRenderer* renderer, size_t width, size_t height) {
  ::Rendering::GL::GLResourceManagerShPtr rsrcMgr = renderer->getResourceManager();

  _pbo = rsrcMgr->createPixelBuffer2d(
      width, height, GL_RED_INTEGER, GL_UNSIGNED_INT, ::Rendering::GL::Resources::BufferAccessType::READ_AND_WRITE);
}

template <>
void QueryIdMapPixelBuffer<unsigned int>::readIdBuffer(int width, int height, unsigned int* idBuffer) {
  CHECK(_pbo);

  _pbo->readPixels(width, height, GL_UNSIGNED_INT, idBuffer);
}

template <>
void QueryIdMapPixelBuffer<int>::_init(::Rendering::GL::GLRenderer* renderer, size_t width, size_t height) {
  ::Rendering::GL::GLResourceManagerShPtr rsrcMgr = renderer->getResourceManager();

  _pbo = rsrcMgr->createPixelBuffer2d(
      width, height, GL_RED_INTEGER, GL_INT, ::Rendering::GL::Resources::BufferAccessType::READ_AND_WRITE);
}

template <>
void QueryIdMapPixelBuffer<int>::readIdBuffer(int width, int height, int* idBuffer) {
  CHECK(_pbo);

  _pbo->readPixels(width, height, GL_INT, idBuffer);
}

}  // namespace QueryRenderer

#include "QueryIdMapPixelBuffer.h"
#include <Rendering/Renderer/GL/GLRenderer.h>
#include <Rendering/Renderer/GL/GLResourceManager.h>

namespace QueryRenderer {

QueryIdMapPixelBuffer::QueryIdMapPixelBuffer(::Rendering::GL::GLRenderer* renderer, size_t width, size_t height)
    : _pbo(nullptr) {
  _init(renderer, width, height);
}

void QueryIdMapPixelBuffer::_init(::Rendering::GL::GLRenderer* renderer, size_t width, size_t height) {
  ::Rendering::GL::GLResourceManagerShPtr rsrcMgr = renderer->getResourceManager();

  _pbo = rsrcMgr->createPixelBuffer2d(
      width, height, GL_RED_INTEGER, GL_UNSIGNED_INT, ::Rendering::GL::Resources::BufferAccessType::READ_AND_WRITE);
}

size_t QueryIdMapPixelBuffer::getWidth() const {
  if (_pbo) {
    return _pbo->getWidth();
  }
  return 0;
}

size_t QueryIdMapPixelBuffer::getHeight() const {
  if (_pbo) {
    return _pbo->getHeight();
  }
  return 0;
}

void QueryIdMapPixelBuffer::resize(size_t width, size_t height) {
  CHECK(_pbo);

  _pbo->resize(width, height);
}

void QueryIdMapPixelBuffer::readIdBuffer(int width, int height, unsigned int* idBuffer) {
  CHECK(_pbo);

  _pbo->readPixels(width, height, GL_UNSIGNED_INT, idBuffer);
}

}  // namespace QueryRenderer

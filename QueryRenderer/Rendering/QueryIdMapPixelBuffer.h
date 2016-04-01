#ifndef QUERYRENDERER_QUERYIDMAPPIXELBUFFER_H_
#define QUERYRENDERER_QUERYIDMAPPIXELBUFFER_H_

#include <Rendering/Renderer/GL/Types.h>
#include <Rendering/Renderer/GL/Resources/Types.h>

#include <memory>

namespace QueryRenderer {

class QueryIdMapPixelBuffer {
 public:
  QueryIdMapPixelBuffer(::Rendering::GL::GLRenderer* renderer, size_t width, size_t height);
  ~QueryIdMapPixelBuffer() {}

  size_t getWidth() const;
  size_t getHeight() const;

  void resize(size_t width, size_t height);
  void readIdBuffer(int width, int height, unsigned int* idBuffer);

  ::Rendering::GL::Resources::GLPixelBuffer2dShPtr getPixelBuffer2d() const { return _pbo; }

 private:
  void _init(::Rendering::GL::GLRenderer* renderer, size_t width, size_t height);

  ::Rendering::GL::Resources::GLPixelBuffer2dShPtr _pbo;
};

}  // namespace QueryRenderer

#endif  // QUERYRENDERER_QUERYIDMAPPIXELBUFFER_H_

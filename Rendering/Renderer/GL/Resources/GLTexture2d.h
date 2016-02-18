#ifndef RENDERING_GL_RESOURCES_GLTEXTURE2D_H_
#define RENDERING_GL_RESOURCES_GLTEXTURE2D_H_

#include "../Types.h"
#include "GLResource.h"
#include "Types.h"

namespace Rendering {
namespace GL {

namespace Resources {

class GLTexture2d : public GLResource {
 public:
  ~GLTexture2d();

  GLResourceType getResourceType() const final { return GLResourceType::TEXTURE_2D; }
  GLuint getId() const final { return _textureId; }

  int getWidth() { return _width; }
  int getHeight() { return _height; }
  GLenum getInternalFormat() { return _internalFormat; }
  GLenum getPixelFormat() { return _pixelFormat; }
  GLenum getPixelType() { return _pixelType; }
  int getNumSamples() { return _numSamples; }
  GLenum getTarget() { return _target; }

  void resize(int width, int height);

 private:
  GLTexture2d(const RendererWkPtr& rendererPtr,
              int width,
              int height,
              GLenum internalFormat,
              GLenum pixelFormat,
              GLenum pixelType,
              int numSamples = 1);

  void _initResource();
  void _cleanupResource() final;
  void _makeEmpty() final;

  int _width;
  int _height;
  GLenum _internalFormat;
  GLenum _pixelFormat;
  GLenum _pixelType;
  int _numSamples;

  GLuint _textureId;
  GLenum _target;

  friend class ::Rendering::GL::GLResourceManager;
};

}  // namespace Resources
}  // namespace GL
}  // namespace Rendering

#endif  // RENDERING_GL_RESOURCES_GLTEXTURE2D_H_

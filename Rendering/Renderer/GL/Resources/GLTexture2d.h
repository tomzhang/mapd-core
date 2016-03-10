#ifndef RENDERING_GL_RESOURCES_GLTEXTURE2D_H_
#define RENDERING_GL_RESOURCES_GLTEXTURE2D_H_

#include "../Types.h"
#include "GLResource.h"
#include "Types.h"

namespace Rendering {
namespace GL {

namespace Resources {

// TODO(croot): make this a Sampler resource object (see www.opengl.org/wiki/Sampler_Object)?
// However, would doing so mess with the "completeness"/"consistency" of the texture?
// A texture needs to be consistent (have texture sampling properties defined) for things
// like glXCopyImageSubDataNV(), for example.
struct GLTexture2dSampleProps {
  GLint minFilter;
  GLint magFilter;
  GLint wrapS;
  GLint wrapT;

  GLTexture2dSampleProps();
  GLTexture2dSampleProps(GLint minFilter, GLint magFilter, GLint wrapS, GLint wrapT);
  GLTexture2dSampleProps(const GLTexture2dSampleProps& props);
  GLTexture2dSampleProps(GLTexture2dSampleProps&& props);

  bool operator==(const GLTexture2dSampleProps& rhs) const;
  bool operator!=(const GLTexture2dSampleProps& rhs) const;

  GLTexture2dSampleProps& operator=(const GLTexture2dSampleProps& rhs);

  void apply(GLenum target = GL_TEXTURE_2D);
};

class GLTexture2d : public GLResource {
 public:
  ~GLTexture2d();

  GLResourceType getResourceType() const final { return GLResourceType::TEXTURE_2D; }
  GLuint getId() const final { return _textureId; }

  size_t getWidth() const { return _width; }
  size_t getHeight() const { return _height; }
  GLenum getInternalFormat() const { return _internalFormat; }
  GLenum getPixelFormat() const { return _pixelFormat; }
  GLenum getPixelType() const { return _pixelType; }
  size_t getNumSamples() const { return _numSamples; }
  GLenum getTarget() const final { return _target; }
  GLTexture2dSampleProps getSamplingProps() const { return _sampleProps; }

  void resize(size_t width, size_t height);

 private:
  explicit GLTexture2d(const RendererWkPtr& rendererPtr,
                       size_t width,
                       size_t height,
                       GLenum internalFormat,
                       GLenum pixelFormat,
                       GLenum pixelType,
                       const GLTexture2dSampleProps& sampleProps = GLTexture2dSampleProps(),
                       size_t numSamples = 1);

  void _initResource();
  void _cleanupResource() final;
  void _makeEmpty() final;

  size_t _width;
  size_t _height;
  GLenum _internalFormat;
  GLenum _pixelFormat;
  GLenum _pixelType;
  GLTexture2dSampleProps _sampleProps;
  size_t _numSamples;

  GLuint _textureId;
  GLenum _target;

  friend class ::Rendering::GL::GLResourceManager;
};

}  // namespace Resources
}  // namespace GL
}  // namespace Rendering

#endif  // RENDERING_GL_RESOURCES_GLTEXTURE2D_H_

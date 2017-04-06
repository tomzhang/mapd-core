#ifndef RENDERING_GL_RESOURCES_GLTEXTURE2DARRAY_H_
#define RENDERING_GL_RESOURCES_GLTEXTURE2DARRAY_H_

#include "../Types.h"
#include "GLResource.h"
#include "Types.h"
#include "GLTexture2d.h"
#include <vector>

namespace Rendering {
namespace GL {

namespace Resources {

class GLTexture2dArray : public GLResource {
 public:
  ~GLTexture2dArray();

  GLuint getId() const final { return _textureArrayId; }
  GLenum getTarget() const final { return _target; }

  size_t getWidth() const { return _width; }
  size_t getHeight() const { return _height; }
  size_t getDepth() const { return _depth; }
  GLenum getInternalFormat() const { return _internalFormat; }
  size_t getNumSamples() const { return _numSamples; }

  // NOTE: cannot resize a texture array after it's been initialized.
  // It is considered immutable after creation.
  // See: https://www.opengl.org/wiki/GLAPI/glTexStorage3D#Description
  // So a resize will result in an entirely new object internally
  // and therefore invalidate anything currently being stored
  void resize(size_t depth);
  void resize(size_t width, size_t height);
  void resize(size_t width, size_t height, size_t depth);

  void copyPixelsFromPixelBuffer(const GLPixelBuffer2dShPtr& pbo,
                                 int arrayIdx = -1,
                                 int startx = 0,
                                 int starty = 0,
                                 int width = -1,
                                 int height = -1);

  /**
   * Update the pixel contents of the texture array
   * to the internally referenced textures.
   * This would be called after any textures this
   * texture array references has been updated.
   * This is necessary to keep the two in sync.
   */
  void updateFromTextures();

  void uploadPixelsToBufferIndex(const size_t width,
                                 const size_t height,
                                 const GLenum pixelFormat,
                                 const GLenum pixelType,
                                 const GLvoid* pixelData,
                                 const int arrayIdx = -1,
                                 const size_t startx = 0,
                                 const size_t starty = 0);

  void getPixels(const GLenum pixelFormat, const GLenum pixelType, GLvoid* pixelData);

 private:
  GLTexture2dArray(const RendererWkPtr& rendererPtr,
                   size_t width,
                   size_t height,
                   size_t depth,
                   GLenum internalFormat,
                   size_t numSamples = 1,
                   const GLTexture2dSampleProps& sampleProps = GLTexture2dSampleProps());

  GLTexture2dArray(const RendererWkPtr& rendererPtr, const std::vector<GLTexture2dShPtr>& initTextures);

  void _initResource();
  void _cleanupResource() final;
  void _validateTexture(const GLTexture2dShPtr& texture);
  void _makeEmpty() final;
  void _rebuild(size_t width, size_t height, size_t depth);

  size_t _width;
  size_t _height;
  size_t _depth;
  GLenum _internalFormat;
  GLTexture2dSampleProps _sampleProps;
  size_t _numSamples;

  GLuint _textureArrayId;
  GLenum _target;

  std::vector<GLTexture2dShPtr> _textureReferences;

  friend class ::Rendering::GL::GLResourceManager;
};

}  // namespace Resources
}  // namespace GL
}  // namespace Rendering

#endif  // RENDERING_GL_RESOURCES_GLTEXTURE2DARRAY_H_

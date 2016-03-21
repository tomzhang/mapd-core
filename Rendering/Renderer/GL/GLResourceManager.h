#ifndef RENDERING_GL_GLRESOURCEMANAGER_H_
#define RENDERING_GL_GLRESOURCEMANAGER_H_

#include "Types.h"
#include "Resources/Types.h"
#include "Resources/Enums.h"
#include "Resources/GLShader.h"
#include "Resources/GLRenderbuffer.h"
#include "Resources/GLTexture2d.h"
#include "Resources/GLTexture2dArray.h"
#include "Resources/GLFramebuffer.h"
#include "Resources/GLVertexBuffer.h"
#include "Resources/GLPixelBuffer2d.h"
#include <string>
#include <vector>

namespace Rendering {
namespace GL {

class GLResourceManager {
 public:
  ~GLResourceManager();

  Resources::GLShaderShPtr createShader(const std::string& vertexShaderSrc, const std::string& fragmentShaderSrc);
  Resources::GLRenderbufferShPtr createRenderbuffer(int width, int height, GLenum internalFormat, int numSamples = 1);
  Resources::GLTexture2dShPtr createTexture2d(
      size_t width,
      size_t height,
      GLenum internalFormat,
      GLenum pixelFormat,
      GLenum pixelType,
      const Resources::GLTexture2dSampleProps& sampleProps = Resources::GLTexture2dSampleProps(),
      size_t numSamples = 1);

  Resources::GLTexture2dArrayShPtr createTexture2dArray(
      size_t width,
      size_t height,
      size_t depth,
      GLenum internalFormat,
      const Resources::GLTexture2dSampleProps& sampleProps = Resources::GLTexture2dSampleProps(),
      size_t numSamples = 1);

  Resources::GLTexture2dArrayShPtr createTexture2dArray(const std::vector<Resources::GLTexture2dShPtr>& initTextures);

  Resources::GLFramebufferShPtr createFramebuffer(const Resources::GLFramebufferAttachmentMap& attachments);

  Resources::GLVertexBufferShPtr createVertexBuffer(
      Resources::BufferAccessType accessType = Resources::BufferAccessType::READ_AND_WRITE,
      Resources::BufferAccessFreq accessFreq = Resources::BufferAccessFreq::STATIC);
  Resources::GLVertexBufferShPtr createVertexBuffer(
      size_t numBytes,
      Resources::BufferAccessType accessType = Resources::BufferAccessType::READ_AND_WRITE,
      Resources::BufferAccessFreq accessFreq = Resources::BufferAccessFreq::STATIC);
  Resources::GLVertexBufferShPtr createVertexBuffer(
      const Resources::GLBufferLayoutShPtr& layoutPtr,
      Resources::BufferAccessType accessType = Resources::BufferAccessType::READ_AND_WRITE,
      Resources::BufferAccessFreq accessFreq = Resources::BufferAccessFreq::STATIC);

  Resources::GLPixelBuffer2dShPtr createPixelBuffer2d(
      size_t width,
      size_t height,
      // GLenum internalFormat,
      GLenum pixelFormat,
      GLenum pixelType,
      Resources::BufferAccessType accessType = Resources::BufferAccessType::READ,
      Resources::BufferAccessFreq accessFreq = Resources::BufferAccessFreq::DYNAMIC);

  template <typename T>
  Resources::GLVertexBufferShPtr createVertexBuffer(
      const std::vector<T>& data,
      const Resources::GLBufferLayoutShPtr& layoutPtr,
      Resources::BufferAccessType accessType = Resources::BufferAccessType::READ_AND_WRITE,
      Resources::BufferAccessFreq accessFreq = Resources::BufferAccessFreq::STATIC) {
    CHECK(!_prntRenderer.expired());

    // TODO(croot): make thread safe?
    Resources::GLVertexBufferShPtr rtn(
        new Resources::GLVertexBuffer(_prntRenderer, data, layoutPtr, accessType, accessFreq));
    _glResources.push_back(Resources::GLResourceWkPtr(rtn));

    return rtn;
  }

  Resources::GLVertexArrayShPtr createVertexArray();
  Resources::GLVertexArrayShPtr createVertexArray(const Resources::VboAttrToShaderAttrMap& vboAttrToShaderAttrMap);

  Resources::GLResourceShPtr getResourcePtr(Resources::GLResource* rsrc);

 protected:
  explicit GLResourceManager(const RendererShPtr& prnt);

  // void createFrameBufferObject();

 private:
  RendererWkPtr _prntRenderer;

  /**
   * Cleans up all resources. This will be called during destruction of the renderer
   * or the resource manager
   */
  void _cleanupResources();

  // TODO(croot): this will hold onto the ptrs for all time right now. It would
  // probably be useful to purge empty ptrs from time-to-time. And it would
  // probably be best to keep this hidden from the user. Perhaps the renderer
  // can do it when idle? Or possibly every nth render call? Need to think
  // about this, but for the time being it shouldn't be a big deal because a
  // weak_ptr takes up very little memory.

  // TODO(croot): improve the data structure used here. May want to separate
  // byte resource types for easier lookups. May want to add a name to resources
  // too for an additional level of categorization.
  std::vector<Resources::GLResourceWkPtr> _glResources;
  void _addGLResource(Resources::GLResourceShPtr glResource);

  friend class ::Rendering::WindowManager;
  friend class GLRenderer;
};

}  // namespace GL
}  // namespace Rendering

#endif  // RENDERING_GL_GLRESOURCEMANAGER_H_

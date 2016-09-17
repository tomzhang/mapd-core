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
#include "Resources/GLVertexArray.h"
#include "Resources/GLPixelBuffer2d.h"
#include "Resources/GLUniformBuffer.h"
#include "Resources/GLIndexBuffer.h"
#include "Resources/GLIndirectDrawBuffer.h"
#include <string>
#include <vector>

namespace Rendering {
namespace GL {

class GLResourceManager {
 public:
  ~GLResourceManager();

  Resources::GLShaderShPtr createShader(const std::string& vertexShaderSrc,
                                        const std::string& fragmentShaderSrc,
                                        const std::string& geometryShaderSrc = "");
  Resources::GLRenderbufferShPtr createRenderbuffer(int width, int height, GLenum internalFormat, int numSamples = 1);
  Resources::GLTexture2dShPtr createTexture2d(
      size_t width,
      size_t height,
      GLenum internalFormat,
      GLenum pixelFormat,
      GLenum pixelType,
      size_t numSamples = 1,
      const Resources::GLTexture2dSampleProps& sampleProps = Resources::GLTexture2dSampleProps(),
      const GLvoid* pixelData = nullptr);

  Resources::GLTexture2dArrayShPtr createTexture2dArray(
      size_t width,
      size_t height,
      size_t depth,
      GLenum internalFormat,
      size_t numSamples = 1,
      const Resources::GLTexture2dSampleProps& sampleProps = Resources::GLTexture2dSampleProps());

  Resources::GLTexture2dArrayShPtr createTexture2dArray(const std::vector<Resources::GLTexture2dShPtr>& initTextures);

  Resources::GLFramebufferShPtr createFramebuffer(const Resources::GLFramebufferAttachmentMap& attachments);

  Resources::GLIndexBufferShPtr createIndexBuffer(
      Resources::GLIndexBuffer::IndexType indexType = Resources::GLIndexBuffer::IndexType::UNSIGNED_INT,
      Resources::BufferAccessType accessType = Resources::BufferAccessType::READ_AND_WRITE,
      Resources::BufferAccessFreq accessFreq = Resources::BufferAccessFreq::STATIC);

  Resources::GLIndexBufferShPtr createIndexBuffer(
      size_t numBytes,
      Resources::GLIndexBuffer::IndexType indexType = Resources::GLIndexBuffer::IndexType::UNSIGNED_INT,
      Resources::BufferAccessType accessType = Resources::BufferAccessType::READ_AND_WRITE,
      Resources::BufferAccessFreq accessFreq = Resources::BufferAccessFreq::STATIC);

  Resources::GLIndexBufferShPtr createIndexBuffer(
      const std::vector<unsigned char>& items,
      Resources::BufferAccessType accessType = Resources::BufferAccessType::READ_AND_WRITE,
      Resources::BufferAccessFreq accessFreq = Resources::BufferAccessFreq::STATIC);

  Resources::GLIndexBufferShPtr createIndexBuffer(
      const std::vector<unsigned short>& items,
      Resources::BufferAccessType accessType = Resources::BufferAccessType::READ_AND_WRITE,
      Resources::BufferAccessFreq accessFreq = Resources::BufferAccessFreq::STATIC);

  Resources::GLIndexBufferShPtr createIndexBuffer(
      const std::vector<unsigned int>& items,
      Resources::BufferAccessType accessType = Resources::BufferAccessType::READ_AND_WRITE,
      Resources::BufferAccessFreq accessFreq = Resources::BufferAccessFreq::STATIC);

  Resources::GLIndirectDrawVertexBufferShPtr createIndirectDrawVertexBuffer(
      Resources::BufferAccessType accessType = Resources::BufferAccessType::READ_AND_WRITE,
      Resources::BufferAccessFreq accessFreq = Resources::BufferAccessFreq::STATIC);

  Resources::GLIndirectDrawVertexBufferShPtr createIndirectDrawVertexBuffer(
      size_t numBytes,
      Resources::BufferAccessType accessType = Resources::BufferAccessType::READ_AND_WRITE,
      Resources::BufferAccessFreq accessFreq = Resources::BufferAccessFreq::STATIC);

  Resources::GLIndirectDrawVertexBufferShPtr createIndirectDrawVertexBuffer(
      const std::vector<Resources::IndirectDrawVertexData>& items,
      Resources::BufferAccessType accessType = Resources::BufferAccessType::READ_AND_WRITE,
      Resources::BufferAccessFreq accessFreq = Resources::BufferAccessFreq::STATIC);

  Resources::GLIndirectDrawIndexBufferShPtr createIndirectDrawIndexBuffer(
      Resources::BufferAccessType accessType = Resources::BufferAccessType::READ_AND_WRITE,
      Resources::BufferAccessFreq accessFreq = Resources::BufferAccessFreq::STATIC);

  Resources::GLIndirectDrawIndexBufferShPtr createIndirectDrawIndexBuffer(
      size_t numBytes,
      Resources::BufferAccessType accessType = Resources::BufferAccessType::READ_AND_WRITE,
      Resources::BufferAccessFreq accessFreq = Resources::BufferAccessFreq::STATIC);

  Resources::GLIndirectDrawIndexBufferShPtr createIndirectDrawIndexBuffer(
      const std::vector<Resources::IndirectDrawIndexData>& items,
      Resources::BufferAccessType accessType = Resources::BufferAccessType::READ_AND_WRITE,
      Resources::BufferAccessFreq accessFreq = Resources::BufferAccessFreq::STATIC);

  Resources::GLVertexBufferShPtr createVertexBuffer(
      Resources::BufferAccessType accessType = Resources::BufferAccessType::READ_AND_WRITE,
      Resources::BufferAccessFreq accessFreq = Resources::BufferAccessFreq::STATIC);
  Resources::GLVertexBufferShPtr createVertexBuffer(
      size_t numBytes,
      Resources::BufferAccessType accessType = Resources::BufferAccessType::READ_AND_WRITE,
      Resources::BufferAccessFreq accessFreq = Resources::BufferAccessFreq::STATIC);

  template <typename T>
  Resources::GLVertexBufferShPtr createVertexBuffer(
      const std::vector<T>& data,
      const Resources::GLBufferLayoutShPtr& layoutPtr,
      Resources::BufferAccessType accessType = Resources::BufferAccessType::READ_AND_WRITE,
      Resources::BufferAccessFreq accessFreq = Resources::BufferAccessFreq::STATIC) {
    CHECK(!_prntRenderer.expired());

    // TODO(croot): make thread safe?
    Resources::GLVertexBufferShPtr rtn(new Resources::GLVertexBuffer(_prntRenderer, accessType, accessFreq));
    _addGLResource(rtn);

    // NOTE: the bufferData call must be done after the resource has been appropriately added

    // TODO(croot): what if the below bufferData() call fails?
    // Should we remove the resource internally?
    rtn->bufferData((void*)&data[0], data.size() * sizeof(T), layoutPtr);

    return rtn;
  }

  Resources::GLPixelBuffer2dShPtr createPixelBuffer2d(
      size_t width,
      size_t height,
      // GLenum internalFormat,
      GLenum pixelFormat,
      GLenum pixelType,
      Resources::BufferAccessType accessType = Resources::BufferAccessType::READ,
      Resources::BufferAccessFreq accessFreq = Resources::BufferAccessFreq::DYNAMIC);

  Resources::GLPixelBuffer2dShPtr createPixelBuffer2d(
      size_t width,
      size_t height,
      // GLenum internalFormat,
      GLenum pixelFormat,
      GLenum pixelType,
      GLvoid* data,
      Resources::BufferAccessType accessType = Resources::BufferAccessType::READ,
      Resources::BufferAccessFreq accessFreq = Resources::BufferAccessFreq::DYNAMIC);

  Resources::GLUniformBufferShPtr createUniformBuffer(
      Resources::BufferAccessType accessType = Resources::BufferAccessType::WRITE,
      Resources::BufferAccessFreq accessFreq = Resources::BufferAccessFreq::DYNAMIC);

  Resources::GLUniformBufferShPtr createUniformBuffer(
      size_t numBytes,
      Resources::BufferAccessType accessType = Resources::BufferAccessType::WRITE,
      Resources::BufferAccessFreq accessFreq = Resources::BufferAccessFreq::DYNAMIC);

  template <typename T>
  Resources::GLUniformBufferShPtr createUniformBuffer(
      const std::vector<T>& data,
      const Resources::GLShaderBlockLayoutShPtr& shaderBlockLayoutPtr,
      Resources::BufferAccessType accessType = Resources::BufferAccessType::WRITE,
      Resources::BufferAccessFreq accessFreq = Resources::BufferAccessFreq::DYNAMIC) {
    CHECK(!_prntRenderer.expired());

    // TODO(croot): make thread safe?
    Resources::GLUniformBufferShPtr rtn(
        new Resources::GLUniformBuffer(_prntRenderer, data, shaderBlockLayoutPtr, accessType, accessFreq));
    _addGLResource(rtn);

    // NOTE: the bufferData call must be done after the resource has been appropriately added

    // TODO(croot): what if the below bufferData() call fails?
    // Should we remove the resource internally?
    rtn->bufferData((void*)&data[0], data.size() * sizeof(T), shaderBlockLayoutPtr);

    return rtn;
  }

  Resources::GLVertexArrayShPtr createVertexArray();
  Resources::GLVertexArrayShPtr createVertexArray(const Resources::VboAttrToShaderAttrMap& vboAttrToShaderAttrMap,
                                                  const Resources::GLIndexBufferShPtr& iboPtr = nullptr);
  Resources::GLVertexArrayShPtr createVertexArray(
      const Resources::VboLayoutAttrToShaderAttrMap& vboLayoutAttrToShaderAttrMap,
      const Resources::GLIndexBufferShPtr& iboPtr = nullptr);

  const Resources::GLResourceShPtr getSharedResourcePtr(const Resources::GLResource* rsrc) const;
  Resources::GLResourceShPtr getSharedResourcePtr(const Resources::GLResource* rsrc);

  template <typename GLResourceType>
  std::shared_ptr<GLResourceType> getSharedResourceTypePtr(const GLResourceType* rsrc) {
    return std::dynamic_pointer_cast<GLResourceType>(getSharedResourcePtr(rsrc));
  }

 protected:
  explicit GLResourceManager(const RendererShPtr& prnt);

  // void createFrameBufferObject();

 private:
  RendererWkPtr _prntRenderer;
  ResourceId _currRsrcId;

  /**
   * Cleans up all resources. This will be called during destruction of the renderer
   * or the resource manager
   */
  void _cleanupResources();

  // TODO(croot): Perhaps improve when weak ptrs are purged?
  // Right now a purge is performed every time a new resource is
  // created. Perhaps the renderer can do it when idle?
  // Or possibly every nth render call?

  // TODO(croot): improve the data structure used here. May want to separate
  // byte resource types for easier lookups. May want to add a name to resources
  // too for an additional level of categorization.
  // Ideally this data structure should be good about doing a lookup by
  // GLResource* too.
  std::vector<Resources::GLResourceWkPtr> _glResources;
  void _addGLResource(Resources::GLResourceShPtr glResource);

  friend class ::Rendering::WindowManager;
  friend class GLRenderer;
};

}  // namespace GL
}  // namespace Rendering

#endif  // RENDERING_GL_GLRESOURCEMANAGER_H_

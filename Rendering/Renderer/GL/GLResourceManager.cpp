#include "GLResourceManager.h"
#include "GLRenderer.h"

namespace Rendering {
namespace GL {

using Resources::GLResource;
using Resources::GLResourceShPtr;
using Resources::GLShaderShPtr;
using Resources::GLShader;
using Resources::GLResourceWkPtr;
using Resources::GLRenderbuffer;
using Resources::GLRenderbufferShPtr;
using Resources::GLTexture2dSampleProps;
using Resources::GLTexture2d;
using Resources::GLTexture2dShPtr;
using Resources::GLTexture2dArray;
using Resources::GLTexture2dArrayShPtr;
using Resources::GLFramebuffer;
using Resources::GLFramebufferShPtr;
using Resources::BufferAccessType;
using Resources::BufferAccessFreq;
using Resources::GLIndexBuffer;
using Resources::GLIndexBufferShPtr;
using Resources::IndirectDrawVertexData;
using Resources::GLIndirectDrawVertexBuffer;
using Resources::GLIndirectDrawVertexBufferShPtr;
using Resources::IndirectDrawIndexData;
using Resources::GLIndirectDrawIndexBuffer;
using Resources::GLIndirectDrawIndexBufferShPtr;
using Resources::GLVertexBuffer;
using Resources::GLVertexBufferShPtr;
using Resources::GLPixelBuffer2d;
using Resources::GLPixelBuffer2dShPtr;
using Resources::GLUniformBuffer;
using Resources::GLUniformBufferShPtr;
using Resources::GLVertexArray;
using Resources::GLVertexArrayShPtr;
using Resources::VboAttrToShaderAttrMap;

GLResourceManager::GLResourceManager(const RendererShPtr& prnt) : _prntRenderer(), _currRsrcId(0) {
  GLRenderer* renderer = dynamic_cast<GLRenderer*>(prnt.get());
  CHECK(renderer != nullptr);
  _prntRenderer = prnt;
}

GLResourceManager::~GLResourceManager() {
  _cleanupResources();
}

void GLResourceManager::_cleanupResources() {
  // TODO(croot): make thread safe?
  for (auto& resource : _glResources) {
    if (!resource.expired()) {
      resource.lock()->_cleanupResource();

      // we're deleting the resource manager before the resources
      // themselves have been cleaned up.
      // We to cleanup the resource and invalidate those objects from further use
    }
  }
}

GLShaderShPtr GLResourceManager::createShader(const std::string& vertexShaderSrc,
                                              const std::string& fragmentShaderSrc,
                                              const std::string& geometryShaderSrc) {
  CHECK(!_prntRenderer.expired());

  // TODO(croot): make thread safe?
  GLShaderShPtr rtn(new GLShader(_prntRenderer));
  _addGLResource(rtn);
  rtn->_initResource(vertexShaderSrc, fragmentShaderSrc, geometryShaderSrc);

  return rtn;
}

GLRenderbufferShPtr GLResourceManager::createRenderbuffer(int width,
                                                          int height,
                                                          GLenum internalFormat,
                                                          int numSamples) {
  CHECK(!_prntRenderer.expired());

  // TODO(croot): make thread safe?
  GLRenderbufferShPtr rtn(new GLRenderbuffer(_prntRenderer, width, height, internalFormat, numSamples));
  _addGLResource(rtn);

  return rtn;
}

GLTexture2dShPtr GLResourceManager::createTexture2d(size_t width,
                                                    size_t height,
                                                    GLenum internalFormat,
                                                    GLenum pixelFormat,
                                                    GLenum pixelType,
                                                    size_t numSamples,
                                                    const GLTexture2dSampleProps& sampleProps,
                                                    const GLvoid* pixelData) {
  CHECK(!_prntRenderer.expired());

  // TODO(croot): make thread safe?
  GLTexture2dShPtr rtn(new GLTexture2d(
      _prntRenderer, width, height, internalFormat, pixelFormat, pixelType, numSamples, sampleProps, pixelData));
  _addGLResource(rtn);

  return rtn;
}

GLTexture2dArrayShPtr GLResourceManager::createTexture2dArray(size_t width,
                                                              size_t height,
                                                              size_t depth,
                                                              GLenum internalFormat,
                                                              size_t numSamples,
                                                              const GLTexture2dSampleProps& sampleProps) {
  CHECK(!_prntRenderer.expired());

  // TODO(croot): make thread safe?
  GLTexture2dArrayShPtr rtn(
      new GLTexture2dArray(_prntRenderer, width, height, depth, internalFormat, numSamples, sampleProps));

  return rtn;
}

GLTexture2dArrayShPtr GLResourceManager::createTexture2dArray(const std::vector<GLTexture2dShPtr>& initTextures) {
  CHECK(!_prntRenderer.expired());

  // TODO(croot): make thread safe?
  GLTexture2dArrayShPtr rtn(new GLTexture2dArray(_prntRenderer, initTextures));

  return rtn;
}

GLFramebufferShPtr GLResourceManager::createFramebuffer(const Resources::GLFramebufferAttachmentMap& attachments) {
  CHECK(!_prntRenderer.expired());

  // TODO(croot): make thread safe?
  GLFramebufferShPtr rtn(new GLFramebuffer(_prntRenderer, attachments));
  _addGLResource(rtn);

  return rtn;
}

GLIndexBufferShPtr GLResourceManager::createIndexBuffer(GLIndexBuffer::IndexType indexType,
                                                        BufferAccessType accessType,
                                                        BufferAccessFreq accessFreq) {
  CHECK(!_prntRenderer.expired());

  // TODO(croot): make thread safe?
  GLIndexBufferShPtr rtn(new GLIndexBuffer(_prntRenderer, indexType, accessType, accessFreq));
  _addGLResource(rtn);

  return rtn;
}

GLIndexBufferShPtr GLResourceManager::createIndexBuffer(size_t numBytes,
                                                        GLIndexBuffer::IndexType indexType,
                                                        BufferAccessType accessType,
                                                        BufferAccessFreq accessFreq) {
  CHECK(!_prntRenderer.expired());

  // TODO(croot): make thread safe?
  GLIndexBufferShPtr rtn(new GLIndexBuffer(_prntRenderer, numBytes, indexType, accessType, accessFreq));
  _addGLResource(rtn);

  return rtn;
}

GLIndexBufferShPtr GLResourceManager::createIndexBuffer(const std::vector<unsigned char>& items,
                                                        BufferAccessType accessType,
                                                        BufferAccessFreq accessFreq) {
  CHECK(!_prntRenderer.expired());

  // TODO(croot): make thread safe?
  GLIndexBufferShPtr rtn(new GLIndexBuffer(_prntRenderer, items, accessType, accessFreq));
  _addGLResource(rtn);

  return rtn;
}

GLIndexBufferShPtr GLResourceManager::createIndexBuffer(const std::vector<unsigned short>& items,
                                                        BufferAccessType accessType,
                                                        BufferAccessFreq accessFreq) {
  CHECK(!_prntRenderer.expired());

  // TODO(croot): make thread safe?
  GLIndexBufferShPtr rtn(new GLIndexBuffer(_prntRenderer, items, accessType, accessFreq));
  _addGLResource(rtn);

  return rtn;
}

GLIndexBufferShPtr GLResourceManager::createIndexBuffer(const std::vector<unsigned int>& items,
                                                        BufferAccessType accessType,
                                                        BufferAccessFreq accessFreq) {
  CHECK(!_prntRenderer.expired());

  // TODO(croot): make thread safe?
  GLIndexBufferShPtr rtn(new GLIndexBuffer(_prntRenderer, items, accessType, accessFreq));
  _addGLResource(rtn);

  return rtn;
}

GLIndirectDrawVertexBufferShPtr GLResourceManager::createIndirectDrawVertexBuffer(BufferAccessType accessType,
                                                                                  BufferAccessFreq accessFreq) {
  CHECK(!_prntRenderer.expired());

  // TODO(croot): make thread safe?
  GLIndirectDrawVertexBufferShPtr rtn(new GLIndirectDrawVertexBuffer(_prntRenderer, accessType, accessFreq));
  _addGLResource(rtn);

  return rtn;
}

GLIndirectDrawVertexBufferShPtr GLResourceManager::createIndirectDrawVertexBuffer(size_t numBytes,
                                                                                  BufferAccessType accessType,
                                                                                  BufferAccessFreq accessFreq) {
  CHECK(!_prntRenderer.expired());

  // TODO(croot): make thread safe?
  GLIndirectDrawVertexBufferShPtr rtn(new GLIndirectDrawVertexBuffer(_prntRenderer, numBytes, accessType, accessFreq));
  _addGLResource(rtn);

  return rtn;
}

GLIndirectDrawVertexBufferShPtr GLResourceManager::createIndirectDrawVertexBuffer(
    const std::vector<IndirectDrawVertexData>& items,
    BufferAccessType accessType,
    BufferAccessFreq accessFreq) {
  CHECK(!_prntRenderer.expired());

  // TODO(croot): make thread safe?
  GLIndirectDrawVertexBufferShPtr rtn(new GLIndirectDrawVertexBuffer(_prntRenderer, items, accessType, accessFreq));
  _addGLResource(rtn);

  return rtn;
}

GLIndirectDrawIndexBufferShPtr GLResourceManager::createIndirectDrawIndexBuffer(BufferAccessType accessType,
                                                                                BufferAccessFreq accessFreq) {
  CHECK(!_prntRenderer.expired());

  // TODO(croot): make thread safe?
  GLIndirectDrawIndexBufferShPtr rtn(new GLIndirectDrawIndexBuffer(_prntRenderer, accessType, accessFreq));
  _addGLResource(rtn);

  return rtn;
}

GLIndirectDrawIndexBufferShPtr GLResourceManager::createIndirectDrawIndexBuffer(size_t numBytes,
                                                                                BufferAccessType accessType,
                                                                                BufferAccessFreq accessFreq) {
  CHECK(!_prntRenderer.expired());

  // TODO(croot): make thread safe?
  GLIndirectDrawIndexBufferShPtr rtn(new GLIndirectDrawIndexBuffer(_prntRenderer, numBytes, accessType, accessFreq));
  _addGLResource(rtn);

  return rtn;
}

GLIndirectDrawIndexBufferShPtr GLResourceManager::createIndirectDrawIndexBuffer(
    const std::vector<IndirectDrawIndexData>& items,
    BufferAccessType accessType,
    BufferAccessFreq accessFreq) {
  CHECK(!_prntRenderer.expired());

  // TODO(croot): make thread safe?
  GLIndirectDrawIndexBufferShPtr rtn(new GLIndirectDrawIndexBuffer(_prntRenderer, items, accessType, accessFreq));
  _addGLResource(rtn);

  return rtn;
}

GLVertexBufferShPtr GLResourceManager::createVertexBuffer(BufferAccessType accessType, BufferAccessFreq accessFreq) {
  CHECK(!_prntRenderer.expired());

  // TODO(croot): make thread safe?
  GLVertexBufferShPtr rtn(new GLVertexBuffer(_prntRenderer, accessType, accessFreq));
  _addGLResource(rtn);

  return rtn;
}

GLVertexBufferShPtr GLResourceManager::createVertexBuffer(size_t numBytes,
                                                          BufferAccessType accessType,
                                                          BufferAccessFreq accessFreq) {
  CHECK(!_prntRenderer.expired());

  // TODO(croot): make thread safe?
  GLVertexBufferShPtr rtn(new GLVertexBuffer(_prntRenderer, numBytes, accessType, accessFreq));
  _addGLResource(rtn);

  return rtn;
}

GLPixelBuffer2dShPtr GLResourceManager::createPixelBuffer2d(size_t width,
                                                            size_t height,
                                                            // GLenum internalFormat,
                                                            GLenum pixelFormat,
                                                            GLenum pixelType,
                                                            Resources::BufferAccessType accessType,
                                                            Resources::BufferAccessFreq accessFreq) {
  CHECK(!_prntRenderer.expired());

  // TODO(croot): make thread safe?
  GLPixelBuffer2dShPtr rtn(new GLPixelBuffer2d(_prntRenderer,
                                               width,
                                               height,
                                               // internalFormat,
                                               pixelFormat,
                                               pixelType,
                                               accessType,
                                               accessFreq));
  _addGLResource(rtn);

  return rtn;
}

GLPixelBuffer2dShPtr GLResourceManager::createPixelBuffer2d(size_t width,
                                                            size_t height,
                                                            // GLenum internalFormat,
                                                            GLenum pixelFormat,
                                                            GLenum pixelType,
                                                            GLvoid* data,
                                                            Resources::BufferAccessType accessType,
                                                            Resources::BufferAccessFreq accessFreq) {
  CHECK(!_prntRenderer.expired());

  // TODO(croot): make thread safe?
  GLPixelBuffer2dShPtr rtn(new GLPixelBuffer2d(_prntRenderer,
                                               width,
                                               height,
                                               // internalFormat,
                                               pixelFormat,
                                               pixelType,
                                               data,
                                               accessType,
                                               accessFreq));
  _addGLResource(rtn);

  return rtn;
}

GLUniformBufferShPtr GLResourceManager::createUniformBuffer(Resources::BufferAccessType accessType,
                                                            Resources::BufferAccessFreq accessFreq) {
  CHECK(!_prntRenderer.expired());

  // TODO(croot): make thread safe?
  GLUniformBufferShPtr rtn(new GLUniformBuffer(_prntRenderer, accessType, accessFreq));
  _addGLResource(rtn);

  return rtn;
}

GLUniformBufferShPtr GLResourceManager::createUniformBuffer(size_t numBytes,
                                                            Resources::BufferAccessType accessType,
                                                            Resources::BufferAccessFreq accessFreq) {
  CHECK(!_prntRenderer.expired());

  // TODO(croot): make thread safe?
  GLUniformBufferShPtr rtn(new GLUniformBuffer(_prntRenderer, numBytes, accessType, accessFreq));
  _addGLResource(rtn);

  return rtn;
}

// GLUniformBufferShPtr GLResourceManager::createUniformBuffer(
//     const Resources::GLShaderBlockLayoutShPtr& shaderBlockLayoutPtr,
//     size_t numItems,
//     Resources::BufferAccessType accessType,
//     Resources::BufferAccessFreq accessFreq) {
//   CHECK(!_prntRenderer.expired());

//   // TODO(croot): make thread safe?
//   GLUniformBufferShPtr rtn(new GLUniformBuffer(_prntRenderer, shaderBlockLayoutPtr, numItems, accessType,
//   accessFreq));
//   _addGLResource(rtn);

//   return rtn;
// }

GLVertexArrayShPtr GLResourceManager::createVertexArray() {
  CHECK(!_prntRenderer.expired());

  // TODO(croot): make thread safe?
  GLVertexArrayShPtr rtn(new GLVertexArray(_prntRenderer));
  _addGLResource(rtn);

  return rtn;
}

GLVertexArrayShPtr GLResourceManager::createVertexArray(const VboAttrToShaderAttrMap& vboAttrToShaderAttrMap,
                                                        const GLIndexBufferShPtr& iboPtr) {
  CHECK(!_prntRenderer.expired());

  // TODO(croot): make thread safe?
  GLVertexArrayShPtr rtn(new GLVertexArray(_prntRenderer, vboAttrToShaderAttrMap, iboPtr));
  _addGLResource(rtn);

  for (auto& item : vboAttrToShaderAttrMap) {
    item.first->_addVertexArray(rtn);
  }

  return rtn;
}

GLVertexArrayShPtr GLResourceManager::createVertexArray(
    const Resources::VboLayoutAttrToShaderAttrMap& vboLayoutAttrToShaderAttrMap,
    const GLIndexBufferShPtr& iboPtr) {
  CHECK(!_prntRenderer.expired());

  // TODO(croot): make thread safe?
  GLVertexArrayShPtr rtn(new GLVertexArray(_prntRenderer, vboLayoutAttrToShaderAttrMap, iboPtr));
  _addGLResource(rtn);

  for (auto& item : vboLayoutAttrToShaderAttrMap) {
    item.first.first->_addVertexArray(rtn);
  }

  return rtn;
}

const GLResourceShPtr GLResourceManager::getSharedResourcePtr(const GLResource* rsrc) const {
  GLResourceShPtr rtn;
  for (size_t i = 0; i < _glResources.size(); ++i) {
    rtn = _glResources[i].lock();
    if (rtn && rtn.get() == rsrc) {
      break;
    }
    rtn = nullptr;
  }
  return rtn;
}

GLResourceShPtr GLResourceManager::getSharedResourcePtr(const GLResource* rsrc) {
  auto rtn = static_cast<const GLResourceManager&>(*this).getSharedResourcePtr(static_cast<const GLResource*>(rsrc));
  return const_cast<GLResourceShPtr&>(rtn);
}

void GLResourceManager::_addGLResource(Resources::GLResourceShPtr glResource) {
  // do a purge of any delete resources first
  _glResources.erase(
      std::remove_if(_glResources.begin(), _glResources.end(), [](auto& glResource) { return glResource.expired(); }),
      _glResources.end());

  RUNTIME_EX_ASSERT(_currRsrcId < std::numeric_limits<decltype(_currRsrcId)>::max(),
                    "The maximum resource limit reached.");

  glResource->_rsrcId = _currRsrcId++;
  _glResources.emplace_back(glResource);
}

}  // namespace GL
}  // namespace Rendering

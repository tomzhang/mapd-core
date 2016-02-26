#include "GLResourceManager.h"
#include "GLRenderer.h"

namespace Rendering {
namespace GL {

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
using Resources::GLVertexBuffer;
using Resources::GLVertexBufferShPtr;
using Resources::GLVertexArray;
using Resources::GLVertexArrayShPtr;
using Resources::VboAttrToShaderAttrMap;

GLResourceManager::GLResourceManager(const RendererShPtr& prnt) : _prntRenderer() {
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
                                              const std::string& fragmentShaderSrc) {
  CHECK(!_prntRenderer.expired());

  // TODO(croot): make thread safe?
  GLShaderShPtr rtn(new GLShader(_prntRenderer, vertexShaderSrc, fragmentShaderSrc));
  _glResources.push_back(GLResourceWkPtr(rtn));

  return rtn;
}

GLRenderbufferShPtr GLResourceManager::createRenderbuffer(int width,
                                                          int height,
                                                          GLenum internalFormat,
                                                          int numSamples) {
  CHECK(!_prntRenderer.expired());

  // TODO(croot): make thread safe?
  GLRenderbufferShPtr rtn(new GLRenderbuffer(_prntRenderer, width, height, internalFormat, numSamples));
  _glResources.push_back(GLResourceWkPtr(rtn));

  return rtn;
}

GLTexture2dShPtr GLResourceManager::createTexture2d(size_t width,
                                                    size_t height,
                                                    GLenum internalFormat,
                                                    GLenum pixelFormat,
                                                    GLenum pixelType,
                                                    const GLTexture2dSampleProps& sampleProps,
                                                    size_t numSamples) {
  CHECK(!_prntRenderer.expired());

  // TODO(croot): make thread safe?
  GLTexture2dShPtr rtn(
      new GLTexture2d(_prntRenderer, width, height, internalFormat, pixelFormat, pixelType, sampleProps, numSamples));
  _glResources.push_back(GLResourceWkPtr(rtn));

  return rtn;
}

GLTexture2dArrayShPtr GLResourceManager::createTexture2dArray(size_t width,
                                                              size_t height,
                                                              size_t depth,
                                                              GLenum internalFormat,
                                                              const GLTexture2dSampleProps& sampleProps,
                                                              size_t numSamples) {
  CHECK(!_prntRenderer.expired());

  // TODO(croot): make thread safe?
  GLTexture2dArrayShPtr rtn(
      new GLTexture2dArray(_prntRenderer, width, height, depth, internalFormat, sampleProps, numSamples));

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
  _glResources.push_back(GLResourceWkPtr(rtn));

  return rtn;
}

GLVertexBufferShPtr GLResourceManager::createVertexBuffer(GLenum usage) {
  CHECK(!_prntRenderer.expired());

  // TODO(croot): make thread safe?
  GLVertexBufferShPtr rtn(new GLVertexBuffer(_prntRenderer, usage));
  _glResources.push_back(GLResourceWkPtr(rtn));

  return rtn;
}

GLVertexBufferShPtr GLResourceManager::createVertexBuffer(size_t numBytes, GLenum usage) {
  CHECK(!_prntRenderer.expired());

  // TODO(croot): make thread safe?
  GLVertexBufferShPtr rtn(new GLVertexBuffer(_prntRenderer, numBytes, usage));
  _glResources.push_back(GLResourceWkPtr(rtn));

  return rtn;
}

GLVertexBufferShPtr GLResourceManager::createVertexBuffer(const Resources::GLBufferLayoutShPtr& layoutPtr,
                                                          GLenum usage) {
  CHECK(!_prntRenderer.expired());

  // TODO(croot): make thread safe?
  GLVertexBufferShPtr rtn(new GLVertexBuffer(_prntRenderer, layoutPtr, usage));
  _glResources.push_back(GLResourceWkPtr(rtn));

  return rtn;
}

GLVertexArrayShPtr GLResourceManager::createVertexArray() {
  CHECK(!_prntRenderer.expired());

  // TODO(croot): make thread safe?
  GLVertexArrayShPtr rtn(new GLVertexArray(_prntRenderer));
  _glResources.push_back(GLResourceWkPtr(rtn));

  return rtn;
}

GLVertexArrayShPtr GLResourceManager::createVertexArray(const VboAttrToShaderAttrMap& vboAttrToShaderAttrMap) {
  CHECK(!_prntRenderer.expired());

  // TODO(croot): make thread safe?
  GLVertexArrayShPtr rtn(new GLVertexArray(_prntRenderer, vboAttrToShaderAttrMap));
  _glResources.push_back(GLResourceWkPtr(rtn));

  return rtn;
}

}  // namespace GL
}  // namespace Rendering

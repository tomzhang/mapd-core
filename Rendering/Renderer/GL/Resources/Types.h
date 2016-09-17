#ifndef RENDERING_GL_RESOURCES_TYPES_H_
#define RENDERING_GL_RESOURCES_TYPES_H_

#include <memory>
#include <string>

namespace Rendering {
namespace GL {
namespace Resources {

class GLResource;
typedef std::weak_ptr<GLResource> GLResourceWkPtr;
typedef std::shared_ptr<GLResource> GLResourceShPtr;

struct GLTexture2dSampleProps;
class GLShader;
typedef std::weak_ptr<GLShader> GLShaderWkPtr;
typedef std::shared_ptr<GLShader> GLShaderShPtr;

class GLFramebuffer;
typedef std::weak_ptr<GLFramebuffer> GLFramebufferWkPtr;
typedef std::shared_ptr<GLFramebuffer> GLFramebufferShPtr;

class GLRenderbuffer;
typedef std::weak_ptr<GLRenderbuffer> GLRenderbufferWkPtr;
typedef std::shared_ptr<GLRenderbuffer> GLRenderbufferShPtr;

class GLTexture2d;
typedef std::weak_ptr<GLTexture2d> GLTexture2dWkPtr;
typedef std::shared_ptr<GLTexture2d> GLTexture2dShPtr;

class GLTexture2dArray;
typedef std::weak_ptr<GLTexture2dArray> GLTexture2dArrayWkPtr;
typedef std::shared_ptr<GLTexture2dArray> GLTexture2dArrayShPtr;

class GLBaseBuffer;
typedef std::weak_ptr<GLBaseBuffer> GLBaseBufferWkPtr;
typedef std::shared_ptr<GLBaseBuffer> GLBaseBufferShPtr;

class GLLayoutManagerBuffer;
typedef std::weak_ptr<GLLayoutManagerBuffer> GLLayoutManagerBufferWkPtr;
typedef std::shared_ptr<GLLayoutManagerBuffer> GLLayoutManagerBufferShPtr;

class GLVertexBuffer;
typedef std::weak_ptr<GLVertexBuffer> GLVertexBufferWkPtr;
typedef std::shared_ptr<GLVertexBuffer> GLVertexBufferShPtr;

class GLPixelBuffer2d;
typedef std::weak_ptr<GLPixelBuffer2d> GLPixelBuffer2dWkPtr;
typedef std::shared_ptr<GLPixelBuffer2d> GLPixelBuffer2dShPtr;

class GLUniformBuffer;
typedef std::weak_ptr<GLUniformBuffer> GLUniformBufferWkPtr;
typedef std::shared_ptr<GLUniformBuffer> GLUniformBufferShPtr;

class GLIndexBuffer;
typedef std::weak_ptr<GLIndexBuffer> GLIndexBufferWkPtr;
typedef std::shared_ptr<GLIndexBuffer> GLIndexBufferShPtr;

class GLBaseIndirectDrawBuffer;
typedef std::weak_ptr<GLBaseIndirectDrawBuffer> GLIndirectDrawBufferWkPtr;
typedef std::shared_ptr<GLBaseIndirectDrawBuffer> GLIndirectDrawBufferShPtr;

class GLIndirectDrawVertexBuffer;
typedef std::weak_ptr<GLIndirectDrawVertexBuffer> GLIndirectDrawVertexBufferWkPtr;
typedef std::shared_ptr<GLIndirectDrawVertexBuffer> GLIndirectDrawVertexBufferShPtr;

class GLIndirectDrawIndexBuffer;
typedef std::weak_ptr<GLIndirectDrawIndexBuffer> GLIndirectDrawIndexBufferWkPtr;
typedef std::shared_ptr<GLIndirectDrawIndexBuffer> GLIndirectDrawIndexBufferShPtr;

class GLBaseBufferLayout;
typedef std::weak_ptr<GLBaseBufferLayout> GLBufferLayoutWkPtr;
typedef std::shared_ptr<GLBaseBufferLayout> GLBufferLayoutShPtr;

class GLCustomBufferLayout;
typedef std::shared_ptr<GLCustomBufferLayout> GLCustomBufferLayoutShPtr;

class GLInterleavedBufferLayout;
typedef std::shared_ptr<GLInterleavedBufferLayout> GLInterleavedBufferLayoutShPtr;

class GLSequentialBufferLayout;
typedef std::shared_ptr<GLSequentialBufferLayout> GLSequentialBufferLayoutShPtr;

class GLShaderBlockLayout;
typedef std::shared_ptr<GLShaderBlockLayout> GLShaderBlockLayoutShPtr;

class GLVertexArray;
typedef std::weak_ptr<GLVertexArray> GLVertexArrayWkPtr;
typedef std::shared_ptr<GLVertexArray> GLVertexArrayShPtr;

struct IndirectDrawVertexData {
  unsigned int count;
  unsigned int instanceCount;
  unsigned int firstIndex;
  unsigned int baseInstance;

  IndirectDrawVertexData() : count(0), instanceCount(0), firstIndex(0), baseInstance(0) {}

  IndirectDrawVertexData(unsigned int count,
                         unsigned int firstIndex = 0,
                         unsigned int instanceCount = 1,
                         unsigned int baseInstance = 0)
      : count(count), instanceCount(instanceCount), firstIndex(firstIndex), baseInstance(baseInstance) {}
};

struct IndirectDrawIndexData {
  unsigned int count;
  unsigned int instanceCount;
  unsigned int firstIndex;
  unsigned int baseVertex;
  unsigned int baseInstance;

  IndirectDrawIndexData() : count(0), instanceCount(0), firstIndex(0), baseVertex(0), baseInstance(0) {}

  IndirectDrawIndexData(unsigned int count,
                        unsigned int firstIndex = 0,
                        unsigned int baseVertex = 0,
                        unsigned int instanceCount = 1,
                        unsigned int baseInstance = 0)
      : count(count),
        instanceCount(instanceCount),
        firstIndex(firstIndex),
        baseVertex(baseVertex),
        baseInstance(baseInstance) {}
};

}  // namespace GL
}  // namespace Resources
}  // namespace Rendering

#endif  // RENDERING_GL_RESOURCES_TYPES_H_

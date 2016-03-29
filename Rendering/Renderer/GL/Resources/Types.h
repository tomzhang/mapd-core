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

class GLBaseBufferLayout;
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

}  // namespace GL
}  // namespace Resources
}  // namespace Rendering

#endif  // RENDERING_GL_RESOURCES_TYPES_H_

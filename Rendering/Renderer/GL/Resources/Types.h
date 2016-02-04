#ifndef RENDERING_GL_RESOURCES_GLRESOURCETYPES_H_
#define RENDERING_GL_RESOURCES_GLRESOURCETYPES_H_

#include <GL/glew.h>
#include <memory>
#include <map>
#include <string>

namespace Rendering {
namespace GL {
namespace Resources {

enum class GLResourceType { SHADER = 0, FRAMEBUFFER, RENDERBUFFER, TEXTURE_2D, VERTEXBUFFER, VERTEXARRAY };

enum class GLBufferType { VERTEX_BUFFER = 0 };

enum class FboBind { READ = GL_READ_FRAMEBUFFER, DRAW = GL_DRAW_FRAMEBUFFER, READ_AND_DRAW = GL_FRAMEBUFFER };

class GLResource;
typedef std::weak_ptr<GLResource> GLResourceWkPtr;
typedef std::shared_ptr<GLResource> GLResourceShPtr;

class GLShader;
typedef std::weak_ptr<GLShader> GLShaderWkPtr;
typedef std::shared_ptr<GLShader> GLShaderShPtr;

class GLFramebuffer;
typedef std::weak_ptr<GLFramebuffer> GLFramebufferWkPtr;
typedef std::shared_ptr<GLFramebuffer> GLFramebufferShPtr;
typedef std::map<GLenum, GLResourceShPtr> GLFramebufferAttachmentMap;

class GLRenderbuffer;
typedef std::weak_ptr<GLRenderbuffer> GLRenderbufferWkPtr;
typedef std::shared_ptr<GLRenderbuffer> GLRenderbufferShPtr;

class GLTexture2d;
typedef std::weak_ptr<GLTexture2d> GLTexture2dWkPtr;
typedef std::shared_ptr<GLTexture2d> GLTexture2dShPtr;

class GLBaseBuffer;
typedef std::weak_ptr<GLBaseBuffer> GLBaseBufferWkPtr;
typedef std::shared_ptr<GLBaseBuffer> GLBaseBufferShPtr;

class GLVertexBuffer;
typedef std::weak_ptr<GLVertexBuffer> GLVertexBufferWkPtr;
typedef std::shared_ptr<GLVertexBuffer> GLVertexBufferShPtr;

class GLBaseBufferLayout;
typedef std::shared_ptr<GLBaseBufferLayout> GLBufferLayoutShPtr;

class GLCustomBufferLayout;
typedef std::shared_ptr<GLCustomBufferLayout> GLCustomBufferLayoutShPtr;

class GLInterleavedBufferLayout;
typedef std::shared_ptr<GLInterleavedBufferLayout> GLInterleavedBufferLayoutShPtr;

class GLSequentialBufferLayout;
typedef std::shared_ptr<GLSequentialBufferLayout> GLSequentialBufferLayoutShPtr;

class GLVertexArray;
typedef std::weak_ptr<GLVertexArray> GLVertexArrayWkPtr;
typedef std::shared_ptr<GLVertexArray> GLVertexArrayShPtr;

std::string to_string(const GLResourceType value);

}  // namespace GL
}  // namespace Resources
}  // namespace Rendering

std::ostream& operator<<(std::ostream& os, const Rendering::GL::Resources::GLResourceType value);

#endif  // RENDERING_GL_RESOURCES_GLRESOURCETYPES_H_

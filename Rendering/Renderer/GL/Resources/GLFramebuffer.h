#ifndef RENDERING_GL_RESOURCES_GLFRAMEBUFFER_H_
#define RENDERING_GL_RESOURCES_GLFRAMEBUFFER_H_

#include "GLResource.h"
#include "Types.h"

#include <boost/multi_index_container.hpp>
#include <boost/multi_index/hashed_index.hpp>
#include <boost/multi_index/random_access_index.hpp>
#include <boost/multi_index/member.hpp>

#include <vector>
#include <unordered_map>
#include <memory>

namespace Rendering {
namespace GL {
namespace Resources {

namespace detail {

/**
 * AttachmentContainer
 *  Class used to manage attachments of a framebuffer
 */

struct AttachmentData {
  GLenum attachmentType;
  GLuint attachedTextureId;
};

struct inorder {};

class AttachmentContainer {
 public:
  ~AttachmentContainer();

  bool hasAttachment(GLenum attachment);
  void addTexture2dAttachment(GLenum attachment, GLuint tex);
  void addRenderbufferAttachment(GLenum attachment, GLuint rbo);
  void removeAttachment(GLenum attachment);
  void enableAttachments();

  void clear();

  static bool isColorAttachment(GLenum attachment, int maxColorAttachments);

 private:
  AttachmentContainer();

  typedef ::boost::multi_index_container<
      AttachmentData,
      ::boost::multi_index::indexed_by<
          // hashed on name
          ::boost::multi_index::hashed_unique<
              ::boost::multi_index::member<AttachmentData, GLenum, &AttachmentData::attachmentType>>,

          ::boost::multi_index::random_access<::boost::multi_index::tag<inorder>>>> AttachmentMap;

  typedef AttachmentMap::index<inorder>::type AttachmentMap_in_order;

  bool _dirty;
  std::vector<GLenum> _activeAttachments;

  AttachmentMap _attachmentMap;

  friend class ::Rendering::GL::Resources::GLFramebuffer;

  int _maxColorAttachments;
};

}  // namespace detail

///////////////////////////////////////////////////////////////////////
/**
 * GLFramebuffer
 *  Class used for managing framebuffers for backend rendering of
 *  database queries.
 */
class GLFramebuffer : public GLResource {
 public:
  ~GLFramebuffer();

  GLResourceType getResourceType() const final { return GLResourceType::FRAMEBUFFER; }
  GLuint getId() const final { return _fbo; }

  int getWidth() const;
  int getHeight() const;

  void resize(int width, int height);

 private:
  GLFramebuffer(const RendererWkPtr& rendererPtr, const GLFramebufferAttachmentMap& attachments);

  // framebuffer object id
  GLuint _fbo;

  // texture buffers
  std::vector<GLTexture2dShPtr> _textureBuffers;

  // render buffers
  std::vector<GLRenderbufferShPtr> _renderBuffers;

  // attachment manager
  detail::AttachmentContainer _attachmentManager;

  void _initResource(const GLFramebufferAttachmentMap& attachments);
  void _cleanupResource() final;
  void _makeEmpty() final;

  // void bindToRenderer(GLRenderer* renderer, FboBind bindType = FboBind::READ_AND_DRAW);

  friend void ::Rendering::GL::State::GLBindState::bindFramebuffer(
      ::Rendering::GL::Resources::FboBind,
      const ::Rendering::GL::Resources::GLFramebufferShPtr&);
  friend class ::Rendering::GL::GLResourceManager;
};

}  // namespace Resources
}  // namespace GL
}  // namespace Rendering

#endif  // RENDERING_GL_RESOURCES_GLFRAMEBUFFER_H_

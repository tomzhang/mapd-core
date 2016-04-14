#ifndef RENDERING_GL_RESOURCES_GLFRAMEBUFFER_H_
#define RENDERING_GL_RESOURCES_GLFRAMEBUFFER_H_

#include "GLResource.h"
#include "Types.h"

#include <GL/glew.h>
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
  bool active;
};

struct inorder {};

struct ChangeActive {
  ChangeActive(bool active) : active(active) {}

  void operator()(AttachmentData& data) { data.active = active; }

 private:
  bool active;
};

class AttachmentContainer {
 public:
  ~AttachmentContainer();

  bool hasAttachment(GLenum attachment);
  void addTexture2dAttachment(GLenum attachment, GLuint tex);
  void addRenderbufferAttachment(GLenum attachment, GLuint rbo);
  void removeAttachment(GLenum attachment);

  void enableAllAttachments();
  void disableAllAttachments();
  void enableAttachments(const std::vector<GLenum>& attachments);
  void disableAttachments(const std::vector<GLenum>& attachments);
  void enableAttachment(GLenum attachment);
  void disableAttachment(GLenum attachment);

  void enableGLAttachments();

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

typedef std::map<GLenum, GLResourceShPtr> GLFramebufferAttachmentMap;

///////////////////////////////////////////////////////////////////////
/**
 * GLFramebuffer
 *  Class used for managing framebuffers for backend rendering of
 *  database queries.
 */
class GLFramebuffer : public GLResource {
 public:
  ~GLFramebuffer();

  GLuint getId() const final { return _fbo; }
  GLenum getTarget() const final { return GL_FRAMEBUFFER; }

  size_t getWidth() const;
  size_t getHeight() const;

  void readPixels(GLenum attachment,
                  size_t startx,
                  size_t starty,
                  size_t width,
                  size_t height,
                  GLenum format,
                  GLenum type,
                  GLvoid* data);

  void copyPixelsToBoundPixelBuffer(GLenum attachment,
                                    size_t startx,
                                    size_t starty,
                                    size_t width,
                                    size_t height,
                                    size_t offsetBytes,
                                    GLenum format,
                                    GLenum type);

  void copyPixelsToPixelBuffer(GLenum attachment,
                               size_t startx,
                               size_t starty,
                               size_t width,
                               size_t height,
                               size_t offsetBytes,
                               GLenum format,
                               GLenum type,
                               GLPixelBuffer2dShPtr& pbo);

  void resize(size_t width, size_t height);

  void enableAllAttachments();
  void disableAllAttachments();
  void enableAttachments(const std::vector<GLenum>& activeAttachments);
  void disableAttachments(const std::vector<GLenum>& attachmentsToDisable);
  void enableAttachment(GLenum attachment);
  void disableAttachment(GLenum attachment);

  void activateEnabledAttachmentsForDrawing();

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

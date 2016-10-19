#ifndef QUERYRENDERER_QUERYFRAMEBUFFER_H_
#define QUERYRENDERER_QUERYFRAMEBUFFER_H_

#include "Types.h"
#include <Rendering/Renderer.h>
#include <Rendering/Renderer/GL/Types.h>
#include <Rendering/Renderer/GL/Resources/Enums.h>
#include <Rendering/Renderer/GL/Resources/Types.h>
// #include <vector>
// #include <unordered_map>
// #include <GL/glew.h>
// #include <memory>

// #include <boost/multi_index_container.hpp>
// #include <boost/multi_index/hashed_index.hpp>
// #include <boost/multi_index/random_access_index.hpp>
// #include <boost/multi_index/member.hpp>

// using namespace ::boost;
// using namespace ::boost::multi_index;

namespace QueryRenderer {

class QueryRenderCompositor;

enum class FboColorBuffer { COLOR_BUFFER = 0, ID_BUFFER, ID2_BUFFER, MAX_TEXTURE_BUFFERS = ID2_BUFFER };
enum class FboRenderBuffer { DEPTH_BUFFER = 0, MAX_RENDER_BUFFERS = DEPTH_BUFFER };

///////////////////////////////////////////////////////////////////////
/**
 * QueryFramebuffer
 *  Class used for managing framebuffers for backend rendering of
 *  database queries.
 */

class QueryFramebuffer {
 public:
  QueryFramebuffer(::Rendering::GL::GLRenderer* renderer,
                   int width,
                   int height,
                   bool doHitTest = false,
                   bool doDepthTest = false,
                   size_t numSamples = 1);

  QueryFramebuffer(QueryRenderCompositor* compositor, ::Rendering::GL::GLRenderer* renderer);
  ~QueryFramebuffer();

  void resize(int width, int height);

  std::pair<std::vector<GLenum>, std::vector<GLenum>> getEnabledDisabledAttachments() const;
  void bindToRenderer(
      ::Rendering::GL::GLRenderer* renderer,
      ::Rendering::GL::Resources::FboBind bindType = ::Rendering::GL::Resources::FboBind::READ_AND_DRAW);

  std::shared_ptr<unsigned char> readColorBuffer(size_t startx = 0, size_t starty = 0, int width = -1, int height = -1);
  void readIdBuffer(size_t startx,
                    size_t starty,
                    int width,
                    int height,
                    unsigned int* idBuffer,
                    const FboColorBuffer idBufferType = FboColorBuffer::ID_BUFFER);
  std::shared_ptr<unsigned int> readIdBuffer(size_t startx = 0,
                                             size_t starty = 0,
                                             int width = -1,
                                             int height = -1,
                                             const FboColorBuffer idBufferType = FboColorBuffer::ID_BUFFER);

  void blitToFramebuffer(QueryFramebuffer& dstFboPtr, size_t startx, size_t starty, size_t width, size_t height);

  void copyRowIdBufferToPbo(QueryIdMapPixelBufferUIntShPtr& pbo);
  void copyTableIdBufferToPbo(QueryIdMapPixelBufferIntShPtr& pbo);

  size_t getWidth() const;
  size_t getHeight() const;
  size_t getNumSamples() const;
  bool doHitTest() const { return _doHitTest; }
  bool doDepthTest() const { return _doDepthTest; }

  void setHitTest(bool doHitTest);
  void setDepthTest(bool doDepthTest);

  ::Rendering::Renderer* getRenderer();
  ::Rendering::GL::GLRenderer* getGLRenderer();

  ::Rendering::GL::Resources::GLFramebufferShPtr getGLFramebuffer() const { return _fbo; }
  ::Rendering::GL::Resources::GLTexture2dShPtr getGLTexture2d(FboColorBuffer texType) const;
  ::Rendering::GL::Resources::GLRenderbufferShPtr getGLRenderbuffer(
      FboRenderBuffer rboType = FboRenderBuffer::DEPTH_BUFFER) const;

  GLuint getId(FboColorBuffer buffer);
  GLuint getId(FboRenderBuffer buffer);

  static ::Rendering::GL::Resources::GLTexture2dShPtr createFboTexture2d(
      ::Rendering::GL::GLResourceManagerShPtr& rsrcMgr,
      FboColorBuffer texType,
      size_t width,
      size_t height,
      size_t numSamples);

  static ::Rendering::GL::Resources::GLRenderbufferShPtr createFboRenderbuffer(
      ::Rendering::GL::GLResourceManagerShPtr& rsrcMgr,
      FboRenderBuffer rboType,
      size_t width,
      size_t height,
      size_t numSamples);

 private:
  bool _defaultDoHitTest, _defaultDoDepthTest;
  bool _doHitTest, _doDepthTest;

  ::Rendering::GL::Resources::GLTexture2dShPtr _rgbaTex;
  ::Rendering::GL::Resources::GLTexture2dShPtr _idTex1;
  ::Rendering::GL::Resources::GLTexture2dShPtr _idTex2;
  ::Rendering::GL::Resources::GLRenderbufferShPtr _rbo;
  ::Rendering::GL::Resources::GLFramebufferShPtr _fbo;

  // framebuffer object id
  // GLuint _fbo;

  // texture buffers
  // std::vector<GLuint> _textureBuffers;

  // render buffers
  // std::vector<GLuint> _renderBuffers;

  // attachment manager
  // AttachmentContainer _attachmentManager;

  void _init(::Rendering::GL::GLRenderer* renderer, int width, int height, size_t numSamples);
  void _init(QueryRenderCompositor* compositor, ::Rendering::GL::GLRenderer* renderer);
};

typedef std::unique_ptr<QueryFramebuffer> QueryFramebufferUqPtr;
typedef std::shared_ptr<QueryFramebuffer> QueryFramebufferShPtr;
}  // namespace QueryRenderer

#endif  // QUERYRENDERER_QUERYFRAMEBUFFER_H_

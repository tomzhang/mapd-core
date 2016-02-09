#ifndef QUERYRENDERER_QUERYFRAMEBUFFER_H_
#define QUERYRENDERER_QUERYFRAMEBUFFER_H_

#include <Rendering/Renderer/GL/Types.h>
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

enum class FboColorBuffer { COLOR_BUFFER = 0, ID_BUFFER, MAX_TEXTURE_BUFFERS = ID_BUFFER };
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
                   bool doDepthTest = false);
  ~QueryFramebuffer();

  void resize(int width, int height);
  void bindToRenderer(
      ::Rendering::GL::GLRenderer* renderer,
      ::Rendering::GL::Resources::FboBind bindType = ::Rendering::GL::Resources::FboBind::READ_AND_DRAW);

  std::shared_ptr<unsigned char> readColorBuffer(size_t startx, size_t starty, size_t width, size_t height);

  int getWidth() const;
  int getHeight() const;

 private:
  bool _doHitTest, _doDepthTest;

  ::Rendering::GL::Resources::GLTexture2dShPtr _rgbaTex;
  ::Rendering::GL::Resources::GLTexture2dShPtr _idTex;
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

  void _init(::Rendering::GL::GLRenderer* renderer, int width, int height);
};

typedef std::unique_ptr<QueryFramebuffer> QueryFramebufferUqPtr;
typedef std::shared_ptr<QueryFramebuffer> QueryFramebufferShPtr;

}  // namespace QueryRenderer

#endif  // QUERYRENDERER_QUERYFRAMEBUFFER_H_

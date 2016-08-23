#include "GLRenderer.h"
#include "GLResourceManager.h"
#include "../../RenderError.h"
#include "../../Objects/ColorRGBA.h"
#include "Resources/GLFramebuffer.h"
#include "GLWindow.h"
#include <thread>
#include <mutex>
#include <array>

namespace Rendering {
namespace GL {

using Objects::ColorRGBA;
using Resources::GLResourceType;
using Resources::GLResourceShPtr;
using Resources::GLTexture2dShPtr;
using Resources::GLTexture2dArrayShPtr;
using Resources::FboBind;
using Resources::GLFramebufferShPtr;
using Resources::GLShaderShPtr;
using Resources::GLVertexArrayShPtr;
using Resources::GLRenderbufferShPtr;
using Resources::GLPixelBuffer2dShPtr;
using Resources::GLUniformBufferShPtr;
using Resources::GLIndexBufferShPtr;

// TODO(croot): make these std::weak_ptr?
thread_local GLRenderer* _currentRenderer;
thread_local Window* _currentWindow;

std::mutex GLRenderer::_currRendererMtx;

GLRenderer* GLRenderer::getCurrentThreadRenderer() {
  return _currentRenderer;
}

Window* GLRenderer::getCurrentThreadWindow() {
  return _currentWindow;
}

// static void setInactiveRendererOnCurrentThread() {
//   if (_currentRenderer) {
//     _currentRenderer->makeInactive();
//   }
// }

GLRenderer::GLRenderer(const RendererSettings& settings)
    : Renderer(settings), _glRsrcMgrPtr(nullptr), _glewInitialized(false), _bindState(this) {
#ifdef GLEW_MX
  memset(&_glewContext, 0, sizeof(GLEWContext));
#endif
}

GLRenderer::~GLRenderer() {
  _glewInitialized = false;

  // #ifdef GLEW_MX
  //   memset(&_glewContext, 0, sizeof(GLEWContext));
  // #endif
}

void GLRenderer::initialize() {
  CHECK(!isInitialized());

  // call the derived class's initialize
  initializeGL();

  setInitialized();
}

void GLRenderer::makeActiveOnCurrentThread(Window* window) {
  // TODO(croot): a window and context can be activated
  // so we need to add the window as a per-thread item.

  // lock access to the current renderer / current window
  std::lock_guard<std::mutex> thread_lock(_currRendererMtx);

  // TODO(croot): Make thread safe while traversing windows?
  Window* windowToUse = nullptr;
  if (!window) {
    windowToUse = getPrimaryWindow();
  } else {
    // TODO(croot): improve this by using a map instead of a list?
    for (auto& attachedWindow : _attachedWindows) {
      if (window == attachedWindow) {
        windowToUse = window;
        break;
      }
    }
  }

  RUNTIME_EX_ASSERT(windowToUse != nullptr,
                    "Cannot make renderer active. Window to make active alongside is not an attached window.");

  if (_currentRenderer == this && _currentWindow == windowToUse) {
    // already active
    return;
  }

  _makeActiveOnCurrentThreadImpl(windowToUse);

  _currentRenderer = this;
  _currentWindow = windowToUse;
}

void GLRenderer::makeInactive() {
  std::lock_guard<std::mutex> thread_lock(_currRendererMtx);

  if (_currentRenderer == this) {
    _makeInactiveImpl();

    _currentRenderer = nullptr;
    _currentWindow = nullptr;
  }

  // TODO(croot): should I throw an error or log a warning
  // if this renderer isn't current?
}

bool GLRenderer::isActiveOnCurrentThread(Window* window) {
  std::lock_guard<std::mutex> thread_lock(_currRendererMtx);

  Window* currWindow = (window != nullptr ? window : getPrimaryWindow());
  if (_currentRenderer != this || _currentWindow != currWindow) {
    return false;
  }
  return true;
}

GLResourceManagerShPtr GLRenderer::getResourceManager() {
  RUNTIME_EX_ASSERT(_glRsrcMgrPtr != nullptr,
                    "The renderer has not been properly created. Cannot get resource manager.");
  return _glRsrcMgrPtr;
}

void GLRenderer::setResourceManager(const GLResourceManagerShPtr& glRsrcMgr) {
  CHECK(_glRsrcMgrPtr == nullptr);
  _glRsrcMgrPtr = glRsrcMgr;
}

#ifdef GLEW_MX
GLEWContext* GLRenderer::glewGetContext() {
  return &_glewContext;
}

const GLEWContext* GLRenderer::glewGetContext() const {
  return &_glewContext;
}
#endif

const GLWindow* GLRenderer::getPrimaryGLWindow() const {
  const Window* window = getPrimaryWindow();
  if (window) {
    return dynamic_cast<const GLWindow*>(window);
  }
  return nullptr;
}

GLWindow* GLRenderer::getPrimaryGLWindow() {
  return const_cast<GLWindow*>(static_cast<const GLRenderer&>(*this).getPrimaryGLWindow());
}

size_t GLRenderer::getGpuId() const {
  const GLWindow* primaryWindow = getPrimaryGLWindow();
  RUNTIME_EX_ASSERT(primaryWindow != nullptr,
                    "The GLRenderer has not been initialized yet. Cannot get the Gpu ID associated with it.");
  return primaryWindow->getGpuId();
}

void GLRenderer::setClearColor(const ColorRGBA& color) {
  MAPD_CHECK_GL_ERROR(glClearColor(color.r(), color.g(), color.b(), color.a()));
}

void GLRenderer::setClearColor(float r, float g, float b, float a) {
  MAPD_CHECK_GL_ERROR(glClearColor(r, g, b, a));
}

void GLRenderer::clearColor() {
  MAPD_CHECK_GL_ERROR(glClear(GL_COLOR_BUFFER_BIT));
}

void GLRenderer::clearAll() {
  MAPD_CHECK_GL_ERROR(glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT | GL_STENCIL_BUFFER_BIT));
}

void GLRenderer::setViewport(int x, int y, int width, int height) {
  // TODO(croot): support indexed viewports?
  MAPD_CHECK_GL_ERROR(glViewport(x, y, width, height));
}

void GLRenderer::setViewport(const Viewport& viewport) {
  // TODO(croot): support indexed viewports?
  MAPD_CHECK_GL_ERROR(glViewport(viewport.getXPos(), viewport.getYPos(), viewport.getWidth(), viewport.getHeight()));
}

Objects::Viewport GLRenderer::getViewport() const {
  // TODO(croot): support indexed viewports?

  std::array<int, 4> data;
  MAPD_CHECK_GL_ERROR(glGetIntegerv(GL_VIEWPORT, &data[0]));

  return Objects::Viewport(data);
}

void GLRenderer::enable(GLenum attr) {
  MAPD_CHECK_GL_ERROR(glEnable(attr));
}

void GLRenderer::disable(GLenum attr) {
  MAPD_CHECK_GL_ERROR(glDisable(attr));
}

void GLRenderer::setBlendFunc(GLenum srcFactor, GLenum dstFactor, int drawBufferId) {
  if (drawBufferId >= 0) {
    MAPD_CHECK_GL_ERROR(glBlendFunci(drawBufferId, srcFactor, dstFactor));
  } else {
    MAPD_CHECK_GL_ERROR(glBlendFunc(srcFactor, dstFactor));
  }
}

void GLRenderer::setBlendFuncSeparate(GLenum srcRGB,
                                      GLenum dstRGB,
                                      GLenum srcAlpha,
                                      GLenum dstAlpha,
                                      int drawBufferId) {
  if (drawBufferId >= 0) {
    MAPD_CHECK_GL_ERROR(glBlendFuncSeparatei(drawBufferId, srcRGB, dstRGB, srcAlpha, dstAlpha));
  } else {
    MAPD_CHECK_GL_ERROR(glBlendFuncSeparate(srcRGB, dstRGB, srcAlpha, dstAlpha));
  }
}

void GLRenderer::setBlendEquation(GLenum mode, int drawBufferId) {
  if (drawBufferId >= 0) {
    MAPD_CHECK_GL_ERROR(glBlendEquationi(drawBufferId, mode));
  } else {
    MAPD_CHECK_GL_ERROR(glBlendEquation(mode));
  }
}

void GLRenderer::setBlendEquationSeparate(GLenum modeRGB, GLenum modeAlpha, int drawBufferId) {
  if (drawBufferId >= 0) {
    MAPD_CHECK_GL_ERROR(glBlendEquationSeparatei(drawBufferId, modeRGB, modeAlpha));
  } else {
    MAPD_CHECK_GL_ERROR(glBlendEquationSeparate(modeRGB, modeAlpha));
  }
}

void GLRenderer::setLineWidth(const float lineWidth) {
  MAPD_CHECK_GL_ERROR(glLineWidth(lineWidth));
}

void GLRenderer::setMinSampleShading(const float minSampleShading) {
  MAPD_CHECK_GL_ERROR(glMinSampleShading(minSampleShading));
}

void GLRenderer::bindResource(const GLResourceShPtr& rsrc) {
  _bindState.bindResource(rsrc);
}

void GLRenderer::bindTexture2d(const GLTexture2dShPtr& texRsrc) {
  _bindState.bindTexture2d(texRsrc);
}

void GLRenderer::bindTexture2dArray(const Resources::GLTexture2dArrayShPtr& texArrayRsrc) {
  _bindState.bindTexture2dArray(texArrayRsrc);
}

void GLRenderer::bindVertexBuffer(const Resources::GLVertexBufferShPtr& vboRsrc) {
  _bindState.bindVertexBuffer(vboRsrc);
}

void GLRenderer::bindFramebuffer(const GLFramebufferShPtr& fboRsrc, const FboBind bindType) {
  _bindState.bindFramebuffer(fboRsrc, bindType);
}

void GLRenderer::bindShader(const GLShaderShPtr& shaderRsrc) {
  _bindState.bindShader(shaderRsrc);
}

void GLRenderer::bindVertexArray(const GLVertexArrayShPtr& vaoRsrc) {
  _bindState.bindVertexArray(vaoRsrc);
}

void GLRenderer::bindRenderbuffer(const Resources::GLRenderbufferShPtr& rboRsrc) {
  _bindState.bindRenderbuffer(rboRsrc);
}

void GLRenderer::bindReadPixelBuffer(const Resources::GLPixelBuffer2dShPtr& pboRsrc) {
  _bindState.bindReadPixelBuffer(pboRsrc);
}

void GLRenderer::bindWritePixelBuffer(const Resources::GLPixelBuffer2dShPtr& pboRsrc) {
  _bindState.bindWritePixelBuffer(pboRsrc);
}

void GLRenderer::bindUniformBuffer(const Resources::GLUniformBufferShPtr& uboRsrc) {
  _bindState.bindUniformBuffer(uboRsrc);
}

void GLRenderer::bindIndexBuffer(const Resources::GLIndexBufferShPtr& uboRsrc) {
  _bindState.bindIndexBuffer(uboRsrc);
}

void GLRenderer::bindIndirectDrawBuffer(const Resources::GLIndirectDrawBufferShPtr& indirectRsrc) {
  _bindState.bindIndirectDrawBuffer(indirectRsrc);
}

Resources::GLTexture2dShPtr GLRenderer::getBoundTexture2d() const {
  return _bindState.getBoundTexture2d();
}

bool GLRenderer::hasBoundTexture2d() const {
  return _bindState.hasBoundTexture2d();
}

Resources::GLTexture2dArrayShPtr GLRenderer::getBoundTexture2dArray() const {
  return _bindState.getBoundTexture2dArray();
}

bool GLRenderer::hasBoundTexture2dArray() const {
  return _bindState.hasBoundTexture2dArray();
}

Resources::GLVertexBufferShPtr GLRenderer::getBoundVertexBuffer() const {
  return _bindState.getBoundVertexBuffer();
}

bool GLRenderer::hasBoundVertexBuffer() const {
  return _bindState.hasBoundVertexBuffer();
}

GLFramebufferShPtr GLRenderer::getBoundFramebuffer(FboBind bindType) const {
  return _bindState.getBoundFramebuffer(bindType);
}

bool GLRenderer::hasBoundFramebuffer(Resources::FboBind bindType) const {
  return _bindState.hasBoundFramebuffer(bindType);
}

GLShaderShPtr GLRenderer::getBoundShader() const {
  return _bindState.getBoundShader();
}

bool GLRenderer::hasBoundShader() const {
  return _bindState.hasBoundShader();
}

GLVertexArrayShPtr GLRenderer::getBoundVertexArray() const {
  return _bindState.getBoundVertexArray();
}

bool GLRenderer::hasBoundVertexArray() const {
  return _bindState.hasBoundVertexArray();
}

GLRenderbufferShPtr GLRenderer::getBoundRenderbuffer() const {
  return _bindState.getBoundRenderbuffer();
}

bool GLRenderer::hasBoundRenderbuffer() const {
  return _bindState.hasBoundRenderbuffer();
}

GLPixelBuffer2dShPtr GLRenderer::getBoundReadPixelBuffer() const {
  return _bindState.getBoundReadPixelBuffer();
}

bool GLRenderer::hasBoundReadPixelBuffer() const {
  return _bindState.hasBoundReadPixelBuffer();
}

GLPixelBuffer2dShPtr GLRenderer::getBoundWritePixelBuffer() const {
  return _bindState.getBoundWritePixelBuffer();
}

bool GLRenderer::hasBoundWritePixelBuffer() const {
  return _bindState.hasBoundWritePixelBuffer();
}

GLUniformBufferShPtr GLRenderer::getBoundUniformBuffer() const {
  return _bindState.getBoundUniformBuffer();
}

bool GLRenderer::hasBoundUniformBuffer() const {
  return _bindState.hasBoundUniformBuffer();
}

GLIndexBufferShPtr GLRenderer::getBoundIndexBuffer() const {
  return _bindState.getBoundIndexBuffer();
}

bool GLRenderer::hasBoundIndexBuffer() const {
  return _bindState.hasBoundIndexBuffer();
}

Resources::GLIndirectDrawBufferShPtr GLRenderer::getBoundIndirectDrawBuffer() const {
  return _bindState.getBoundIndirectDrawBuffer();
}

bool GLRenderer::hasBoundIndirectDrawBuffer() const {
  return _bindState.hasBoundIndirectDrawBuffer();
}

Resources::GLIndirectDrawVertexBufferShPtr GLRenderer::getBoundIndirectDrawVertexBuffer() const {
  return _bindState.getBoundIndirectDrawVertexBuffer();
}

bool GLRenderer::hasBoundIndirectDrawVertexBuffer() const {
  return _bindState.hasBoundIndirectDrawVertexBuffer();
}

Resources::GLIndirectDrawIndexBufferShPtr GLRenderer::getBoundIndirectDrawIndexBuffer() const {
  return _bindState.getBoundIndirectDrawIndexBuffer();
}

bool GLRenderer::hasBoundIndirectDrawIndexBuffer() const {
  return _bindState.hasBoundIndirectDrawIndexBuffer();
}

void GLRenderer::drawVertexBuffers(GLenum primitiveMode, size_t startIndex, int numItemsToDraw) {
  CHECK(hasBoundVertexBuffer() || hasBoundVertexArray());

  if (numItemsToDraw < 0) {
    if (hasBoundVertexArray()) {
      numItemsToDraw = getBoundVertexArray()->numVertices();
    } else {
      numItemsToDraw = getBoundVertexBuffer()->numVertices();
    }
  }

  MAPD_CHECK_GL_ERROR(glDrawArrays(primitiveMode, startIndex, numItemsToDraw));
}

void GLRenderer::drawIndirectVertexBuffers(GLenum primitiveMode, size_t startIndex, int numItemsToDraw) {
  Resources::GLIndirectDrawVertexBufferShPtr indirectBuffer = getBoundIndirectDrawVertexBuffer();
  CHECK(indirectBuffer != nullptr && indirectBuffer->getResourceType() == GLResourceType::INDIRECT_DRAW_VERTEX_BUFFER &&
        (hasBoundVertexBuffer() || hasBoundVertexArray()));

  if (numItemsToDraw < 0) {
    numItemsToDraw = indirectBuffer->numItems();
  }

  MAPD_CHECK_GL_ERROR(glMultiDrawArraysIndirect(
      primitiveMode, BUFFER_OFFSET(startIndex * indirectBuffer->getStructByteSize()), numItemsToDraw, 0));
}

void GLRenderer::drawIndexBuffers(GLenum primitiveMode, size_t startIndex, int numItemsToDraw) {
  GLIndexBufferShPtr ibo = getBoundIndexBuffer();
  CHECK(ibo != nullptr);

  if (numItemsToDraw < 0) {
    numItemsToDraw = ibo->numItems();
  }

  MAPD_CHECK_GL_ERROR(glDrawElements(primitiveMode,
                                     numItemsToDraw,
                                     static_cast<GLenum>(ibo->getIndexType()),
                                     BUFFER_OFFSET(startIndex * ibo->getIndexTypeByteSize())));
}

void GLRenderer::drawIndirectIndexBuffers(GLenum primitiveMode, size_t startIndex, int numItemsToDraw) {
  Resources::GLIndirectDrawIndexBufferShPtr indirectBuffer = getBoundIndirectDrawIndexBuffer();
  GLIndexBufferShPtr ibo = getBoundIndexBuffer();
  CHECK(indirectBuffer != nullptr && indirectBuffer->getResourceType() == GLResourceType::INDIRECT_DRAW_INDEX_BUFFER &&
        ibo != nullptr);

  if (numItemsToDraw < 0) {
    numItemsToDraw = indirectBuffer->numItems();
  }

  MAPD_CHECK_GL_ERROR(glMultiDrawElementsIndirect(primitiveMode,
                                                  static_cast<GLenum>(ibo->getIndexType()),
                                                  BUFFER_OFFSET(startIndex * indirectBuffer->getStructByteSize()),
                                                  numItemsToDraw,
                                                  0));
}

void GLRenderer::getReadFramebufferPixels(GLenum readBuffer,
                                          size_t startx,
                                          size_t starty,
                                          size_t width,
                                          size_t height,
                                          GLenum format,
                                          GLenum type,
                                          GLvoid* data) {
  // TODO(croot): validate that this renderer is active on the current thread
  // Such a test could be made into a macro and is only activated in debug
  // mode which might be a reasonable option to keep such checks from
  // bogging the pipe down.
  // validateActiveRenderer();

  GLint currReadBuffer;
  MAPD_CHECK_GL_ERROR(glGetIntegerv(GL_READ_BUFFER, &currReadBuffer));
  MAPD_CHECK_GL_ERROR(glReadBuffer(readBuffer));
  MAPD_CHECK_GL_ERROR(glReadPixels(startx, starty, width, height, format, type, data));

  if (currReadBuffer != static_cast<GLint>(readBuffer)) {
    MAPD_CHECK_GL_ERROR(glReadBuffer(currReadBuffer));
  }
}

void GLRenderer::getBoundTexture2dPixels(size_t width, size_t height, GLenum format, GLenum type, GLvoid* data) {
  GLTexture2dShPtr boundTex = _bindState.getBoundTexture2d();
  RUNTIME_EX_ASSERT(boundTex != nullptr,
                    "Cannot get pixels of bound 2D texture. There isn't a 2D texture currently bound.");

  GLenum target = boundTex->getTarget();
  RUNTIME_EX_ASSERT(target == GL_TEXTURE_2D,
                    "Cannot get pixels of bound 2D texture. It is a multi-sample texture. Can only get pixels of a  "
                    "multi-sampled texture when attached to an FBO");

  size_t texWidth = boundTex->getWidth();
  size_t texHeight = boundTex->getHeight();
  RUNTIME_EX_ASSERT(width == texWidth && height == texHeight,
                    "Cannot get pixels of bound 2D texture. Resolution of output data does not match the resolution of "
                    "the bound texture: (" +
                        std::to_string(width) + "x" + std::to_string(height) + " != " + std::to_string(texWidth) + "x" +
                        std::to_string(texHeight) + ")");

  // TODO(croot): expose mipmap level

  // TODO(croot): check the format & type passed as arguments vs format/type of the texture?
  // I think a check like that could be useful, but would that limit usage somehow?
  MAPD_CHECK_GL_ERROR(glGetTexImage(target, 0, format, type, data));
}

void GLRenderer::getBoundTexture2dArrayPixels(size_t width,
                                              size_t height,
                                              size_t depth,
                                              GLenum format,
                                              GLenum type,
                                              GLvoid* data) {
  GLTexture2dArrayShPtr boundTexArray = _bindState.getBoundTexture2dArray();
  RUNTIME_EX_ASSERT(boundTexArray != nullptr,
                    "Cannot get pixels from bound 2D texture array. There isn't a 2D texture array currently bound.");

  GLenum target = boundTexArray->getTarget();
  RUNTIME_EX_ASSERT(target == GL_TEXTURE_2D_ARRAY,
                    "Cannot get pixels of bound 2D texture array. It is multi-sampled. Can only get pixels of a  "
                    "non multi-sampled texture array.");

  size_t texWidth = boundTexArray->getWidth();
  size_t texHeight = boundTexArray->getHeight();
  size_t texDepth = boundTexArray->getDepth();
  RUNTIME_EX_ASSERT(
      width == texWidth && height == texHeight && depth == texDepth,
      "Cannot get pixels of bound 2D texture array. Resolution of output data does not match the resolution of "
      "the bound texture: (" +
          std::to_string(width) + "x" + std::to_string(height) + "x" + std::to_string(depth) + " != " +
          std::to_string(texWidth) + "x" + std::to_string(texHeight) + "x" + std::to_string(texDepth) + ")");

  // TODO(croot): expose mipmap level

  // TODO(croot): check the format & type passed as arguments vs format/type of the texture?
  // I think a check like that could be useful, but would that limit usage somehow?
  MAPD_CHECK_GL_ERROR(glGetTexImage(target, 0, format, type, data));
}

void GLRenderer::_initGLEW(GLWindow* primaryWindow) {
  if (_glewInitialized) {
    return;
  }

  makeActiveOnCurrentThread(primaryWindow);

  glewExperimental = GL_TRUE;  // needed for core profile
  GLenum err = glewInit();
  if (err != 0) {
    char errstr[512];
    snprintf(errstr, sizeof(errstr), "%s", glewGetErrorString(err));
    THROW_RUNTIME_EX("GLEW error: Couldn\'t initialize glew. " + std::string(errstr));
  }

  err = glGetError();  // clear error code? This is due to a deprecated GL function call during glewInit()

  _glewInitialized = true;
}

void GLRenderer::_cleanupResources() {
  // need to make ourselves current to cleanup resources
  GLRenderer* currRenderer = nullptr;
  Window* currWindow = nullptr;
  if (!isActiveOnCurrentThread()) {
    currRenderer = getCurrentThreadRenderer();
    currWindow = getCurrentThreadWindow();
  }

  makeActiveOnCurrentThread();
  _glRsrcMgrPtr->_cleanupResources();

  if (currRenderer) {
    currRenderer->makeActiveOnCurrentThread(currWindow);
  } else {
    makeInactive();
  }
}

/**
 * TEMPLATE SPECIALIZATIONS -- need to enclose this in a "namespace {}" block
 * for certain compilers
 */

template <>
bool GLRenderer::getAttr(GLenum attrName) {
  GLboolean rtn;
  MAPD_CHECK_GL_ERROR(glGetBooleanv(attrName, &rtn));
  return rtn;
}

template <>
int GLRenderer::getAttr(GLenum attrName) {
  int rtn;
  MAPD_CHECK_GL_ERROR(glGetIntegerv(attrName, &rtn));
  return rtn;
}

template <>
float GLRenderer::getAttr(GLenum attrName) {
  float rtn;
  MAPD_CHECK_GL_ERROR(glGetFloatv(attrName, &rtn));
  return rtn;
}

template <>
double GLRenderer::getAttr(GLenum attrName) {
  double rtn;
  MAPD_CHECK_GL_ERROR(glGetDoublev(attrName, &rtn));
  return rtn;
}

template <>
int64_t GLRenderer::getAttr(GLenum attrName) {
  int64_t rtn;
  MAPD_CHECK_GL_ERROR(glGetInteger64v(attrName, &rtn));
  return rtn;
}

template <>
std::string GLRenderer::getAttr(GLenum attrName) {
  std::string rtn = reinterpret_cast<const char*>(glGetString(attrName));
  return rtn;
}

template <>
std::vector<bool> GLRenderer::getAttrv(GLenum attrName) {
  std::vector<GLboolean> data(1);
  MAPD_CHECK_GL_ERROR(glGetBooleanv(attrName, &data[0]));
  std::vector<bool> rtn(data.size());
  for (size_t i = 0; i < data.size(); ++i) {
    rtn[i] = static_cast<bool>(data[i]);
  }
  return rtn;
}

template <>
std::vector<int> GLRenderer::getAttrv(GLenum attrName) {
  std::vector<int> rtn(1);
  MAPD_CHECK_GL_ERROR(glGetIntegerv(attrName, &rtn[0]));
  return rtn;
}

template <>
std::vector<float> GLRenderer::getAttrv(GLenum attrName) {
  std::vector<float> rtn(1);
  MAPD_CHECK_GL_ERROR(glGetFloatv(attrName, &rtn[0]));
  return rtn;
}

template <>
std::vector<double> GLRenderer::getAttrv(GLenum attrName) {
  std::vector<double> rtn(1);
  MAPD_CHECK_GL_ERROR(glGetDoublev(attrName, &rtn[0]));
  return rtn;
}

template <>
std::vector<int64_t> GLRenderer::getAttrv(GLenum attrName) {
  std::vector<int64_t> rtn(1);
  MAPD_CHECK_GL_ERROR(glGetInteger64v(attrName, &rtn[0]));
  return rtn;
}

}  // namespace GL
}  // namespace Rendering

#include "GLRenderer.h"
#include "GLResourceManager.h"
#include "../../RenderError.h"
#include "../../Objects/ColorRGBA.h"
#include "Resources/GLFramebuffer.h"
#include "GLWindow.h"
#include <thread>
#include <mutex>
#include <iostream>

namespace Rendering {
namespace GL {

using Objects::ColorRGBA;
using Resources::GLResourceShPtr;
using Resources::FboBind;
using Resources::GLFramebufferShPtr;
using Resources::GLShaderShPtr;
using Resources::GLVertexArrayShPtr;
using Resources::GLFramebufferShPtr;

thread_local GLRenderer* _currentRenderer;

const GLRenderer* GLRenderer::getCurrentThreadRenderer() {
  return _currentRenderer;
}

// GLRenderer::GLRenderer(const WindowShPtr& parentWindowPtr) : Renderer(parentWindowPtr), _glewInitialized(false) {
GLRenderer::GLRenderer(const RendererSettings& settings)
    : Renderer(settings), _glRsrcMgrPtr(nullptr), _glewInitialized(false), _bindState(this) {
#ifdef GLEW_MX
  memset(&_glewContext, 0, sizeof(GLEWContext));
#endif
}

GLRenderer::~GLRenderer() {
  std::cerr << "CROOT GLRenderer destructor" << std::endl;
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

void GLRenderer::makeActiveOnCurrentThread(const Window* window) {
  // threadLocalStorage.set(this);
  _currentRenderer = this;
}

void GLRenderer::makeInactive() {
  // threadLocalStorage.set(nullptr);
  _currentRenderer = nullptr;
}

bool GLRenderer::isActiveOnCurrentThread() {
  // return threadLocalStorage.isCurrent(this);
  return _currentRenderer == this;
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
  MAPD_CHECK_GL_ERROR(glViewport(x, y, width, height));
}

void GLRenderer::setViewport(const Viewport& viewport) {
  MAPD_CHECK_GL_ERROR(glViewport(viewport.getXPos(), viewport.getYPos(), viewport.getWidth(), viewport.getHeight()));
}

void GLRenderer::enable(GLenum attr) {
  MAPD_CHECK_GL_ERROR(glEnable(attr));
}

void GLRenderer::disable(GLenum attr) {
  MAPD_CHECK_GL_ERROR(glDisable(attr));
}

void GLRenderer::setBlendFunc(GLenum srcFactor, GLenum dstFactor) {
  MAPD_CHECK_GL_ERROR(glBlendFunc(srcFactor, dstFactor));
}

void GLRenderer::drawVertexBuffers(GLenum primitiveMode, int startIndex, int numItemsToDraw) {
  MAPD_CHECK_GL_ERROR(glDrawArrays(primitiveMode, startIndex, numItemsToDraw));
}

void GLRenderer::bindResource(const GLResourceShPtr& rsrc) {
  _bindState.bindResource(rsrc);
}

void GLRenderer::bindFramebuffer(FboBind bindType, const GLFramebufferShPtr& fboRsrc) {
  _bindState.bindFramebuffer(bindType, fboRsrc);
}

void GLRenderer::bindShader(const GLShaderShPtr& shaderRsrc) {
  _bindState.bindShader(shaderRsrc);
}

void GLRenderer::bindVertexArray(const GLVertexArrayShPtr& vaoRsrc) {
  _bindState.bindVertexArray(vaoRsrc);
}

GLFramebufferShPtr GLRenderer::getBoundFbo(FboBind bindType) const {
  return _bindState.getBoundFbo(bindType);
}

bool GLRenderer::hasBoundFbo(Resources::FboBind bindType) const {
  return _bindState.hasBoundFbo(bindType);
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

void GLRenderer::_initGLEW(const GLWindow* primaryWindow) {
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
  _glRsrcMgrPtr->_cleanupResources();
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

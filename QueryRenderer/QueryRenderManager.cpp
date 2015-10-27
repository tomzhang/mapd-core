#include "QueryRenderManager.h"
#include "QueryFramebuffer.h"
#include <glog/logging.h>
#include <time.h>
#include <iostream>
#include <stdexcept>
#include <fstream>
#include <sstream>
#include <map>
#include "rapidjson/document.h"

// using namespace MapD_Renderer;
using ::MapD_Renderer::QueryRenderManager;
using ::MapD_Renderer::QueryRenderer;
using ::MapD_Renderer::PngData;

void glfwErrorCallback(int error, const char* errstr) {
  // TODO(croot): should we throw an exception?
  // NOTE: There are cases when an error is caught here, but
  // is not fatal -- i.e. putting GLFW in headless mode.
  LOG(ERROR) << "GLFW error: 0x" << std::hex << error << ": " << errstr << std::endl;
}

// QueryRenderManager::QueryRenderManager(int queryResultBufferSize, bool debugMode) : _debugMode(debugMode),
// _activeRenderer(nullptr), _windowPtr(nullptr, glfwDestroyWindow), _queryResultVBOPtr(nullptr) {
QueryRenderManager::QueryRenderManager(unsigned int queryResultBufferSize, bool debugMode)
    : _debugMode(debugMode),
      _activeRenderer(nullptr),
      _windowPtr(nullptr),
      _queryResultVBOPtr(new QueryResultVertexBuffer(queryResultBufferSize)),
      _queryResultBufferSize(queryResultBufferSize) {
  _initGLFW();
  _initQueryResultBuffer();
}

QueryRenderManager::~QueryRenderManager() {
  glfwTerminate();
}

void QueryRenderManager::_initGLFW() {
  // set the error callback
  glfwSetErrorCallback(glfwErrorCallback);

  if (!glfwInit()) {
    std::runtime_error err("GLFW error: Couldn't initialize GLFW");
    LOG(ERROR) << err.what();
    throw err;
  }

  glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
  glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 1);
  glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
  glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);

  // _windowPtr.reset(glfwCreateWindow(1, 1, "", NULL, NULL));
  _windowPtr = glfwCreateWindow(1, 1, "", NULL, NULL);
  if (_windowPtr == nullptr) {
    // TODO(croot): is this necessary? Will the error callback catch
    // all possible errors?
    std::runtime_error err("GLFW error: Couldn\'t create a window.");
    LOG(ERROR) << err.what();
    throw err;
  }

  glfwMakeContextCurrent(_windowPtr);

  glewExperimental = GL_TRUE;  // needed for core profile
  GLenum err = glewInit();
  if (err != 0) {
    char errstr[512];
    snprintf(errstr, sizeof(errstr), "%s", glewGetErrorString(err));
    std::runtime_error err(std::string("GLEW error: Couldn\'t initialize glew. ") + errstr);
    LOG(ERROR) << err.what();
    throw err;
  }

  // glGetError();  // clear error code - this always throws error but seems to not matter

  // indicates how many frames to wait until buffers are swapped.
  glfwSwapInterval(1);
}

void QueryRenderManager::_initQueryResultBuffer() {
  // TODO(croot): Using GL_PIXEL_UNPACK_BUFFER for the target in the following
  // constructor causes the framebuffer to fail with an incomplete attachment error.
  // The following opengl wiki: https://www.opengl.org/wiki/GLAPI/glBindBuffer
  // states that: "While a non-zero buffer object is bound to the GL_PIXEL_UNPACK_BUFFER target, the following commands
  // are affected: glCompressedTexImage1D​, glCompressedTexImage2D​, glCompressedTexImage3D​,
  // glCompressedTexSubImage1D​, glCompressedTexSubImage2D​, glCompressedTexSubImage3D​, glTexImage1D​,
  // glTexImage2D​, glTexImage3D​, glTexSubImage1D​, glTexSubImage2D​, and glTexSubImage3D​. The pointer
  // parameter is interpreted as an offset within the buffer object measured in basic machine units."
  //
  // So if GL_PIXEL_UNPACK_BUFFER is needed for CUDA interoperability, this could be a problem.
  // _queryResultVBOPtr.reset(new VertexBuffer(_queryResultBufferSize, GL_ARRAY_BUFFER, GL_DYNAMIC_COPY));
  // _queryResultVBOPtr.reset(new VertexBuffer(_queryResultBufferSize, GL_PIXEL_UNPACK_BUFFER, GL_DYNAMIC_COPY));
}

void QueryRenderManager::_setActiveUserWidget(int userId, int widgetId) const {
  UserWidgetPair userWidget = std::make_pair(userId, widgetId);

  if (userWidget != _activeUserWidget) {
    auto userIter = _rendererDict.find(userId);

    if (userIter == _rendererDict.end()) {
      throw std::runtime_error("User id: " + std::to_string(userId) + " does not exist.");
    }

    WidgetRendererMap* wfMap = userIter->second.get();

    auto widgetIter = wfMap->find(widgetId);

    if (widgetIter == wfMap->end()) {
      throw std::runtime_error("Widget id: " + std::to_string(widgetId) + " for user id: " + std::to_string(userId) +
                               " does not exist.");
    }

    _activeRenderer = widgetIter->second.get();
    _activeUserWidget = userWidget;
  }
}

QueryRenderer* QueryRenderManager::_getRendererForUserWidget(int userId, int widgetId) const {
  _setActiveUserWidget(userId, widgetId);
  return _activeRenderer;
}

bool QueryRenderManager::inDebugMode() const {
  return (_debugMode == true);
}

bool QueryRenderManager::hasUser(int userId) const {
  return (_rendererDict.find(userId) != _rendererDict.end());
}

bool QueryRenderManager::hasUserWidget(int userId, int widgetId) const {
  auto userIter = _rendererDict.find(userId);

  if (userIter == _rendererDict.end()) {
    return false;
  }

  return (userIter->second->find(widgetId) == userIter->second->end());
}

bool QueryRenderManager::hasUserWidget(const UserWidgetPair& userWidgetPair) const {
  return hasUserWidget(userWidgetPair.first, userWidgetPair.second);
}

void QueryRenderManager::addUserWidget(int userId,
                                       int widgetId,
                                       const std::string& configJSON,
                                       bool doHitTest,
                                       bool doDepthTest) {
  WidgetRendererMap* wfMap;

  auto userIter = _rendererDict.find(userId);

  if (userIter == _rendererDict.end()) {
    wfMap = new WidgetRendererMap();
    _rendererDict[userId] = std::unique_ptr<WidgetRendererMap>(wfMap);
  } else {
    wfMap = userIter->second.get();

    if (wfMap->find(widgetId) != wfMap->end()) {
      // a framebuffer already exists! Throw an error.
      throw std::runtime_error("User id: " + std::to_string(userId) + " with widget id: " + std::to_string(widgetId) +
                               " already exists.");
    }
  }

  // (*wfMap)[widgetId] = QueryRendererUqPtr(new QueryRenderer(configJSON, doHitTest, doDepthTest, (_debugMode ?
  // _windowPtr.get() : nullptr)));
  (*wfMap)[widgetId] = QueryRendererUqPtr(
      new QueryRenderer(configJSON, _queryResultVBOPtr, doHitTest, doDepthTest, (_debugMode ? _windowPtr : nullptr)));

  _setActiveUserWidget(userId, widgetId);
}

void QueryRenderManager::addUserWidget(const UserWidgetPair& userWidgetPair,
                                       const std::string& configJSON,
                                       bool doHitTest,
                                       bool doDepthTest) {
  addUserWidget(userWidgetPair.first, userWidgetPair.second, configJSON, doHitTest, doDepthTest);
}

void QueryRenderManager::setJSONConfigForUserWidget(int userId, int widgetId, const std::string& configJSON) {
  QueryRenderer* renderer = _getRendererForUserWidget(userId, widgetId);
  // renderer->setJSONConfig(configJSON, (_debugMode ? _windowPtr.get() : nullptr));
  renderer->setJSONConfig(configJSON, (_debugMode ? _windowPtr : nullptr));
}

void QueryRenderManager::setJSONConfigForUserWidget(const UserWidgetPair& userWidgetPair,
                                                    const std::string& configJSON) {
  setJSONConfigForUserWidget(userWidgetPair.first, userWidgetPair.second, configJSON);
}

// int QueryRenderManager::getActiveUserId() const {
//     if (_activeRenderer) {
//         return _activeFramebufferIds.first;
//     }

//     return -1;
// }

// int QueryRenderManager::getActiveWidgetId() const {
//     if (_activeRenderer) {
//         return _activeFramebufferIds.second;
//     }

//     return -1;
// }

void QueryRenderManager::setUserWidgetWidthHeight(int userId, int widgetId, int width, int height) {
  QueryRenderer* renderer = _getRendererForUserWidget(userId, widgetId);
  // renderer->setWidthHeight(width, height, (_debugMode ? _windowPtr.get() : nullptr));
  renderer->setWidthHeight(width, height, (_debugMode ? _windowPtr : nullptr));
}

void QueryRenderManager::setUserWidgetWidthHeight(const UserWidgetPair& userWidgetPair, int width, int height) {
  setUserWidgetWidthHeight(userWidgetPair.first, userWidgetPair.second, width, height);
}

void QueryRenderManager::renderUserWidget(int userId, int widgetId) const {
  QueryRenderer* renderer = _getRendererForUserWidget(userId, widgetId);
  renderer->render();

  if (_debugMode) {
    glBindFramebuffer(GL_DRAW_FRAMEBUFFER, 0);
    glDrawBuffer(GL_BACK);

    // TODO(croot): need an API to set the framebuffer's read buffer
    renderer->getFramebuffer()->bindToRenderer(BindType::READ);
    glReadBuffer(GL_COLOR_ATTACHMENT0);

    int framebufferWidth, framebufferHeight;
    // glfwGetFramebufferSize(_windowPtr.get(), &framebufferWidth, &framebufferHeight);
    glfwGetFramebufferSize(_windowPtr, &framebufferWidth, &framebufferHeight);

    glBlitFramebuffer(0,
                      0,
                      framebufferWidth,
                      framebufferHeight,
                      0,
                      0,
                      framebufferWidth,
                      framebufferHeight,
                      GL_COLOR_BUFFER_BIT,
                      GL_NEAREST);
  }
}

void QueryRenderManager::renderUserWidget(const UserWidgetPair& userWidgetPair) const {
  renderUserWidget(userWidgetPair.first, userWidgetPair.second);
}

PngData QueryRenderManager::renderToPng(int userId, int widgetId) const {
  QueryRenderer* renderer = _getRendererForUserWidget(userId, widgetId);
  renderer->render();

  // unsigned char* pixels = new unsigned char[width * height * 4];
  int width = renderer->getWidth();
  int height = renderer->getHeight();
  int r, g, b, a;

  gdImagePtr im = gdImageCreateTrueColor(width, height);

  unsigned char* pixels = new unsigned char[width * height * 4];

  // TODO(croot): Make an improved read-pixels API for framebuffers
  // renderer->getFramebuffer()->bindToRenderer(BindType::READ);
  renderer->getFramebuffer()->bindToRenderer(BindType::READ);
  glReadBuffer(GL_COLOR_ATTACHMENT0);
  glReadPixels(0, 0, width, height, GL_RGBA, GL_UNSIGNED_BYTE, pixels);

  int idx = 0;
  for (int j = 0; j < height; ++j) {
    for (int i = 0; i < width; ++i) {
      r = pixels[idx++];
      g = pixels[idx++];
      b = pixels[idx++];
      a = 127 -
          (pixels[idx++] /
           2);  // need to convert the alpha into a gd-compliant alpha (0 [fully opague] - 127 [fully transparent])

      gdImageSetPixel(im, i, height - j - 1, gdTrueColorAlpha(r, g, b, a));
    }
  }

  int pngSize;
  char* pngPtr = reinterpret_cast<char*>(gdImagePngPtr(im, &pngSize));

  gdImageDestroy(im);
  delete[] pixels;

  return PngData(pngPtr, pngSize);
}

PngData QueryRenderManager::renderToPng(const UserWidgetPair& userWidgetPair) const {
  return renderToPng(userWidgetPair.first, userWidgetPair.second);
}

unsigned int QueryRenderManager::getIdAt(int userId, int widgetId, int x, int y) const {
  QueryRenderer* renderer = _getRendererForUserWidget(userId, widgetId);
  return renderer->getIdAt(x, y);
}

unsigned int QueryRenderManager::getIdAt(const UserWidgetPair& userWidgetPair, int x, int y) const {
  return getIdAt(userWidgetPair.first, userWidgetPair.second, x, y);
}

int randColor() {
  return rand() % 256;
}

int randAlpha() {
  return rand() % 128;
}

PngData QueryRenderManager::getColorNoisePNG(int width, int height) {
  srand(time(NULL));

  // unsigned char* pixels = new unsigned char[width * height * 4];
  int r, g, b, a;

  gdImagePtr im = gdImageCreateTrueColor(width, height);

  std::map<unsigned int, int> colorMap;

  for (int i = 0; i < width; ++i) {
    for (int j = 0; j < height; ++j) {
      r = randColor();
      g = randColor();
      b = randColor();
      a = 0;

      unsigned int colorId = 16777216 * a + 65536 * b + 256 * g + r;

      if (colorMap.find(colorId) == colorMap.end()) {
        colorMap[colorId] = gdImageColorAllocateAlpha(im, r, g, b, a);
      }

      gdImageSetPixel(im, i, j, colorMap[colorId]);
    }
  }

  int pngSize;
  char* pngPtr = (char*)gdImagePngPtr(im, &pngSize);

  gdImageDestroy(im);

  return PngData(pngPtr, pngSize);
}

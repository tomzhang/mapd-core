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
using ::MapD_Renderer::CudaHandle;
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
QueryRenderManager::QueryRenderManager(unsigned int queryResultBufferSize, GLFWwindow* prntWindow, bool debugMode)
    : _debugMode(debugMode),
      _activeRenderer(nullptr),
      _windowPtr(nullptr),
      _queryResultVBOPtr(new QueryResultVertexBuffer(queryResultBufferSize)),
      _queryResultBufferSize(queryResultBufferSize) {
  _initGLFW(prntWindow);
  _initQueryResultBuffer();
}

QueryRenderManager::~QueryRenderManager() {
  // TODO(croot): don't use glfwTerminate() -- use whatever the window deletion function is
  // glfwTerminate();
  glfwDestroyWindow(_windowPtr);
}

void QueryRenderManager::_initGLFW(GLFWwindow* prntWindow) {
  // set the error callback
  glfwSetErrorCallback(glfwErrorCallback);

  if (!prntWindow) {
    // need to init glfw -- this needs to happen in the main thread
    if (!glfwInit()) {
      std::runtime_error err("GLFW error: Couldn't initialize GLFW");
      LOG(ERROR) << err.what();
      throw err;
    }
  }

  glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
  glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 1);
  glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
  glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);

  // _windowPtr.reset(glfwCreateWindow(1, 1, "", NULL, NULL));
  _windowPtr = glfwCreateWindow(1, 1, "", NULL, prntWindow);
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

CudaHandle QueryRenderManager::getCudaHandle() {
  if (!_queryResultVBOPtr) {
    LOG(FATAL) << "The query render manager is in a corrupt state.";
  }

  return _queryResultVBOPtr->getCudaHandlePreQuery();
}

void QueryRenderManager::setActiveUserWidget(int userId, int widgetId) {
  UserWidgetPair userWidget = std::make_pair(userId, widgetId);

  if (userWidget != _activeUserWidget) {
    auto userIter = _rendererDict.find(userId);

    if (userIter == _rendererDict.end()) {
      std::runtime_error err("User id: " + std::to_string(userId) + " does not exist.");
      LOG(ERROR) << err.what();
      throw err;
    }

    WidgetRendererMap* wfMap = userIter->second.get();

    auto widgetIter = wfMap->find(widgetId);

    if (widgetIter == wfMap->end()) {
      std::runtime_error err("Widget id: " + std::to_string(widgetId) + " for user id: " + std::to_string(userId) +
                             " does not exist.");
      LOG(ERROR) << err.what();
      throw err;
    }

    _activeRenderer = widgetIter->second.get();
    _activeUserWidget = userWidget;
  }
}

void QueryRenderManager::setActiveUserWidget(const UserWidgetPair& userWidgetPair) {
  setActiveUserWidget(userWidgetPair.first, userWidgetPair.second);
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

void QueryRenderManager::addUserWidget(int userId, int widgetId, bool doHitTest, bool doDepthTest) {
  WidgetRendererMap* wfMap;

  auto userIter = _rendererDict.find(userId);

  if (userIter == _rendererDict.end()) {
    wfMap = new WidgetRendererMap();
    _rendererDict[userId] = std::unique_ptr<WidgetRendererMap>(wfMap);
  } else {
    wfMap = userIter->second.get();

    if (wfMap->find(widgetId) != wfMap->end()) {
      // a framebuffer already exists! Throw an error.
      std::runtime_error err("User id: " + std::to_string(userId) + " with widget id: " + std::to_string(widgetId) +
                             " already exists.");
      LOG(ERROR) << err.what();
      throw err;
    }
  }

  // (*wfMap)[widgetId] = QueryRendererUqPtr(new QueryRenderer(configJSON, doHitTest, doDepthTest, (_debugMode ?
  // _windowPtr.get() : nullptr)));
  (*wfMap)[widgetId] = QueryRendererUqPtr(
      new QueryRenderer(_queryResultVBOPtr, doHitTest, doDepthTest, (_debugMode ? _windowPtr : nullptr)));

  // TODO: should we set this as active the newly added ids as active?
  // setActiveUserWidget(userId, widgetId);
}

void QueryRenderManager::addUserWidget(const UserWidgetPair& userWidgetPair, bool doHitTest, bool doDepthTest) {
  addUserWidget(userWidgetPair.first, userWidgetPair.second, doHitTest, doDepthTest);
}

void QueryRenderManager::configureRender(const rapidjson::Document& jsonDocument, QueryDataLayout* dataLayoutPtr) {
  if (!_activeRenderer) {
    std::runtime_error err(
        "There is no active user/widget id. Must set a user/widget id active before configuring the render.");
    LOG(ERROR) << err.what();
    throw err;
  }

  // need to update the data layout of the query result buffer before building up
  // from the json obj
  if (dataLayoutPtr) {
    _activeRenderer->updateQueryResultBufferPostQuery(
        dataLayoutPtr->convertToBufferLayout(), dataLayoutPtr->numRows, dataLayoutPtr->invalidKey);
  }

  _activeRenderer->setJSONDocument(jsonDocument, (_debugMode ? _windowPtr : nullptr));
}

// void QueryRenderManager::setJSONConfigForUserWidget(int userId, int widgetId, const std::string& configJSON) {
//   QueryRenderer* renderer = _getRendererForUserWidget(userId, widgetId);
//   // renderer->setJSONConfig(configJSON, (_debugMode ? _windowPtr.get() : nullptr));
//   renderer->setJSONConfig(configJSON, (_debugMode ? _windowPtr : nullptr));
// }

// void QueryRenderManager::setJSONConfigForUserWidget(const UserWidgetPair& userWidgetPair,
//                                                     const std::string& configJSON) {
//   setJSONConfigForUserWidget(userWidgetPair.first, userWidgetPair.second, configJSON);
// }

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

// void QueryRenderManager::setUserWidgetWidthHeight(int userId, int widgetId, int width, int height) {
//   QueryRenderer* renderer = _getRendererForUserWidget(userId, widgetId);
//   // renderer->setWidthHeight(width, height, (_debugMode ? _windowPtr.get() : nullptr));
//   renderer->setWidthHeight(width, height, (_debugMode ? _windowPtr : nullptr));
// }

// void QueryRenderManager::setUserWidgetWidthHeight(const UserWidgetPair& userWidgetPair, int width, int height) {
//   setUserWidgetWidthHeight(userWidgetPair.first, userWidgetPair.second, width, height);
// }

void QueryRenderManager::setWidthHeight(int width, int height) {
  if (!_activeRenderer) {
    std::runtime_error err(
        "There is no active user/widget id. Must set a user/widget id active before setting width/height.");
    LOG(ERROR) << err.what();
    throw err;
  }
  _activeRenderer->setWidthHeight(width, height, (_debugMode ? _windowPtr : nullptr));
}

void QueryRenderManager::render() const {
  if (!_activeRenderer) {
    std::runtime_error err("There is no active user/widget id. Must set a user/widget id active before rendering.");
    LOG(ERROR) << err.what();
    throw err;
  }

  _activeRenderer->render();

  if (_debugMode) {
    glBindFramebuffer(GL_DRAW_FRAMEBUFFER, 0);
    glDrawBuffer(GL_BACK);

    // TODO(croot): need an API to set the framebuffer's read buffer
    _activeRenderer->getFramebuffer()->bindToRenderer(BindType::READ);
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

PngData QueryRenderManager::renderToPng() const {
  if (!_activeRenderer) {
    std::runtime_error err("There is no active user/widget id. Must set a user/widget id active before rendering.");
    LOG(ERROR) << err.what();
    throw err;
  }

  _activeRenderer->render();

  // unsigned char* pixels = new unsigned char[width * height * 4];
  int width = _activeRenderer->getWidth();
  int height = _activeRenderer->getHeight();
  int r, g, b, a;

  gdImagePtr im = gdImageCreateTrueColor(width, height);
  // the above gdImageCreateTrueColor() constructs a default image
  // with an black opaque background color. This is the only way
  // i've found to make a fully transparent background --- extracting
  // the black background color index and setting it fully transparent
  // with the gdImageColorTransparent() call.
  int black = gdImageColorExact(im, 0, 0, 0);
  gdImageColorTransparent(im, black);

  unsigned char* pixels = new unsigned char[width * height * 4];

  // TODO(croot): Make an improved read-pixels API for framebuffers
  // _activeRenderer->getFramebuffer()->bindToRenderer(BindType::READ);
  _activeRenderer->getFramebuffer()->bindToRenderer(BindType::READ);
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

unsigned int QueryRenderManager::getIdAt(int x, int y) const {
  if (!_activeRenderer) {
    std::runtime_error err(
        "There is no active user/widget id. Must set a user/widget id active before requesting pixel data.");
    LOG(ERROR) << err.what();
    throw err;
  }
  return _activeRenderer->getIdAt(x, y);
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
  int black = gdImageColorExact(im, 0, 0, 0);
  gdImageColorTransparent(im, black);

  for (int i = 0; i < width; ++i) {
    for (int j = 0; j < height; ++j) {
      r = randColor();
      g = randColor();
      b = randColor();
      a = 0;

      gdImageSetPixel(im, i, j, gdTrueColorAlpha(r, g, b, a));
    }
  }

  int pngSize;
  char* pngPtr = (char*)gdImagePngPtr(im, &pngSize);

  gdImageDestroy(im);

  return PngData(pngPtr, pngSize);
}

#include "QueryRenderManager.h"
#include "QueryFramebuffer.h"
#include "../QueryEngine/Execute.h"

#include <GL/glew.h>

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
QueryRenderManager::QueryRenderManager(const Executor* executor,
                                       unsigned int queryResultBufferSize,
                                       GLFWwindow* prntWindow,
                                       bool debugMode)
    : _debugMode(debugMode),
      _activeRenderer(nullptr),
      _windowPtr(nullptr),
      _queryResultVBOPtr(new QueryResultVertexBuffer(queryResultBufferSize)),
      _queryResultBufferSize(queryResultBufferSize),
      executor_(executor) {
  _initGLFW(prntWindow);
  _initQueryResultBuffer();
}

QueryRenderManager::~QueryRenderManager() {
  glfwDestroyWindow(_windowPtr);
}

void QueryRenderManager::_initGLFW(GLFWwindow* prntWindow) {
  // set the error callback
  glfwSetErrorCallback(glfwErrorCallback);

  if (!prntWindow) {
    // need to init glfw -- this needs to happen in the main thread
    if (!glfwInit()) {
      RUNTIME_EX_ASSERT(glfwInit(), "GLFW error: Couldn\'t initialize GLFW.");
    }
  }

  glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
  glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 1);
  glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
  glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);
  glfwWindowHint(GLFW_VISIBLE, _debugMode);

  // _windowPtr.reset(glfwCreateWindow(1, 1, "", NULL, NULL));
  _windowPtr = glfwCreateWindow(1, 1, "", NULL, prntWindow);
  RUNTIME_EX_ASSERT(_windowPtr != nullptr, "GLFW error: Couldn't create a window.");

  glfwMakeContextCurrent(_windowPtr);

  glewExperimental = GL_TRUE;  // needed for core profile
  GLenum err = glewInit();
  if (err != 0) {
    char errstr[512];
    snprintf(errstr, sizeof(errstr), "%s", glewGetErrorString(err));
    THROW_RUNTIME_EX("GLEW error: Couldn\'t initialize glew. " + std::string(errstr));
  }

  err = glGetError();  // clear error code? This is due to a deprecated GL function call during glewInit()

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

  std::lock_guard<std::mutex> render_lock(_mtx);
  glfwMakeContextCurrent(_windowPtr);
  CudaHandle rtn = _queryResultVBOPtr->getCudaHandlePreQuery();
  glfwMakeContextCurrent(nullptr);

  return rtn;
}

void QueryRenderManager::setActiveUserWidget(int userId, int widgetId) {
  // TODO(croot): should we put thread locks in here? that probably makes sense.

  UserWidgetPair userWidget = std::make_pair(userId, widgetId);

  if (userWidget != _activeUserWidget) {
    auto userIter = _rendererDict.find(userId);

    RUNTIME_EX_ASSERT(userIter != _rendererDict.end(), "User id: " + std::to_string(userId) + " does not exist.");

    WidgetRendererMap* wfMap = userIter->second.get();

    auto widgetIter = wfMap->find(widgetId);

    RUNTIME_EX_ASSERT(
        widgetIter != wfMap->end(),
        "Widget id: " + std::to_string(widgetId) + " for user id: " + std::to_string(userId) + " does not exist.");

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

    RUNTIME_EX_ASSERT(wfMap->find(widgetId) == wfMap->end(),
                      "Cannot add user widget. User id: " + std::to_string(userId) + " with widget id: " +
                          std::to_string(widgetId) + " already exists.");
  }

  (*wfMap)[widgetId] = QueryRendererUqPtr(
      new QueryRenderer(executor_, _queryResultVBOPtr, doHitTest, doDepthTest, (_debugMode ? _windowPtr : nullptr)));

  // TODO(croot): should we set this as active the newly added ids as active?
  // setActiveUserWidget(userId, widgetId);
}

void QueryRenderManager::addUserWidget(const UserWidgetPair& userWidgetPair, bool doHitTest, bool doDepthTest) {
  addUserWidget(userWidgetPair.first, userWidgetPair.second, doHitTest, doDepthTest);
}

void QueryRenderManager::configureRender(const rapidjson::Document& jsonDocument, QueryDataLayout* dataLayoutPtr) {
  RUNTIME_EX_ASSERT(_activeRenderer != nullptr,
                    "ConfigureRender: There is no active user/widget id. Must set a user/widget id active before "
                    "configuring the render.");

  std::lock_guard<std::mutex> render_lock(_mtx);
  glfwMakeContextCurrent(_windowPtr);

  // need to update the data layout of the query result buffer before building up
  // from the json obj
  if (dataLayoutPtr) {
    _activeRenderer->updateQueryResultBufferPostQuery(
        dataLayoutPtr->convertToBufferLayout(), dataLayoutPtr->numRows, dataLayoutPtr->invalidKey);
  }

  _activeRenderer->setJSONDocument(jsonDocument, (_debugMode ? _windowPtr : nullptr));
  glfwMakeContextCurrent(nullptr);
}

void QueryRenderManager::setWidthHeight(int width, int height) {
  RUNTIME_EX_ASSERT(_activeRenderer != nullptr,
                    "setWidthHeight: There is no active user/widget id. Must set an active user/widget id before "
                    "setting width/height.");

  std::lock_guard<std::mutex> render_lock(_mtx);
  glfwMakeContextCurrent(_windowPtr);
  _activeRenderer->setWidthHeight(width, height, (_debugMode ? _windowPtr : nullptr));
  glfwMakeContextCurrent(nullptr);
}

void QueryRenderManager::render() {
  RUNTIME_EX_ASSERT(_activeRenderer != nullptr,
                    "render(): There is no active user/widget id. Must set a user/widget id active before rendering.");

  std::lock_guard<std::mutex> render_lock(_mtx);
  glfwMakeContextCurrent(_windowPtr);
  _activeRenderer->render();
  glfwSwapBuffers(_windowPtr);

  if (_debugMode) {
    MAPD_CHECK_GL_ERROR(glBindFramebuffer(GL_DRAW_FRAMEBUFFER, 0));
    MAPD_CHECK_GL_ERROR(glDrawBuffer(GL_BACK));

    // TODO(croot): need an API to set the framebuffer's read buffer
    _activeRenderer->getFramebuffer()->bindToRenderer(BindType::READ);
    MAPD_CHECK_GL_ERROR(glReadBuffer(GL_COLOR_ATTACHMENT0));

    int framebufferWidth, framebufferHeight;
    // glfwGetFramebufferSize(_windowPtr.get(), &framebufferWidth, &framebufferHeight);
    MAPD_CHECK_GL_ERROR(glfwGetFramebufferSize(_windowPtr, &framebufferWidth, &framebufferHeight));

    MAPD_CHECK_GL_ERROR(glBlitFramebuffer(0,
                                          0,
                                          framebufferWidth,
                                          framebufferHeight,
                                          0,
                                          0,
                                          framebufferWidth,
                                          framebufferHeight,
                                          GL_COLOR_BUFFER_BIT,
                                          GL_NEAREST));
  }
  glfwMakeContextCurrent(nullptr);
}

PngData QueryRenderManager::renderToPng() {
  RUNTIME_EX_ASSERT(_activeRenderer != nullptr,
                    "There is no active user/widget id. Must set a user/widget id active before rendering.");

  std::lock_guard<std::mutex> render_lock(_mtx);
  glfwMakeContextCurrent(_windowPtr);
  _activeRenderer->render();
  glfwSwapBuffers(_windowPtr);

  int width = _activeRenderer->getWidth();
  int height = _activeRenderer->getHeight();
  int r, g, b, a;

  gdImagePtr im = gdImageCreateTrueColor(width, height);
  // the above gdImageCreateTrueColor() constructs a default image
  // with a fully opaque black background color. This is the only way
  // i've found to make a fully transparent background --- extracting
  // the black background color index and setting it fully transparent
  // with the gdImageColorTransparent() call.
  int black = gdImageColorExact(im, 0, 0, 0);
  gdImageColorTransparent(im, black);

  unsigned char* pixels = new unsigned char[width * height * 4];

  // TODO(croot): Make an improved read-pixels API for framebuffers
  // _activeRenderer->getFramebuffer()->bindToRenderer(BindType::READ);
  _activeRenderer->getFramebuffer()->bindToRenderer(BindType::READ);
  MAPD_CHECK_GL_ERROR(glReadBuffer(GL_COLOR_ATTACHMENT0));
  MAPD_CHECK_GL_ERROR(glReadPixels(0, 0, width, height, GL_RGBA, GL_UNSIGNED_BYTE, pixels));

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
  std::shared_ptr<char> pngPtr(reinterpret_cast<char*>(gdImagePngPtr(im, &pngSize)), gdFree);

  gdImageDestroy(im);
  delete[] pixels;

  glfwMakeContextCurrent(nullptr);

  return PngData(pngPtr, pngSize);
}

unsigned int QueryRenderManager::getIdAt(int x, int y) {
  RUNTIME_EX_ASSERT(
      _activeRenderer != nullptr,
      "getIdAt(): There is no active user/widget id. Must set an active user/widget id before requesting pixel data.");

  std::lock_guard<std::mutex> render_lock(_mtx);
  glfwMakeContextCurrent(_windowPtr);
  unsigned int id = _activeRenderer->getIdAt(x, y);
  glfwMakeContextCurrent(nullptr);
  return id;
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
  std::shared_ptr<char> pngPtr(reinterpret_cast<char*>(gdImagePngPtr(im, &pngSize)), gdFree);

  gdImageDestroy(im);

  return PngData(pngPtr, pngSize);
}

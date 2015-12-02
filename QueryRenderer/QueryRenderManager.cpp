#include "QueryRenderManager.h"
#include "QueryFramebuffer.h"

#include <GL/glew.h>
#include <png.h>

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

const ::MapD_Renderer::UserWidgetPair QueryRenderManager::_emptyUserWidget = std::make_pair(-1, -1);

// QueryRenderManager::QueryRenderManager(int queryResultBufferSize, bool debugMode) : _debugMode(debugMode),
// _activeRenderer(nullptr), _windowPtr(nullptr, glfwDestroyWindow), _queryResultVBOPtr(nullptr) {
QueryRenderManager::QueryRenderManager(const Executor* executor,
                                       unsigned int queryResultBufferSize,
                                       GLFWwindow* prntWindow,
                                       bool debugMode)
    : _debugMode(debugMode),
      _activeRenderer(nullptr),
      _activeUserWidget(_emptyUserWidget),
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

  if (!_debugMode) {
    glfwWindowHint(GLFW_VISIBLE, GL_FALSE);
  }

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

  // unset the gl context for the current thread
  // TODO(croot): create a context resource wrapper object to handle
  // these kind of issues. It should be non-copyable - moveable-only
  // and handle thread-switching appropriately.
  glfwMakeContextCurrent(nullptr);
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

  return (userIter->second->find(widgetId) != userIter->second->end());
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

void QueryRenderManager::removeUserWidget(int userId, int widgetId) {
  auto userIter = _rendererDict.find(userId);

  RUNTIME_EX_ASSERT(userIter != _rendererDict.end(),
                    "User id " + std::to_string(userId) + " does not exist. Cannot remove the caches for " +
                        std::to_string(userId) + ":" + std::to_string(widgetId) + ".");

  WidgetRendererMap* wfMap = userIter->second.get();
  auto widgetIter = wfMap->find(widgetId);

  RUNTIME_EX_ASSERT(widgetIter != wfMap->end(),
                    "Widget id " + std::to_string(widgetId) + " for user id " + std::to_string(userId) +
                        " does not exist. Cannot remove widget.");

  wfMap->erase(widgetIter);

  if (userId == _activeUserWidget.first && widgetId == _activeUserWidget.second) {
    _activeUserWidget = _emptyUserWidget;
    _activeRenderer = nullptr;
  }
}
void QueryRenderManager::removeUserWidget(const UserWidgetPair& userWidgetPair) {
  removeUserWidget(userWidgetPair.first, userWidgetPair.second);
}

// Removes all widgets/sessions for a particular user id.
void QueryRenderManager::removeUser(int userId) {
  auto userIter = _rendererDict.find(userId);

  RUNTIME_EX_ASSERT(userIter != _rendererDict.end(),
                    "User id " + std::to_string(userId) + " does not exist. Cannot remove its caches.");

  _rendererDict.erase(userIter);

  if (userId == _activeUserWidget.first) {
    _activeUserWidget = _emptyUserWidget;
    _activeRenderer = nullptr;
  }
}

void QueryRenderManager::configureRender(const std::shared_ptr<rapidjson::Document>& jsonDocumentPtr,
                                         QueryDataLayout* dataLayoutPtr) {
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

  _activeRenderer->setJSONDocument(jsonDocumentPtr, false, (_debugMode ? _windowPtr : nullptr));
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

void writePngData(png_structp png_ptr, png_bytep data, png_size_t length) {
  std::vector<char>* pngData = reinterpret_cast<std::vector<char>*>(png_get_io_ptr(png_ptr));
  size_t currSz = pngData->size();
  pngData->resize(currSz + length);
  std::memcpy(&(*pngData)[0] + currSz, data, length);
}

void flushPngData(png_structp png_ptr) {
  // Do nothing
  (void)png_ptr; /* Stifle compiler warning */
}

PngData QueryRenderManager::renderToPng(int compressionLevel) {
  RUNTIME_EX_ASSERT(_activeRenderer != nullptr,
                    "There is no active user/widget id. Must set a user/widget id active before rendering.");

  RUNTIME_EX_ASSERT(compressionLevel >= -1 && compressionLevel <= 9,
                    "Invalid compression level " + std::to_string(compressionLevel) + ". It must be a " +
                        "value between 0 (no zlib compression) to 9 (most zlib compression), or -1 (use default).");

  std::lock_guard<std::mutex> render_lock(_mtx);
  glfwMakeContextCurrent(_windowPtr);
  _activeRenderer->render();
  glfwSwapBuffers(_windowPtr);

  int width = _activeRenderer->getWidth();
  int height = _activeRenderer->getHeight();
  unsigned char* pixels = new unsigned char[width * height * 4];

  // TODO(croot): Make an improved read-pixels API for framebuffers
  // _activeRenderer->getFramebuffer()->bindToRenderer(BindType::READ);
  _activeRenderer->getFramebuffer()->bindToRenderer(BindType::READ);
  MAPD_CHECK_GL_ERROR(glReadBuffer(GL_COLOR_ATTACHMENT0));
  MAPD_CHECK_GL_ERROR(glReadPixels(0, 0, width, height, GL_RGBA, GL_UNSIGNED_BYTE, pixels));

  // Now build the png stream using libpng

  png_structp png_ptr = png_create_write_struct(PNG_LIBPNG_VER_STRING, nullptr, nullptr, nullptr);
  assert(png_ptr != nullptr);

  png_infop info_ptr = png_create_info_struct(png_ptr);
  assert(info_ptr != nullptr);

  // TODO(croot) - rather than going the setjmp route, you can enable the
  // PNG_SETJMP_NOT_SUPPORTED compiler flag which would result in asserts
  // when libpng errors, according to its docs.
  // if (setjmp(png_jmpbuf(png_ptr))) {
  //   std::cerr << "Got a libpng error" << std::endl;
  //   // png_destroy_info_struct(png_ptr, &info_ptr);
  //   png_destroy_write_struct(&png_ptr, &info_ptr);
  //   assert(false);
  // }

  // using a vector to store the png bytes. I'm doing this to take advantage of the
  // optimized allocation vectors do when resizing. The only downside of this approach
  // is that the vector maintains the memory, so I have to copy the vector's internal
  // memory to my own buffer
  // TODO(croot) - I could just use a vector of bytes/chars instead of
  // a shared_ptr<char>(new char[]), but I'd have to be sure to do a "shrink-to-fit" on
  // the vector if I did this to deallocate any unused memory. This might be just
  // as costly as a full memcpy --- or maybe not since the vector's memory itself is
  // also fully deallocated -- this might be a better approach.
  std::vector<char> pngData;

  png_set_write_fn(png_ptr, &pngData, writePngData, flushPngData);

  // set filtering?
  png_set_filter(png_ptr, PNG_FILTER_TYPE_BASE, PNG_FILTER_NONE);
  // png_set_filter(png_ptr, PNG_FILTER_TYPE_BASE, PNG_FILTER_SUB);
  // png_set_filter(png_ptr, PNG_FILTER_TYPE_BASE, PNG_FILTER_UP);
  // png_set_filter(png_ptr, PNG_FILTER_TYPE_BASE, PNG_FILTER_AVG);
  // png_set_filter(png_ptr, PNG_FILTER_TYPE_BASE, PNG_FILTER_PAETH);
  // png_set_filter(png_ptr, PNG_FILTER_TYPE_BASE, PNG_ALL_FILTERS);

  // set filter weights/preferences? I can't seem to get this
  // to make a difference
  // double weights[3] = {2.0, 1.5, 1.1};
  // double costs[PNG_FILTER_VALUE_LAST] = {2.0, 2.0, 1.0, 2.0, 2.0};
  // png_set_filter_heuristics(png_ptr, PNG_FILTER_HEURISTIC_WEIGHTED, 3, weights, costs);

  // set zlib compression level
  // if (compressionLevel >= 0) {
  //  png_set_compression_level(png_ptr, compressionLevel);
  //}
  png_set_compression_level(png_ptr, compressionLevel);

  // other zlib params?
  // png_set_compression_mem_level(png_ptr, 8);
  // png_set_compression_strategy(png_ptr, PNG_Z_DEFAULT_STRATEGY);
  // png_set_compression_window_bits(png_ptr, 15);
  // png_set_compression_method(png_ptr, 8);
  // png_set_compression_buffer_size(png_ptr, 8192);

  // skip the 8 bytes signature?
  // png_set_sig_bytes(png_ptr, 8);

  int interlace_type = PNG_INTERLACE_NONE;  // or PNG_INTERLACE_ADAM7 if we ever want interlacing
  png_set_IHDR(png_ptr,
               info_ptr,
               width,
               height,
               8,
               PNG_COLOR_TYPE_RGB_ALPHA,
               interlace_type,
               PNG_COMPRESSION_TYPE_DEFAULT,
               PNG_FILTER_TYPE_DEFAULT);

  /* write out the PNG header info (everything up to first IDAT) */
  png_write_info(png_ptr, info_ptr);

  // make sure < 8-bit images are packed into pixels as tightly as possible - only necessary
  // for palette images, which we're not doing yet
  // png_set_packing(png_ptr);

  png_byte* row_pointers[height];

  for (int j = 0; j < height; ++j) {
    // invert j -- input pixel rows go bottom up, where pngs are
    // defined top-down.
    row_pointers[j] = &pixels[(height - j - 1) * width * 4];
  }

  png_write_image(png_ptr, row_pointers);

  // can alternatively write per-row, but this didn't
  // seem to make a difference. I thought that perhaps
  // this could be parallelized, but png_write_row() doesn't
  // appear to be a fixed-function call.
  // for (j = 0; j < height; ++j) {
  //   png_write_row(png_ptr, row_pointers[j]);
  // }

  png_write_end(png_ptr, info_ptr);

  int pngSize = pngData.size();
  std::shared_ptr<char> pngPtr(new char[pngSize], std::default_delete<char[]>());
  char* pngDataPtr = pngPtr.get();
  std::memcpy(pngDataPtr, &pngData[0], pngSize);

  png_destroy_write_struct(&png_ptr, &info_ptr);

  delete[] pixels;
  glfwMakeContextCurrent(nullptr);

  return PngData(pngPtr, pngSize);
}

unsigned int QueryRenderManager::getIdAt(int x, int y) {
  RUNTIME_EX_ASSERT(_activeRenderer != nullptr,
                    "getIdAt(): There is no active user/widget id. Must set an active user/widget id before "
                    "requesting pixel data.");

  std::lock_guard<std::mutex> render_lock(_mtx);
  glfwMakeContextCurrent(_windowPtr);
  unsigned int id = _activeRenderer->getIdAt(x, y);
  glfwMakeContextCurrent(nullptr);

  // ids go from 0 to numitems-1, but since we're storing
  // the ids as unsigned ints, and there isn't a way to specify the
  // clear value for secondary buffers, we need to account for that
  // offset here
  return id-1;
}


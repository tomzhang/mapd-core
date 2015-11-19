#include "RenderContext.h"
#include "QueryRendererError.h"
#include <glog/logging.h>

using namespace MapD_Renderer;

void glfwErrorCallback(int error, const char* errstr) {
  // TODO(croot): should we throw an exception?
  // NOTE: There are cases when an error is caught here, but
  // is not fatal -- i.e. putting GLFW in headless mode.
  LOG(ERROR) << "GLFW error: 0x" << std::hex << error << ": " << errstr << std::endl;
}

unsigned int RenderContext::_numContexts = 0;
RenderContext* RenderContext::_currentCtx = nullptr;
// GLEWContext* RenderContext::currentGlewCtx = nullptr;

GLEWContext* glewGetContext() {
  // return RenderContext::currentGlewCtx;
  return (RenderContext::_currentCtx ? RenderContext::_currentCtx->glewContextPtr : nullptr);
}

void RenderContext::setCurrentContext(MapD_Renderer::RenderContext* ctx) {
  glfwMakeContextCurrent(ctx->windowPtr);
  _currentCtx = ctx;
  // currentGlewCtx = (ctx ? ctx->glewContextPtr : nullptr);
}

RenderContext* RenderContext::getCurrentContext() {
  return _currentCtx;
}

RenderContext::RenderContext(bool createWindow, RenderContext* prnt) {
  if (_numContexts == 0) {
    glfwSetErrorCallback(glfwErrorCallback);

    // need to init glfw -- this needs to happen in the main thread
    // TODO(croot) - can I verify if in main thread?
    if (!glfwInit()) {
      RUNTIME_EX_ASSERT(glfwInit(), "GLFW error: Couldn\'t initialize GLFW.");
    }
  }

  glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
  glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 1);
  glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
  glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);
  glfwWindowHint(GLFW_VISIBLE, createWindow);

  windowPtr = glfwCreateWindow(1, 1, "", nullptr, (prnt ? prnt->windowPtr : nullptr));
  RUNTIME_EX_ASSERT(windowPtr != nullptr, "GLFW error: Couldn't create a window.");

  glewContextPtr = new GLEWContext();

  // glfwMakeContextCurrent(windowPtr);
  setCurrentContext(this);

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

  _id = _numContexts++;
}

RenderContext::~RenderContext() {
  if (RenderContext::getCurrentContext() == this) {
    RenderContext::setCurrentContext(nullptr);
  }

  glfwDestroyWindow(windowPtr);
  windowPtr = nullptr;
  delete glewContextPtr;
  glewContextPtr = nullptr;
}

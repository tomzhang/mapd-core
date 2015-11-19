#ifndef RENDERCONTEXT_H_
#define RENDERCONTEXT_H_

#include <GL/glew.h>
#include <GLFW/glfw3.h>

GLEWContext* glewGetContext();  // needs to be defined for glew multiple context support

namespace MapD_Renderer {

// void glfwErrorCallback(int error, const char* errstr) {
//   // TODO(croot): should we throw an exception?
//   // NOTE: There are cases when an error is caught here, but
//   // is not fatal -- i.e. putting GLFW in headless mode.
//   LOG(ERROR) << "GLFW error: 0x" << std::hex << error << ": " << errstr << std::endl;
// }

struct RenderContext {
  GLFWwindow* windowPtr;
  GLEWContext* glewContextPtr;

  RenderContext(bool createWindow = false, RenderContext* prnt = nullptr);
  ~RenderContext();

  static void setCurrentContext(RenderContext* ctx);
  static RenderContext* getCurrentContext();
  static RenderContext* _currentCtx;

  // static GLEWContext* currentGlewCtx;  // this needs to reside in global memory according
  // to the Multiple Rendering Context section of
  // http://glew.sourceforge.net/advanced.html

 private:
  static unsigned int _numContexts;

  unsigned int _id;
};

}  // namespace MapD_Renderer

#endif  // RENDERCONTEXT_H_

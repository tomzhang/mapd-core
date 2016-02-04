#include "EglGLRenderer.h"
#include "EglDisplayManager.h"
#include "EglGLWindow.h"
// #include "EglUtils.h"
#include "MapDEGL.h"
// #include <EGL/egl.h>
// #include <GL/glew.h>
#include <iostream>

namespace Rendering {

using Settings::IntSetting;
using Settings::IntConstant;

namespace GL {
namespace EGL {

void makeWindowAndContextCurrent(const Rendering::Window* window,
                                 const EglDisplayShPtr& dpyPtr,
                                 const EGLContext& eglCtx) {
  const EglGLWindow* win = dynamic_cast<const EglGLWindow*>(window);
  CHECK(win);

  EGLSurface surface = win->getEGLSurface();
  CHECK(surface);

  CHECK(dpyPtr != nullptr);

  // TODO(croot): Should we make the opengl api current this every time this renderer is made current?
  eglBindAPI(EGL_OPENGL_API);
  eglMakeCurrent(dpyPtr->getEGLDisplay(), surface, surface, eglCtx);
}

EglGLRenderer::EglGLRenderer(const RendererSettings& settings) : GLRenderer(settings), _dpyPtr(nullptr), _eglCtx(0) {
}

EglGLRenderer::~EglGLRenderer() {
  std::cerr << "CROOT EglGLRenderer Destructor" << std::endl;

  if (_eglCtx) {
    CHECK(_dpyPtr != nullptr);

    // cleanup any resources
    _cleanupResources();

    EGLBoolean rtn = MAPD_CHECK_EGL_ERROR(eglDestroyContext(_dpyPtr->getEGLDisplay(), _eglCtx));
    RUNTIME_EX_ASSERT(rtn == EGL_TRUE, "EGL error trying to destroy context.");
  }
}

void EglGLRenderer::makeActiveOnCurrentThread(const Window* window) {
  // TODO(croot): a window and context can be activated
  // so we need to add the window as a per-thread item.

  // if (isActiveOnCurrentThread() && !_windowsDirty) {
  //   // already active
  //   return;
  // }

  // TODO(croot): Make thread safe?
  if (!window) {
    makeWindowAndContextCurrent(getPrimaryWindow(), _dpyPtr, _eglCtx);
  } else {
    for (auto& attachedWindow : _attachedWindows) {
      if (window == attachedWindow) {
        makeWindowAndContextCurrent(window, _dpyPtr, _eglCtx);
        break;
      }
    }
  }

  GLRenderer::makeActiveOnCurrentThread(window);

  _windowsDirty = false;
}

void EglGLRenderer::makeInactive() {
  if (!isActiveOnCurrentThread()) {
    // early out
    return;
  }

  CHECK(_dpyPtr != nullptr);
  MAPD_CHECK_EGL_ERROR(eglMakeCurrent(_dpyPtr->getEGLDisplay(), EGL_NO_SURFACE, EGL_NO_SURFACE, EGL_NO_CONTEXT));

  GLRenderer::makeInactive();
}

void EglGLRenderer::initializeGL() {
  CHECK(!isInitialized());

  const Window* primaryWindow = getPrimaryWindow();
  CHECK(primaryWindow);

  const EglGLWindow* primaryEglWindow = dynamic_cast<const EglGLWindow*>(primaryWindow);
  CHECK(primaryEglWindow);

  _initEGLDisplay(primaryEglWindow);

  EGLConfig cfg = _chooseEGLConfig(primaryEglWindow);

  EGLDisplay dpy = _dpyPtr->getEGLDisplay();

  // make sure we're using the opengl API
  // NOTE: This is set on a per-thread basis.
  // TODO(croot): If we support multiple client-side APIs in
  // the future, we'll need to set this as a per-thread attribute.
  // eglBindAPI needs to be appropriately set for eglGetCurrentContext(),
  // eglGetCurrentDisplay(), eglGetCurrentSurface(), eglCopyBuffers(), eglSwapBuffers(),
  // eglSwapInterval(), eglMakeCurrent(), eglWaitClient(), & eglWaitNative()
  EGLBoolean rtn = MAPD_CHECK_EGL_ERROR(eglBindAPI(EGL_OPENGL_API));
  RUNTIME_EX_ASSERT(rtn == EGL_TRUE, "EGL error trying to use the OpenGL API.");

  // now create the EGL context
  // TODO(croot): deal with context sharing appropriately ....
  EGLContext shCtx = EGL_NO_CONTEXT;
  // if (_parentWindowPtr->hasParent()) {
  //   const GlxGLWindow* prntGlxWindow = dynamic_cast<const GlxGLWindow*>(_prnt.get());
  //   RUNTIME_EX_ASSERT(prntGlxWindow != nullptr,
  //                     "Error: GlxGLWindow: Cannot create shared GLX context. Parent window is not a GLX window.");
  //   shCtx = prntGlxWindow->getGLXContext();
  // }

  // TODO(croot): do we need to do anything for RGBA16F/RGBA32F settings?
  std::vector<int> attributes;

  attributes.push_back(EGL_CONTEXT_OPENGL_PROFILE_MASK);
  bool useCoreProfile = isSettingActive(_settings, IntSetting::USE_CORE_PROFILE, true);
  if (useCoreProfile) {
    LOG(INFO) << "Renderer Setting: " << IntSetting::USE_CORE_PROFILE << ": True.";
    attributes.push_back(EGL_CONTEXT_OPENGL_CORE_PROFILE_BIT);
  } else {
    LOG(INFO) << "Renderer Setting: " << IntSetting::USE_CORE_PROFILE << ": False.";
    attributes.push_back(EGL_CONTEXT_OPENGL_COMPATIBILITY_PROFILE_BIT);
  }

  int32_t majorVersion = 0, minorVersion = 0;

  majorVersion = _settings.getIntSetting(IntSetting::OPENGL_MAJOR);
  IntConstant majorVersionConstant = Settings::convertToIntConstant(majorVersion);

  switch (majorVersionConstant) {
    default:
      if (majorVersion > 0) {
        minorVersion = _settings.getIntSetting(IntSetting::OPENGL_MINOR);
        RUNTIME_EX_ASSERT(minorVersion >= 0,
                          std::to_string(minorVersion) + "is an invalid value for " +
                              Settings::to_string(IntSetting::OPENGL_MINOR) + " when " +
                              Settings::to_string(IntSetting::OPENGL_MAJOR) + "is specified.");

        LOG(INFO) << "Renderer Setting: <" << IntSetting::OPENGL_MAJOR << ">.<" << IntSetting::OPENGL_MINOR
                  << ">: " << majorVersion << "." << minorVersion << ".";

        attributes.push_back(EGL_CONTEXT_MAJOR_VERSION);
        attributes.push_back(majorVersion);
        attributes.push_back(EGL_CONTEXT_MINOR_VERSION);
        attributes.push_back(minorVersion);

        break;
      }

      // let pass thru
      LOG(WARNING) << "Invalid value for " << IntSetting::OPENGL_MAJOR << ". Using default (highest supported "
                                                                          "version).";

    case IntConstant::UNDEFINED:
    case IntConstant::DEFAULT:
    case IntConstant::AUTO:
    case IntConstant::ON:
    case IntConstant::OFF:
      break;
  }

  // always use forward-compatible GL contexts (i.e. deprecate old API)
  // attributes.push_back(EGL_CONTEXT_OPENGL_FORWARD_COMPATIBLE);
  // attributes.push_back(EGL_TRUE);

  // TODO(croot): support robust buffer access? EGL_CONTEXT_OPENGL_ROBUST_ACCESS,
  // GL_ARB_robustness extension (also EGL_CONTEXT_OPENGL_RESET_NOTIFICATION_STRATEGY)

  // close the attribute list
  attributes.push_back(EGL_NONE);

  _eglCtx = MAPD_CHECK_EGL_ERROR(eglCreateContext(dpy, cfg, shCtx, &attributes[0]));

  RUNTIME_EX_ASSERT(_eglCtx != EGL_NO_CONTEXT, "Error: EglGLRenderer: Context creation failed.");

  _initGLEW(primaryEglWindow);
}

void EglGLRenderer::_initEGLDisplay(const EglGLWindow* primaryWindow) {
  CHECK(_dpyPtr == nullptr);

  _dpyPtr = primaryWindow->getEGLDisplayPtr();

  RUNTIME_EX_ASSERT(_dpyPtr, "EglGLRenderer: Cannot connect to EGL display.");
}

EGLConfig EglGLRenderer::_chooseEGLConfig(const EglGLWindow* primaryWindow) {
  return primaryWindow->getEGLConfig();

  // CHECK(_dpyPtr != nullptr);
  // EGLDisplay dpy = _dpyPtr->getEGLDisplay();

  // std::vector<int> attributes;

  // // TODO(croot): retrieve render type from settings or argument

  // const IntConstant drawableType = primaryWindow->getIntSettingAsConstant(IntSetting::DRAWABLE_TYPE);

  // // set up attributes for the color and ancillary buffers
  // appendFbAttrs(this, drawableType, _settings, attributes);

  // // the attribute list needs to always end with None
  // attributes.push_back(EGL_NONE);

  // int numCfgs;
  // EGLConfig cfg;

  // EGLBoolean rtn = MAPD_CHECK_EGL_ERROR(eglChooseConfig(dpy, &attributes[0], &cfg, 1, &numCfgs));

  // RUNTIME_EX_ASSERT(rtn == EGL_TRUE && numCfgs > 0, "Error trying to choose an EGL config for renderer.");

  // return cfg;
}

bool EglGLRenderer::_verifyWindowIsAttachable(const Window* window) {
  return (window != nullptr && dynamic_cast<const EglGLWindow*>(window) != nullptr);
}

}  // namespace EGL
}  // namespace GL
}  // namespace Rendering

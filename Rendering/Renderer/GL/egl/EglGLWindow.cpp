#include "EglGLWindow.h"
#include "EglDisplayManager.h"
#include "EglGLRenderer.h"
#include "EglUtils.h"
#include "MapDEGL.h"
#include <string>

namespace Rendering {

using Settings::StrSetting;
using Settings::IntConstant;
using Settings::IntSetting;

namespace GL {
namespace EGL {

EglGLWindow::EglGLWindow(const Settings::WindowSettings& windowSettings,
                         EglDisplayManager& displayMgr,
                         const WindowShPtr& prnt)
    : GLWindow(windowSettings, prnt), _dpyPtr(nullptr), _gpuId(-1), _surface(0), _eglCfg(0) {
  _initEGLDisplay(displayMgr);
}

EglGLWindow::~EglGLWindow() {
  if (_surface && _dpyPtr) {
    eglDestroySurface(_dpyPtr->getEGLDisplay(), _surface);
    _surface = 0;
  }
}

size_t EglGLWindow::getGpuId() const {
  RUNTIME_EX_ASSERT(_gpuId >= 0 && _dpyPtr,
                    "The GLWindow has not been initialized yet. Cannot get the Gpu ID associated with it.");
  return _gpuId;
}

void EglGLWindow::_initEGLDisplay(EglDisplayManager& displayMgr) {
  int gpuId = _settings.getIntSetting(IntSetting::GPU_ID);
  if (gpuId < 0) {
    gpuId = 0;
  }

  _dpyPtr = displayMgr.connectToDisplay(gpuId);
  RUNTIME_EX_ASSERT(_dpyPtr, "EglGLWindow: Cannot connect to device " + std::to_string(gpuId));

  _gpuId = gpuId;
}

void EglGLWindow::_init(Renderer* renderer) {
  EglGLRenderer* eglRenderer = dynamic_cast<EglGLRenderer*>(renderer);
  RUNTIME_EX_ASSERT(eglRenderer != nullptr, "Cannot initialize EglGLWindow. Renderer is not a EglGLRenderer.");

  // TODO(croot): Do we need to do anything special about initializing
  // for multiple threads?

  _eglCfg = _chooseEGLConfig(eglRenderer);

  _initSurface(eglRenderer, _eglCfg);

  // we need an initialized renderer to create the
  // drawable properly, so ensure that the renderer
  // this window's going to attach to is initialized.
  _initRenderer(eglRenderer);

  eglRenderer->makeActiveOnCurrentThread(this);
  _initSwap(eglRenderer);
}

EGLConfig EglGLWindow::_chooseEGLConfig(EglGLRenderer* renderer) {
  CHECK(_dpyPtr != nullptr);
  EGLDisplay dpy = _dpyPtr->getEGLDisplay();

  std::vector<int> attributes;

  // only supporting opengl
  attributes.push_back(EGL_RENDERABLE_TYPE);
  attributes.push_back(EGL_OPENGL_BIT);

  // // Setup the drawable type.
  const IntConstant drawableType = _settings.getIntSettingAsConstant(IntSetting::DRAWABLE_TYPE);
  switch (drawableType) {
    case IntConstant::PBUFFER:
      attributes.push_back(EGL_SURFACE_TYPE);
      attributes.push_back(EGL_PBUFFER_BIT);
      break;
    default:
      LOG(WARNING) << "Unknown drawable type: " << drawableType << ". Using " << IntConstant::WINDOW << " as default.";
    case IntConstant::DEFAULT:
    case IntConstant::AUTO:
    case IntConstant::ON:
    case IntConstant::OFF:
    case IntConstant::WINDOW:

      attributes.push_back(EGL_SURFACE_TYPE);
      attributes.push_back(EGL_WINDOW_BIT);
      break;
    case IntConstant::FBO:
      // TODO(croot): is FBO a pixmap?
      attributes.push_back(EGL_SURFACE_TYPE);
      attributes.push_back(EGL_PIXMAP_BIT);
      break;
  }

  // TODO(croot): what about EGL_MULTISAMPLE_RESOLVE_BOX_BIT, EGL_SWAP_BEHAVIOR_PRESERVED_BIT
  // for EGL_SURFACE_TYPE?

  LOG(INFO) << "Window Setting: " << IntSetting::DRAWABLE_TYPE << ": " << drawableType << ".";

  // // handles setting up attributes for the color and ancillary buffers
  appendFbAttrs(renderer, drawableType, _settings, attributes);

  // handle stereo rendering
  // bool doStereo = isSettingActive(_settings, IntSetting::USE_STEREO, false);
  // if (doStereo) {
  // THROW_RUNTIME_EX("EGL does not currently support stereo rendering");
  //   LOG(INFO) << "Window Setting: " << IntSetting::USE_STEREO << ": True.";
  // }

  // the attribute list needs to always end with None
  attributes.push_back(EGL_NONE);

  int numCfgs;
  EGLConfig cfg;

  EGLBoolean rtn = MAPD_CHECK_EGL_ERROR(eglChooseConfig(dpy, &attributes[0], &cfg, 1, &numCfgs));

  RUNTIME_EX_ASSERT(rtn == EGL_TRUE && numCfgs > 0, "Error trying to choose an EGL config for window.");

  return cfg;
}

void EglGLWindow::_initSurface(EglGLRenderer* renderer, EGLConfig& cfg) {
  CHECK(_dpyPtr != nullptr);
  EGLDisplay dpy = _dpyPtr->getEGLDisplay();

  std::vector<int> attributes;

  // handle double buffering
  const IntConstant drawableType = _settings.getIntSettingAsConstant(IntSetting::DRAWABLE_TYPE);
  if (drawableType != IntConstant::PBUFFER) {
    attributes.push_back(EGL_RENDER_BUFFER);
    bool doDoubleBuffering = isSettingActive(_settings, IntSetting::USE_DOUBLEBUFFER, true);
    if (!doDoubleBuffering) {
      attributes.push_back(EGL_SINGLE_BUFFER);
      LOG(INFO) << "Window Setting: " << IntSetting::USE_DOUBLEBUFFER << ": False.";
    } else {
      attributes.push_back(EGL_BACK_BUFFER);
      LOG(INFO) << "Window Setting: " << IntSetting::USE_DOUBLEBUFFER << ": True.";
    }
  }

  // Create the surface
  switch (drawableType) {
    case IntConstant::PBUFFER: {
      int32_t w, h;
      Viewport vp = getViewport();

      w = vp.getWidth();
      h = vp.getHeight();

      attributes.push_back(EGL_WIDTH);
      attributes.push_back(w);
      attributes.push_back(EGL_HEIGHT);
      attributes.push_back(h);

      attributes.push_back(EGL_NONE);

      _surface = MAPD_CHECK_EGL_ERROR(eglCreatePbufferSurface(dpy, cfg, &attributes[0]));
    } break;
    default:
      LOG(WARNING) << "Unknown drawable type: " << drawableType << ". Using " << IntConstant::WINDOW << " as default.";
    case IntConstant::DEFAULT:
    case IntConstant::AUTO:
    case IntConstant::ON:
    case IntConstant::OFF:
    case IntConstant::WINDOW:
      // TODO(croot): need a way to use a platform-specific window, and implementing that
      // means we need to choose a config that matches the native window.

      // finish the attributes off
      attributes.push_back(EGL_NONE);

      _surface = MAPD_CHECK_EGL_ERROR(eglCreateWindowSurface(dpy, cfg, 0, &attributes[0]));
      // _surface = MAPD_CHECK_EGL_ERROR(eglCreatePlatformWindowSurface(dpy, cfg, 0, &attributes[0]));
      break;
    case IntConstant::FBO:
      // TODO(croot): need a way to use a platform-specific pixmap

      // finish the attributes off
      attributes.push_back(EGL_NONE);

      _surface = MAPD_CHECK_EGL_ERROR(eglCreatePixmapSurface(dpy, cfg, 0, &attributes[0]));
      // _surface = MAPD_CHECK_EGL_ERROR(eglCreatePlatformPixmapSurface(dpy, cfg, 0, &attributes[0]));
      break;
  }

  // TODO(croot): should we deal with EGL_GL_COLORSPACE? It defaults to EGL_GL_COLORSPACE_LINEAR
  // and that's what we want, but we may want to support sRGB at some point.
}

void EglGLWindow::_initRenderer(EglGLRenderer* renderer) {
  if (!renderer->isInitialized()) {
    renderer->initialize();
  }
}

void EglGLWindow::_initSwap(EglGLRenderer* renderer) {
  CHECK(_dpyPtr != nullptr);
  EGLBoolean hasErrors = MAPD_CHECK_EGL_ERROR(eglSwapInterval(_dpyPtr->getEGLDisplay(), 1));
  RUNTIME_EX_ASSERT(hasErrors == EGL_TRUE, "EGL error setting swap interval");
}

void EglGLWindow::swapBuffers() {
  CHECK(_dpyPtr != nullptr);
  CHECK(_surface);

  EGLBoolean hasErrors = MAPD_CHECK_EGL_ERROR(eglSwapBuffers(_dpyPtr->getEGLDisplay(), _surface));
  RUNTIME_EX_ASSERT(hasErrors == EGL_TRUE, "EGL error trying to swap buffers.");
}

}  // namespace EGL
}  // namespace GL
}  // namespace Rendering

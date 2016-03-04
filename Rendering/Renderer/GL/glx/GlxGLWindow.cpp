#include "GlxGLWindow.h"
#include "GlxGLRenderer.h"
#include "GlxUtils.h"
#include "X11DisplayManager.h"
#include <assert.h>
#include <X11/Xlib.h>
#include <X11/Xatom.h>
#include <vector>
#include <string>
#include "../../../RenderError.h"

namespace Rendering {

using Settings::IntSetting;
using Settings::IntConstant;
using Settings::StrSetting;

namespace GL {
namespace GLX {

static Bool WaitForNotify(Display* dpy, XEvent* e, char* arg) {
  // This is from Equalizer
  return (e->type == MapNotify) && (e->xmap.window == (::Window)arg);

  // TODO(croot): check that the event's window is the window that is recently
  // created?
  // return (e->type == MapNotify);
}

GlxGLWindow::GlxGLWindow(const WindowSettings& windowSettings,
                         X11DisplayManager& displayManager,
                         const WindowShPtr& prnt)
    : GLWindow(windowSettings, prnt), _dpyConnection(nullptr, -1), _xDrawable(0) {
  _initXDisplay(displayManager);
}

GlxGLWindow::~GlxGLWindow() {
  if (_xDrawable && _dpyConnection.first) {
    // TODO(croot): what if the drawable is a fbo or pbuffer?
    // TODO(croot): This may not be the best place to call
    // XDestroyWindow() since we're holding onto the renderer/context
    // as a member variable (_rendererPtr), and that won't get
    // destroyed until after this destructor, which may not
    // be desirable. We may want to pull this out and let
    // the X11WindowManager.cpp handle the XDestroyWindow() call.

    Display* dpy = _dpyConnection.first.get();

    const IntConstant drawableType = _settings.getIntSettingAsConstant(IntSetting::DRAWABLE_TYPE);
    switch (drawableType) {
      case IntConstant::PBUFFER:
        glXDestroyPbuffer(dpy, _xDrawable);
        break;
      case IntConstant::FBO:
        glXDestroyPixmap(dpy, _xDrawable);
        break;
      default:
        XDestroyWindow(dpy, _xDrawable);
        break;
    }
  }
}

size_t GlxGLWindow::getGpuId() const {
  RUNTIME_EX_ASSERT(_dpyConnection.second >= 0,
                    "The GLWindow has not been initialized yet. Cannot get the Gpu ID associated with it.");
  return _dpyConnection.second;
}

X11DisplayShPtr GlxGLWindow::getXDisplayPtr() const {
  return _dpyConnection.first;
}

int GlxGLWindow::getXScreenId() const {
  return _dpyConnection.second;
}

const DisplayScreenPair& GlxGLWindow::getXConnectionData() const {
  return _dpyConnection;
}

// GLXContext GlxGLWindow::getGLXContext() const {
//   RUNTIME_EX_ASSERT(_rendererPtr != nullptr, "GlxGLRenderer is uninitialized.");

//   GlxGLRenderer* glxGLRenderer = dynamic_cast<GlxGLRenderer*>(_rendererPtr.get());

//   RUNTIME_EX_ASSERT(glxGLRenderer, "GLRenderer was initialized as the wrong type.");
//   return glxGLRenderer->getGLXContext();
// }

X11ID GlxGLWindow::getXDrawable() const {
  return _xDrawable;
}

// #ifdef GLEW_MX
// const GLXEWContext* GlxGLWindow::glxewGetContext() const {
//   RUNTIME_EX_ASSERT(_rendererPtr != nullptr, "GlxGLRenderer is uninitialized.");

//   GlxGLRenderer* glxGLRenderer = dynamic_cast<GlxGLRenderer*>(_rendererPtr.get());

//   RUNTIME_EX_ASSERT(glxGLRenderer, "GLRenderer was initialized as the wrong type.");
//   return glxGLRenderer->glxewGetContext();
// }

// GLXEWContext* GlxGLWindow::glxewGetContext() {
//   return const_cast<GLXEWContext*>(static_cast<const GlxGLWindow&>(*this).glxewGetContext());
// }
// #endif

void GlxGLWindow::swapBuffers() {
  CHECK(_dpyConnection.first != nullptr);
  CHECK(_xDrawable != 0);

  glXSwapBuffers(_dpyConnection.first.get(), _xDrawable);
}

void GlxGLWindow::_setXDrawable(X11ID xDrawable) {
  CHECK(_dpyConnection.first != nullptr);
  Display* dpy = _dpyConnection.first.get();

  if (xDrawable == _xDrawable) {
    // early out
    return;
  }

  if (_xDrawable) {
    _exitEventHandler();
  }

  _xDrawable = xDrawable;

  if (!xDrawable) {
    // undefined drawable. No need to continue.
    return;
  }

  // TODO(croot): retrieve render type from settings or argument

  const IntConstant drawableType = _settings.getIntSettingAsConstant(IntSetting::DRAWABLE_TYPE);

  if (drawableType != IntConstant::OFF) {
    _initEventHandler();
  }

  switch (drawableType) {
    case IntConstant::PBUFFER:
      unsigned int width, height;
      glXQueryDrawable(dpy, xDrawable, GLX_WIDTH, &width);
      glXQueryDrawable(dpy, xDrawable, GLX_HEIGHT, &height);
      setViewport(0, 0, width, height);
      break;

    case IntConstant::FBO:
      // if (drawableType == 1) {  // if (drawableType == FBO)
      // TODO(croot): make sure pixel viewport has a valid area here.
      // } else {
      if (1) {
        THROW_RUNTIME_EX("Error: GlxGLWindow: Unknown drawable type: " + std::to_string(xDrawable));
      }
      break;

    case IntConstant::DEFAULT:
    case IntConstant::AUTO:
    case IntConstant::ON:
    case IntConstant::OFF:
    case IntConstant::WINDOW:
    default:
      XWindowAttributes wa;
      XGetWindowAttributes(dpy, xDrawable, &wa);

      ::Window root, parent, *children;
      unsigned int nChildren;

      XQueryTree(dpy, xDrawable, &root, &parent, &children, &nChildren);

      if (children != 0) {
        XFree(children);
      }

      int x, y;
      ::Window childReturn;
      XTranslateCoordinates(dpy, parent, root, wa.x, wa.y, &x, &y, &childReturn);

      setViewport(x, y, wa.width, wa.height);
      break;
  }
}

void GlxGLWindow::_init(Renderer* renderer) {
  GlxGLRenderer* glxRenderer = dynamic_cast<GlxGLRenderer*>(renderer);
  RUNTIME_EX_ASSERT(glxRenderer != nullptr, "Cannot initialize GlxGLWindow. Renderer is not a GlxGLRenderer.");

  // TODO(croot): should initialize glx on renderer construction, but that would
  // require a display for the renderer. Perhaps the WindowManager will hold a
  // data structure of all the open displays to use and either pass that as a constructor
  // argument or have an api to query for open displays
  glxRenderer->_initGLX(_dpyConnection);

  FbConfigShPtr fbConfigPtr = _chooseFbConfig(glxRenderer);
  // FbConfigShPtr fbConfigPtr = glxRenderer->getFbConfigPtr();

  _initXDrawable(glxRenderer, fbConfigPtr);

  _initRenderer(glxRenderer);

  _initSwap(glxRenderer);
}

void GlxGLWindow::_initXDisplay(X11DisplayManager& displayMgr) {
  std::string displayStr = _settings.getStrSetting(StrSetting::DISPLAY);

  int gpuId = _settings.getIntSetting(IntSetting::GPU_ID);
  _dpyConnection = displayMgr.connectToDisplay(displayStr, gpuId);
  RUNTIME_EX_ASSERT(_dpyConnection.first, "GlxGLWindow: Cannot connect to X server.");
}

#ifdef GLEW_MX
#undef glxewGetContext
#define glxewGetContext renderer->glxewGetContext
#endif
FbConfigShPtr GlxGLWindow::_chooseFbConfig(GlxGLRenderer* renderer) {
  CHECK(_dpyConnection.first != nullptr);
  Display* dpy = _dpyConnection.first.get();
  int screen = _dpyConnection.second;

  RUNTIME_EX_ASSERT(GLXEW_VERSION_1_3 || GLXEW_SGIX_fbconfig,
                    "Error: GlxGLWindow: GLX framebuffer config support required!");

  std::vector<int> attributes;

  // Setup the drawable type.
  const IntConstant drawableType = _settings.getIntSettingAsConstant(IntSetting::DRAWABLE_TYPE);
  switch (drawableType) {
    case IntConstant::PBUFFER:
      attributes.push_back(GLX_DRAWABLE_TYPE);
      attributes.push_back(GLX_PBUFFER_BIT);
      break;

    default:
      LOG(WARNING) << "Unknown drawable type: " << drawableType << ". Using " << IntConstant::WINDOW << " as default.";

    case IntConstant::DEFAULT:
    case IntConstant::AUTO:
    case IntConstant::ON:
    case IntConstant::OFF:
    case IntConstant::FBO:  // this is not a bug. FBO needs an fbConfig with a visual for a dummy window
    case IntConstant::WINDOW:
      attributes.push_back(GLX_X_RENDERABLE);
      attributes.push_back(True);
      break;
  }

  LOG(INFO) << "Window Setting: " << IntSetting::DRAWABLE_TYPE << ": " << drawableType << ".";

  // handles setting up attributes for the color and ancillary buffers
  appendFbBufferAttrs(renderer, drawableType, _settings, attributes);

  // handle stereo rendering
  // bool doStereo = isSettingActive(_settings, IntSetting::USE_STEREO, false);
  // if (doStereo) {
  //     attributes.push_back(GLX_STEREO);
  //     attributes.push_back(True);
  //   LOG(INFO) << "Window Setting: " << IntSetting::USE_STEREO << ": True.";
  // }

  // handle double buffering
  bool doDoubleBuffering = isSettingActive(_settings, IntSetting::USE_DOUBLEBUFFER, true);
  if (doDoubleBuffering) {
    attributes.push_back(GLX_DOUBLEBUFFER);
    attributes.push_back(True);
    LOG(INFO) << "Window Setting: " << IntSetting::USE_DOUBLEBUFFER << ": True.";
  }

  // the attribute list needs to always end with None
  attributes.push_back(None);

  // backoff list? See: Equalizer:src/eq/glx/Window.cpp, lines: 289-302

  // Now choose an fb config given the set of attributes
  PFNGLXCHOOSEFBCONFIGSGIXPROC chooseFBConfig = GLXEW_VERSION_1_3 ? glXChooseFBConfig : glXChooseFBConfigSGIX;

  // NOTE: the default screen will be that which is set in the GPU_ID setting of the window
  // settings.
  int nConfigs = 0;

  // make a unique ptr to the config ptrs created in the above chooseFBConfig() call
  // to ensure it gets freed at all possible return locations.
  FbConfigShPtr fbConfigPtr(chooseFBConfig(dpy, screen, &attributes[0], &nConfigs), XFree);

  // if couldn't find configs, remove backoff attrs 1 by 1, looking for configs that may work
  // See: Equalizer:src/eq/glx/Window.cpp, lines: 312-324

  RUNTIME_EX_ASSERT(
      nConfigs != 0,
      "Error: GlxGLWindow: cannot find a framebuffer configuration matching the window settings supplied.");

  return fbConfigPtr;
}

void GlxGLWindow::_initRenderer(GlxGLRenderer* renderer) {
  if (!renderer->isInitialized()) {
    renderer->initialize();
  }
}

void GlxGLWindow::_initXDrawable(GlxGLRenderer* renderer, const FbConfigShPtr& fbConfigPtr) {
  CHECK(fbConfigPtr != nullptr);

  // TODO(croot): retrieve render type from settings or argument
  bool success = false;
  X11ID drawable = 0;
  const IntConstant drawableType = _settings.getIntSettingAsConstant(IntSetting::DRAWABLE_TYPE);

  switch (drawableType) {
    case IntConstant::PBUFFER:
      success = _configurePBuffer(renderer, fbConfigPtr, drawable);
      break;

    case IntConstant::FBO:
      success = false;  // TODO(croot) success = _createGlxWindow(fbConfig);
      break;

    case IntConstant::OFF:
      break;

    case IntConstant::DEFAULT:
    case IntConstant::AUTO:
    case IntConstant::ON:
    case IntConstant::WINDOW:
    default:
      success = _configureWindow(renderer, fbConfigPtr, drawable);
      break;
  }

  RUNTIME_EX_ASSERT(success, "Error: GlxGLWindow: X Drawable configuration failed.");
  _setXDrawable(drawable);

  if (drawableType == IntConstant::FBO) {
    success = false;  // TODO(croot) success = configureFBO();
  }

  RUNTIME_EX_ASSERT(success, "ERROR: GlxGLWindow: Initialization failed.");
}

#ifdef GLEW_MX
#undef glxewGetContext
#define glxewGetContext renderer->glxewGetContext
#endif

void GlxGLWindow::_initSwap(GlxGLRenderer* renderer) {
  // TODO(croot): verify that from settings that we're trying to
  // create a window here.

  // TODO(croot): do we need to verify double buffering?

  // TODO(croot): get swap interval from settings?

  const IntConstant drawableType = _settings.getIntSettingAsConstant(IntSetting::DRAWABLE_TYPE);
  if (drawableType == IntConstant::PBUFFER) {
    // no swapping for pbuffers
    return;
  }

  renderer->makeActiveOnCurrentThread(this);
  const int32_t swapInterval = 1;

  // #ifdef GLEW_MX
  //   if (GLXEW_SGI_swap_control) {
  //     glXSwapIntervalSGI(swapInterval < 0 ? 1 : swapInterval);  // TODO(croot): set a default swap interval
  //     somewhere
  //   } else {
  // #else
  //   {
  // #endif
  //     LOG(WARNING) << "GLX_SGI_swap_control not supported. Ignoring swap interval in settings.";
  //   }

  if (GLXEW_SGI_swap_control) {
    glXSwapIntervalSGI(swapInterval < 0 ? 1 : swapInterval);  // TODO(croot): set a default swap interval somewhere
  } else {
    LOG(WARNING) << "GLX_SGI_swap_control not supported. Ignoring swap interval in settings.";
  }
}

bool GlxGLWindow::_configureWindow(GlxGLRenderer* renderer, const FbConfigShPtr& fbConfigPtr, X11ID& drawable) {
  CHECK(_dpyConnection.first != nullptr);
  Display* dpy = _dpyConnection.first.get();
  // int screen = _dpyConnection.second;

  CHECK(fbConfigPtr != nullptr);

  int32_t x, y, w, h;
  Viewport vp = getViewport();

  x = vp.getXPos();
  y = vp.getYPos();
  w = vp.getWidth();
  h = vp.getHeight();

  // x = WindowSettings::getViewportSetting(
  //     _settings, IntSetting::X_POS, 0);  // TODO(croot): expose default X_POS somewhere
  // y = WindowSettings::getViewportSetting(
  //     _settings, IntSetting::X_POS, 0);  // TODO(croot): expose default Y_POS somewhere
  // w = WindowSettings::getViewportSetting(
  //     _settings, IntSetting::WIDTH, 100);  // TODO(croot): expose default WIDTH somewhere
  // h = WindowSettings::getViewportSetting(
  //     _settings, IntSetting::HEIGHT, 100);  // TODO(croot): expose default HEIGHT somewhere

  bool isFullscreen = isSettingActive(_settings, IntSetting::USE_FULLSCREEN, false);
  bool showDecoration = isSettingActive(_settings, IntSetting::USE_DECORATION, true);

  // if (isFullscreen) {
  //   w = DisplayWidth(dpy, screen);
  //   h = DisplayHeight(dpy, screen);
  //   x = 0;
  //   y = 0;
  // }

  // TODO(croot): where do we put the setViewport()? On the window or on the renderer? Perhaps in both places - the
  // window viewport controls the full window size, but the rendererer would handle sub-views in the window?
  // setViewport(x, y, w, h);

  drawable = _createGlxWindow(renderer, fbConfigPtr, x, y, w, h);

  if (!drawable) {
    return false;
  }

  // map and wait for MapNotify event
  XMapWindow(dpy, drawable);

  XEvent event;
  XIfEvent(dpy, &event, WaitForNotify, (XPointer)(drawable));

  XMoveResizeWindow(dpy, drawable, x, y, w, h);
  XFlush(dpy);

  if (isFullscreen || !showDecoration) {
    // TODO(croot): what does the following call do?
    // On the surface, it seems to enable keyboard events only for the
    // new window created, leaving it impossible to interact with any
    // other windows. There must be something being done in equalizer
    // with window events to properly handle things, or the intent is to never leave fullscreen
    // when the window is set to be fullscreen. This might be the case for
    // fullscreen games. Until I investigate more, leave this commented out.
    // XGrabKeyboard(dpy, drawable, true, GrabModeAsync, GrabModeAsync, CurrentTime);
  }

  return true;
}

#ifdef GLEW_MX
#undef glxewGetContext
#define glxewGetContext renderer->glxewGetContext
#endif

X11ID GlxGLWindow::_createGlxWindow(GlxGLRenderer* renderer,
                                    const FbConfigShPtr& fbConfigPtr,
                                    int32_t x,
                                    int32_t y,
                                    int32_t w,
                                    int32_t h) {
  // TODO(croot): make sure we configured to create a window
  // CHECK(!pbuffer);
  CHECK(_dpyConnection.first != nullptr);
  Display* dpy = _dpyConnection.first.get();
  int screen = _dpyConnection.second;

  CHECK(fbConfigPtr != nullptr);
  GLXFBConfig* fbConfig = fbConfigPtr.get();

  // #ifdef GLEW_MX
  //   std::unique_ptr<XVisualInfo, int (*)(void*)> visInfoUqPtr(
  //       GLXEW_VERSION_1_3 ? glXGetVisualFromFBConfig(dpy, fbConfig[0]) : glXGetVisualFromFBConfigSGIX(dpy,
  //       fbConfig[0]),
  //       XFree);
  // #else
  //   std::unique_ptr<XVisualInfo, int (*)(void*)> visInfoUqPtr(glXGetVisualFromFBConfig(dpy, fbConfig[0]), XFree);
  // #endif

  std::unique_ptr<XVisualInfo, int (*)(void*)> visInfoUqPtr(
      GLXEW_VERSION_1_3 ? glXGetVisualFromFBConfig(dpy, fbConfig[0]) : glXGetVisualFromFBConfigSGIX(dpy, fbConfig[0]),
      XFree);

  RUNTIME_EX_ASSERT(visInfoUqPtr != nullptr, "Error: GlxGLWindow: No visual for fb config. Window creation failed.");
  XVisualInfo* visInfo = visInfoUqPtr.get();

  // NOTE: the default screen will be that set with the GPU_ID of the window settings
  ::Window prnt = RootWindow(dpy, screen);

  XSetWindowAttributes wa;
  wa.colormap = XCreateColormap(dpy, prnt, visInfo->visual, AllocNone);
  wa.background_pixmap = None;
  wa.border_pixel = 0;
  wa.event_mask = StructureNotifyMask | VisibilityChangeMask | ExposureMask | KeyPressMask | KeyReleaseMask |
                  PointerMotionMask | ButtonPressMask | ButtonReleaseMask;

  bool showDecoration = isSettingActive(_settings,
                                        IntSetting::USE_DECORATION,
                                        true);  // TODO(croot): expose window setting defaults somewhere

  bool isFullscreen = isSettingActive(_settings, IntSetting::USE_FULLSCREEN, false);

  if (showDecoration || isFullscreen) {
    wa.override_redirect = False;
  } else {
    wa.override_redirect = True;
  }

  ::Window window = XCreateWindow(dpy,
                                  prnt,
                                  x,
                                  y,
                                  w,
                                  h,
                                  0,
                                  visInfo->depth,
                                  InputOutput,
                                  visInfo->visual,
                                  CWBackPixmap | CWBorderPixel | CWEventMask | CWColormap | CWOverrideRedirect,
                                  &wa);

  RUNTIME_EX_ASSERT(window, "Error: GlxGLWindow: Failed creating X11 window.");

  std::stringstream windowTitle;

  // TODO(croot): add name string to title
  windowTitle << _settings.getStrSetting(StrSetting::NAME);

  XStoreName(dpy, window, windowTitle.str().c_str());

  // Register for close window request
  Atom deleteAtom = XInternAtom(dpy, "WM_DELETE_WINDOW", False);
  XSetWMProtocols(dpy, window, &deleteAtom, 1);

  if (isFullscreen) {
    Atom atoms[2] = {XInternAtom(dpy, "_NET_WM_STATE_FULLSCREEN", False), None};
    XChangeProperty(
        dpy, window, XInternAtom(dpy, "_NET_WM_STATE", False), XA_ATOM, 32, PropModeReplace, (unsigned char*)atoms, 1);
  }

  return window;
}

bool GlxGLWindow::_configurePBuffer(GlxGLRenderer* renderer, const FbConfigShPtr& fbConfigPtr, X11ID& drawable) {
  CHECK(_dpyConnection.first != nullptr);
  Display* dpy = _dpyConnection.first.get();

  CHECK(fbConfigPtr != nullptr);
  GLXFBConfig* fbConfig = fbConfigPtr.get();

  int32_t w, h;
  Viewport vp = getViewport();

  w = vp.getWidth();
  h = vp.getHeight();

  std::vector<int> attributes = {
      GLX_PBUFFER_WIDTH, w, GLX_PBUFFER_HEIGHT, h, GLX_LARGEST_PBUFFER, True, GLX_PRESERVED_CONTENTS, True, None};

  drawable = glXCreatePbuffer(dpy, fbConfig[0], &attributes[0]);
  RUNTIME_EX_ASSERT(drawable != 0, "GlxGLWindow error creating pbuffer.");

  XFlush(dpy);

  return true;
}

void GlxGLWindow::_initEventHandler() {
  // TODO(croot): setup an event handler class
}

void GlxGLWindow::_exitEventHandler() {
  // TODO(croot): destory whatever event handler object(s) were created in the _initEventHandler() func
}

}  // namespace GLX
}  // namespace GL
}  // namespace Rendering

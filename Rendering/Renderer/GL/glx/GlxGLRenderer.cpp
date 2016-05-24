#include "GlxGLRenderer.h"
#include "GlxGLWindow.h"
// #include "../../../RenderError.h"
// #include "../Settings.h"
#include "GlxUtils.h"
#include <glog/logging.h>

namespace Rendering {

using Settings::IntSetting;
using Settings::IntConstant;

namespace GL {
namespace GLX {

void makeWindowAndContextCurrent(const Rendering::Window* window,
                                 const X11DisplayShPtr& xDisplayPtr,
                                 const GLXContext& glxContext) {
  const GlxGLWindow* win = dynamic_cast<const GlxGLWindow*>(window);
  CHECK(win);

  CHECK(xDisplayPtr != nullptr);

  auto xDrawable = win->getXDrawable();
  CHECK(xDrawable != 0);

  glXMakeCurrent(xDisplayPtr.get(), xDrawable, glxContext);
}

GlxGLRenderer::GlxGLRenderer(const RendererSettings& settings)
    : GLRenderer(settings),
      _glxContext(0),
      _dpyConnection(nullptr, -1),
      _fbConfigPtr(nullptr, XFree),
      _glxewInitialized(false) {
#ifdef GLEW_MX
  memset(&_glxewContext, 0, sizeof(GLXEWContext));
#endif
}

// GlxGLRenderer::GlxGLRenderer(const WindowShPtr& parentWindowPtr, const FbConfigUqPtr& fbConfigPtr)
//     : GlxGLRenderer(parentWindowPtr) {
//   initialize(fbConfigPtr);
// }

GlxGLRenderer::~GlxGLRenderer() {
  if (_glxContext) {
    CHECK(_dpyConnection.first != nullptr);

    // cleanup any resources
    _cleanupResources();

    glXDestroyContext(_dpyConnection.first.get(), _glxContext);
  }

  // #ifdef GLEW_MX
  //   memset(&_glxewContext, 0, sizeof(GLXEWContext));
  // #endif
}

void GlxGLRenderer::_makeActiveOnCurrentThreadImpl(Window* window) {
  CHECK(_dpyConnection.first && _glxContext);
  makeWindowAndContextCurrent(window, _dpyConnection.first, _glxContext);
}

void GlxGLRenderer::_makeInactiveImpl() {
  CHECK(_dpyConnection.first != nullptr);
  glXMakeCurrent(_dpyConnection.first.get(), None, nullptr);
}

GLXContext GlxGLRenderer::getGLXContext() const {
  return _glxContext;
}

X11DisplayShPtr GlxGLRenderer::getXDisplayPtr() const {
  return _dpyConnection.first;
}

int GlxGLRenderer::getXScreenId() const {
  return _dpyConnection.second;
}

/** for CROOT testing **/
FbConfigShPtr GlxGLRenderer::getFbConfigPtr() const {
  return _fbConfigPtr;
}

#ifdef GLEW_MX
const GLXEWContext* GlxGLRenderer::glxewGetContext() const {
  return &_glxewContext;
}

GLXEWContext* GlxGLRenderer::glxewGetContext() {
  return const_cast<GLXEWContext*>(static_cast<const GlxGLRenderer&>(*this).glxewGetContext());
}
#endif

void GlxGLRenderer::_initXDisplay(const GlxGLWindow* primaryWindow) {
  CHECK(_dpyConnection.first == nullptr);

  _dpyConnection.first = primaryWindow->getXDisplayPtr();
  _dpyConnection.second = primaryWindow->getXScreenId();

  RUNTIME_EX_ASSERT(_dpyConnection.first, "GlxGLRenderer: Cannot connect to X server.");
}

void GlxGLRenderer::_initGLX(const DisplayScreenPair& altDisplayData) {
  if (_glxewInitialized) {
    return;
  }

  Display* dpy = nullptr;
  int screen = -1;

  if (_dpyConnection.first && _dpyConnection.second >= 0) {
    dpy = _dpyConnection.first.get();
    screen = _dpyConnection.second;
  } else if (altDisplayData.first && altDisplayData.second >= 0) {
    dpy = altDisplayData.first.get();
    screen = altDisplayData.second;
  }

  CHECK(dpy != nullptr);

  int major, event, error;
  if (!XQueryExtension(dpy, "GLX", &major, &event, &error)) {
    THROW_RUNTIME_EX("Error: GlxGLWindow: GLX not found!")
  }

  //----- Create and make current a temporary GL context to initialize GLXEW

  // visual
  std::vector<int> attributes;
  attributes.push_back(GLX_RGBA);
  attributes.push_back(None);

  std::unique_ptr<XVisualInfo, int (*)(void*)> visualInfoPtr(glXChooseVisual(dpy, screen, &attributes.front()), XFree);
  RUNTIME_EX_ASSERT(visualInfoPtr != nullptr,
                    "Error: GlxGLRenderer: GLX initialization failed. Basic RGBA pixel format not found.");
  XVisualInfo* visualInfo = visualInfoPtr.get();

  // context
  GLXContext context = glXCreateContext(dpy, visualInfo, 0, True);
  RUNTIME_EX_ASSERT(context != 0,
                    "Error: GlxGLRenderer: GLX initialization failed. Failed to create temporary GLX context.");

  // window
  const X11ID parent = RootWindow(dpy, screen);
  XSetWindowAttributes wa;
  wa.colormap = XCreateColormap(dpy, parent, visualInfo->visual, AllocNone);
  wa.background_pixmap = None;
  wa.border_pixel = 0;
  X11ID drawable = XCreateWindow(dpy,
                                 parent,
                                 0,
                                 0,
                                 16,
                                 16,
                                 0,
                                 visualInfo->depth,
                                 InputOutput,
                                 visualInfo->visual,
                                 CWBackPixmap | CWBorderPixel | CWColormap,
                                 &wa);
  RUNTIME_EX_ASSERT(drawable != 0,
                    "Error: GlxGLRenderer: GLX initialization failed. Failed to create a temporary window.");

  XSync(dpy, False);

  GLRenderer* currRenderer = GLRenderer::getCurrentThreadRenderer();

  glXMakeCurrent(dpy, drawable, context);

// now initialize glxew

#ifdef GLEW_MX
  const GLenum result = glxewInit();
#else
  const GLenum result = glewInit();
#endif

  bool success = result == GLEW_OK;
  if (success) {
    LOG(INFO) << "GLXEW initialization successful";
  } else {
    THROW_RUNTIME_EX("Error: GlxGLRenderer: GLX initialization failed. glxewInit() failed.");
  }

  XSync(dpy, False);
  glXDestroyContext(dpy, context);
  XDestroyWindow(dpy, drawable);

  // clear out the current context since we've messed with it here
  // TODO(croot): should we preserve the context in the state it
  // was previously? Also, is it possible by destroying the tmp context
  // above will restore the previous context state?

  // For some reason the glXMakeCurrent() call below caused a BadMatch (invalid parameter attributes)
  // error, so commenting it out for now.
  // glXMakeCurrent(dpy, None, nullptr);
  if (currRenderer) {
    currRenderer->makeInactive();
  }

  _glxewInitialized = true;
}

FbConfigShPtr GlxGLRenderer::_chooseFbConfig(const GlxGLWindow* primaryWindow) {
  CHECK(_dpyConnection.first != nullptr);
  Display* dpy = _dpyConnection.first.get();

  RUNTIME_EX_ASSERT(GLXEW_VERSION_1_3 || GLXEW_SGIX_fbconfig,
                    "Error: GlxGLRenderer: GLX framebuffer config support required!");

  std::vector<int> attributes;

  // TODO(croot): retrieve render type from settings or argument

  const IntConstant drawableType = primaryWindow->getIntSettingAsConstant(IntSetting::DRAWABLE_TYPE);

  // set up attributes for the color and ancillary buffers
  appendFbBufferAttrs(this, drawableType, _settings, attributes);

  // the attribute list needs to always end with None
  attributes.push_back(None);

  // backoff list? See: Equalizer:src/eq/glx/Window.cpp, lines: 289-302

  PFNGLXCHOOSEFBCONFIGSGIXPROC chooseFBConfig = GLXEW_VERSION_1_3 ? glXChooseFBConfig : glXChooseFBConfigSGIX;

  const int screen = _dpyConnection.second;

  int nConfigs = 0;

  // make a unique ptr to the config ptrs created in the above chooseFBConfig() call
  // to ensure it gets freed at all possible return locations.
  FbConfigShPtr fbConfigPtr(chooseFBConfig(dpy, screen, &attributes[0], &nConfigs), XFree);

  // if couldn't find configs, remove backoff attrs 1 by 1, looking for configs that may work
  // See: Equalizer:src/eq/glx/Window.cpp, lines: 312-324

  RUNTIME_EX_ASSERT(
      nConfigs != 0,
      "Error: GlxGLRenderer: cannot find a framebuffer configuration matching the window settings supplied.");

  return fbConfigPtr;
}

void GlxGLRenderer::initializeGL() {
  CHECK(!isInitialized());

  Window* primaryWindow = getPrimaryWindow();
  CHECK(primaryWindow);
  GlxGLWindow* primaryGlxWindow = dynamic_cast<GlxGLWindow*>(primaryWindow);
  CHECK(primaryGlxWindow);

  _initXDisplay(primaryGlxWindow);
  _initGLX();

  _fbConfigPtr = _chooseFbConfig(primaryGlxWindow);

  Display* dpy = _dpyConnection.first.get();

  CHECK(_fbConfigPtr != nullptr);
  GLXFBConfig* fbConfig = _fbConfigPtr.get();

  // now create the GLX context
  // TODO(croot): deal with context sharing appropriately ....
  GLXContext shCtx = 0;
  // if (_parentWindowPtr->hasParent()) {
  //   const GlxGLWindow* prntGlxWindow = dynamic_cast<const GlxGLWindow*>(_prnt.get());
  //   RUNTIME_EX_ASSERT(prntGlxWindow != nullptr,
  //                     "Error: GlxGLWindow: Cannot create shared GLX context. Parent window is not a GLX window.");
  //   shCtx = prntGlxWindow->getGLXContext();
  // }

  int renderType = GLX_RGBA_TYPE;

  IntConstant drawableType = primaryGlxWindow->getIntSettingAsConstant(IntSetting::DRAWABLE_TYPE);
  IntConstant colorType = _settings.getIntSettingAsConstant(IntSetting::BITS_RGBA);
  if (drawableType == IntConstant::PBUFFER &&
      (colorType == IntConstant::RGBA16F || colorType == IntConstant::RGBA32F)) {
#ifdef GLX_RGBA_FLOAT_TYPE_ARB
    renderType = GLX_RGBA_FLOAT_TYPE_ARB;
#else
    renderType = GLX_RGBA_FLOAT_TYPE;
#endif  // GLX_RGBA_FLOAT_TYPE_ARB
  }

  GLXContext glxContext = 0;

  bool useCoreProfile = isSettingActive(_settings, IntSetting::USE_CORE_PROFILE, true);

  if (glXCreateContextAttribsARB && useCoreProfile) {
    LOG(INFO) << "Renderer Setting: " << IntSetting::USE_CORE_PROFILE << ": True.";

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
          break;
        }

        // let pass thru
        LOG(WARNING) << "Invalid value for " << IntSetting::OPENGL_MAJOR << ". Using default (highest supported "
                                                                            "version).";

      case IntConstant::UNDEFINED:
      case IntConstant::DEFAULT:
      case IntConstant::AUTO:
      case IntConstant::ON:
      case IntConstant::OFF: {
        // create a temporary context to get gl info
        GLXContext tmpCtx = glXCreateContextAttribsARB(dpy, fbConfig[0], shCtx, True, NULL);

        GLRenderer* currRenderer = GLRenderer::getCurrentThreadRenderer();

        // TODO(croot): should we check for the current bound context and set it back
        // to the original when we're done?
        makeWindowAndContextCurrent(primaryWindow, _dpyConnection.first, tmpCtx);

        const char* glVersion = (const char*)glGetString(GL_VERSION);
        if (glVersion) {
          // TODO(croot): store the max glVersion somewhere?
          sscanf(glVersion, "%d.%d", &majorVersion, &minorVersion);

          // TODO(croo): check that the previous sscanf
          // always gets
          // an appropriate major and minor
        }

        // TODO(croot): should we set back to whatever the current context
        // was prior to this code? Not sure that's necessary because
        // we set this soon to be build context current upon building
        _makeInactiveImpl();
        if (currRenderer) {
          currRenderer->makeInactive();
        }

        glXDestroyContext(dpy, tmpCtx);
      } break;
    }

    LOG(INFO) << "Renderer Setting: <" << IntSetting::OPENGL_MAJOR << ">.<" << IntSetting::OPENGL_MINOR
              << ">: " << majorVersion << "." << minorVersion << ".";
    std::vector<int> ctxAttributes = {
        GLX_CONTEXT_MAJOR_VERSION_ARB,
        majorVersion,
        GLX_CONTEXT_MINOR_VERSION_ARB,
        minorVersion,
        GLX_RENDER_TYPE,
        renderType,
        GLX_CONTEXT_PROFILE_MASK_ARB,
        GLX_CONTEXT_CORE_PROFILE_BIT_ARB,
        None  // always end attribute lists with this
    };

    glxContext = glXCreateContextAttribsARB(dpy, fbConfig[0], shCtx, True, &ctxAttributes[0]);
  } else {
    if (GLXEW_VERSION_1_3) {
      glxContext = glXCreateNewContext(dpy, fbConfig[0], renderType, shCtx, True);
    } else {
      glxContext = glXCreateContextWithConfigSGIX(dpy, fbConfig[0], renderType, shCtx, True);
    }
  }

  // Equalizer does an extra check for darwin if glxContext
  // is undefined at this point.
  // See Equalizer:src/eq/glx/window.cpp, lines 395-428, and
  // references a bug
  // report here:
  // http://xquartz.macosforge.org/trac/ticket/466
  // TODO(croot): verify whether this error still exists

  RUNTIME_EX_ASSERT(glxContext != 0, "Error: GlxGLRenderer: Context creation failed.");

  _glxContext = glxContext;

  _initGLEW(primaryGlxWindow);
}

bool GlxGLRenderer::_verifyWindowIsAttachable(const Window* window) {
  // GlxGLWindow* window = dynamic_cast<GlxGLWindow*>(windowPtr.get());
  return (window != nullptr && dynamic_cast<const GlxGLWindow*>(window) != nullptr);
}

}  // namespace GLX
}  // namespace GL
}  // namespace Rendering

#ifndef RENDERING_GL_RESOURCES_GLRESOURCE_H_
#define RENDERING_GL_RESOURCES_GLRESOURCE_H_

#include "../Types.h"
#include "../../../RenderError.h"
#include "../GLRenderer.h"
#include "Types.h"
#include <glog/logging.h>

namespace Rendering {
namespace GL {
namespace Resources {

class GLResource {
 public:
  virtual ~GLResource() {}

  virtual GLResourceType getResourceType() const = 0;
  virtual GLuint getId() const = 0;
  virtual GLenum getTarget() const = 0;

  bool isUsable() const { return (!_rendererPtr.expired() && _usable); }
  bool isUsableOnThread() const { return (isUsable() && _rendererPtr.lock()->isActiveOnCurrentThread()); }
  Renderer* getRenderer() { return _rendererPtr.lock().get(); }
  GLRenderer* getGLRenderer() { return dynamic_cast<GLRenderer*>(_rendererPtr.lock().get()); }

 protected:
  GLResource(const RendererWkPtr& rendererPtr) : _rendererPtr(), _usable(false) {
    CHECK(!rendererPtr.expired());
    GLRenderer* renderer = dynamic_cast<GLRenderer*>(rendererPtr.lock().get());
    CHECK(renderer != nullptr);
    _rendererPtr = rendererPtr;
  }

  RendererWkPtr _rendererPtr;

  void setUsable() { _usable = true; }
  void setUnusable() { _usable = false; }

  void validateRenderer(GLRenderer* renderer = nullptr, bool checkThread = true) const {
    GLRenderer* currRenderer = renderer;

    RUNTIME_EX_ASSERT(!_rendererPtr.expired(), "The renderer for this resource has been removed. Cannot use resource.");

    // TODO(croot): check that the renderer is a valid renderer for this resource.
    // That means the renderer is the ownere of this resource or is a renderer
    // this resource is shared with.
    if (!currRenderer) {
      currRenderer = dynamic_cast<GLRenderer*>(_rendererPtr.lock().get());
    }

    if (checkThread) {
      RUNTIME_EX_ASSERT(
          currRenderer->isActiveOnCurrentThread(),
          "The renderer for the resource hasn't been properly activated on the current thread. Cannot use resource.");
    }
  }

  void validateUsability(GLRenderer* renderer = nullptr, bool checkThread = true) const {
    validateRenderer(renderer, checkThread);
    RUNTIME_EX_ASSERT(_usable == true,
                      "This resource hasn't been properly setup. Cannot use resource. This could be a result of the "
                      "resource being uninitialized, an error occurred during initialization, or the resource was "
                      "cleaned up by an outside source.");
  }

  // void validateBound(GLRenderer* renderer = nullptr, bool checkThread = true) {
  //   validateUsability(renderer, checkThread);
  //   RUNTIME_EX_ASSERT(renderer->getBoundResource(getResourceType()) != this, "This resource hasn't been properly
  //   bound to the renderer.");
  // }

  void cleanupResource() {
    if (isUsableOnThread()) {
      _cleanupResource();
    } else if (isUsable()) {
      // TODO(croot): do we automatically put the context on the current thread
      // to do a proper cleanup?
      THROW_RUNTIME_EX("Trying to clean up resource, but render context isn't activated on the current thread.");
    } else {
      _makeEmpty();
    }

    setUnusable();
  }

 private:
  virtual void bindToRenderer(GLRenderer* renderer) {
    THROW_RUNTIME_EX("The resource's generic bindToResource() method is not defined.");
  }

  virtual void _cleanupResource() = 0;

  /**
   * Only called in the case where the owner of this object hangs onto it longer than the renderer.
   * In other words, the owning renderer was deleted first. So this function should just invalidate
   * any member variables that need to be emptied as a result.
   */
  virtual void _makeEmpty() = 0;

  bool _usable;

  friend class ::Rendering::GL::State::GLBindState;
  friend class ::Rendering::GL::GLResourceManager;
};

}  // namespace Resource
}  // namespace GL
}  // namespace Rendering

#endif  // RENDERING_GL_RESOURCES_GLRESOURCE_H_

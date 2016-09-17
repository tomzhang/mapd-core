#ifndef RENDERING_GL_RESOURCES_GLRESOURCE_H_
#define RENDERING_GL_RESOURCES_GLRESOURCE_H_

#include "../Types.h"
#include "../../../RenderError.h"
#include "../GLRenderer.h"
#include "../../../Window.h"
#include "Types.h"
#include <thread>
#include <glog/logging.h>

namespace Rendering {
namespace GL {
namespace Resources {

class GLResource {
 public:
  virtual ~GLResource() {}

  GLResourceType getResourceType() const { return _rsrcType; }
  UniqueResourceId getRsrcId() const;
  virtual GLuint getId() const = 0;
  virtual GLenum getTarget() const = 0;

  bool isUsable() const { return (!_rendererPtr.expired() && _usable); }
  bool isUsableOnThread() const {
    return (isUsable() && !_rendererPtr.expired() && _rendererPtr.lock()->isActiveOnCurrentThread());
  }
  const Renderer* getRenderer() const;
  Renderer* getRenderer();
  GLRenderer* getGLRenderer();
  GLRendererShPtr getGLRendererShPtr();
  GLResourceManagerShPtr getGLResourceManager();

 protected:
  GLResource(const RendererWkPtr& rendererPtr, GLResourceType rsrcType);

  RendererWkPtr _rendererPtr;

  void setUsable() { _usable = true; }
  void setUnusable() { _usable = false; }

  void validateRenderer(const char* filename,
                        int lineno,
                        GLRenderer* renderer = nullptr,
                        bool checkThread = true) const;

  void validateUsability(const char* filename,
                         int lineno,
                         GLRenderer* renderer = nullptr,
                         bool checkThread = true) const;

  void cleanupResource();

  const GLResourceShPtr getSharedResourceFromPtr(const GLResource* rsrc);

  template <typename GLResourceType>
  const std::shared_ptr<GLResourceType> getSharedResourceFromTypePtr(const GLResourceType* rsrc) {
    return std::dynamic_pointer_cast<GLResourceType>(getSharedResourceFromPtr(rsrc));
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
  GLResourceType _rsrcType;
  ResourceId _rsrcId;

  friend class ::Rendering::GL::State::GLBindState;
  friend class ::Rendering::GL::GLResourceManager;
};

}  // namespace Resource
}  // namespace GL
}  // namespace Rendering

#endif  // RENDERING_GL_RESOURCES_GLRESOURCE_H_

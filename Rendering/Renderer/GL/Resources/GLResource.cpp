#include "GLResource.h"
#include "../GLResourceManager.h"

namespace Rendering {
namespace GL {
namespace Resources {

GLResource::GLResource(const RendererWkPtr& rendererPtr, GLResourceType rsrcType)
    : _rendererPtr(), _usable(false), _rsrcType(rsrcType) {
  CHECK(!rendererPtr.expired());
  GLRenderer* renderer = dynamic_cast<GLRenderer*>(rendererPtr.lock().get());
  CHECK(renderer != nullptr);
  _rendererPtr = rendererPtr;
}

UniqueResourceId GLResource::getRsrcId() const {
  auto rndr = getRenderer();
  RUNTIME_EX_ASSERT(rndr != nullptr, "The renderer for the resource has been deleted. Cannot retrieve its unique id.");
  return std::make_pair(rndr->getId(), _rsrcId);
}

const Renderer* GLResource::getRenderer() const {
  if (_rendererPtr.expired()) {
    return nullptr;
  }
  return _rendererPtr.lock().get();
}

Renderer* GLResource::getRenderer() {
  return const_cast<Renderer*>(static_cast<const GLResource&>(*this).getRenderer());
}

GLRenderer* GLResource::getGLRenderer() {
  if (_rendererPtr.expired()) {
    return nullptr;
  }
  return dynamic_cast<GLRenderer*>(_rendererPtr.lock().get());
}

GLRendererShPtr GLResource::getGLRendererShPtr() {
  if (_rendererPtr.expired()) {
    return nullptr;
  }
  return std::dynamic_pointer_cast<GLRenderer>(_rendererPtr.lock());
}

GLResourceManagerShPtr GLResource::getGLResourceManager() {
  auto renderer = getGLRenderer();
  if (renderer) {
    return renderer->getResourceManager();
  }
  return nullptr;
}

void GLResource::validateRenderer(const char* filename, int lineno, GLRenderer* renderer, bool checkThread) const {
  GLRenderer* currRenderer = renderer;

  RUNTIME_EX_ASSERT(!_rendererPtr.expired(),
                    std::string(filename) + ":" + std::to_string(lineno) + " The renderer for a resource of type " +
                        to_string(getResourceType()) + " has been removed. Cannot use resource.");

  // TODO(croot): check that the renderer is a valid renderer for this resource.
  // That means the renderer is the ownere of this resource or is a renderer
  // this resource is shared with.
  if (!currRenderer) {
    currRenderer = dynamic_cast<GLRenderer*>(_rendererPtr.lock().get());
  }
  CHECK(currRenderer);

  if (checkThread && !currRenderer->isActiveOnCurrentThread()) {
    Window* window = currRenderer->getPrimaryWindow();
    GLRenderer* currentRenderer = GLRenderer::getCurrentThreadRenderer();
    Window* currentWindow = GLRenderer::getCurrentThreadWindow();

    std::stringstream ss;

    ss << filename << ":" << lineno << " The renderer for the resource of type " << getResourceType()
       << " hasn't been properly activated on the current thread - threadId: " << std::this_thread::get_id()
       << " Cannot use resource. Renderer : " << std::hex << currRenderer << " - gpuId : " << currRenderer->getGpuId()
       << ", window: " << std::hex << window << " - name: " << window->getName()
       << ". Current active renderer: " << std::hex << currentRenderer
       << " - gpuId: " << (currentRenderer ? currentRenderer->getGpuId() : -1) << ", active window: " << std::hex
       << currentWindow << " - name: " << (currentWindow ? currentWindow->getName() : "undefined");

    THROW_RUNTIME_EX(ss.str());
  }
}

void GLResource::validateUsability(const char* filename, int lineno, GLRenderer* renderer, bool checkThread) const {
  validateRenderer(filename, lineno, renderer, checkThread);
  RUNTIME_EX_ASSERT(_usable == true,
                    std::string(filename) + ":" + std::to_string(lineno) + " The resource of type " +
                        to_string(getResourceType()) +
                        " is not usable. Cannot use resource. This could be a result of the "
                        "resource being uninitialized, an error occurred during initialization, or the resource was "
                        "cleaned up or flagged unusable by an outside source.");
}

void GLResource::cleanupResource() {
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

const GLResourceShPtr GLResource::getSharedResourceFromPtr(const GLResource* rsrc) {
  auto rsrcMgr = getGLResourceManager();
  if (!rsrcMgr) {
    return nullptr;
  }
  return rsrcMgr->getSharedResourcePtr(rsrc);
}

}  // namespace Resource
}  // namespace GL
}  // namespace Rendering

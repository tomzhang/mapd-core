#ifndef RENDERING_GL_RESOURCES_GLVERTEXARRAYRESOURCE_H_
#define RENDERING_GL_RESOURCES_GLVERTEXARRAYRESOURCE_H_

#include "GLVertexArray.h"
#include "../TypeGL.h"

namespace Rendering {
namespace GL {
namespace Resources {

template <class... Args>
class GLVertexArrayResource {
 public:
  virtual ~GLVertexArrayResource() { _cleanupResource(); }

 protected:
  void _cleanupResource() { _deleteAllVertexArrays(); }

  size_t _vaoSize() {
    _cleanupVaoRefs();
    return _vaoRefs.size();
  }

  bool _validateOneVaoInUse(Args... args) {
    _cleanupVaoRefs();
    auto sz = _vaoRefs.size();
    CHECK(!sz || !_vaoRefs.begin()->second.expired());
    return sz == 0 || (sz == 1 && _doesVaoUseThisResource(_vaoRefs.begin()->second.lock(), args...));
  }

  void _markVertexArraysDirty(Args... args) {
    GLVertexArrayShPtr vaoPtr;
    std::vector<GLVertexArray*> vaosToDelete;
    for (auto& vao : _vaoRefs) {
      vaoPtr = vao.second.lock();
      if (vaoPtr && _doesVaoUseThisResource(vaoPtr, args...)) {
        _setVaoDirtyFlag(vaoPtr, true);
      } else {
        vaosToDelete.push_back(vao.first);
      }
    }

    for (auto& vaoPtr : vaosToDelete) {
      _vaoRefs.erase(vaoPtr);
    }
  }

  void _cleanupVaoRefs() {
    GLVertexArrayShPtr vaoPtr;
    std::vector<GLVertexArray*> vaosToDelete;
    for (auto& vao : _vaoRefs) {
      if (vao.second.expired()) {
        vaosToDelete.push_back(vao.first);
      }
    }

    for (auto& vaos : vaosToDelete) {
      _vaoRefs.erase(vaos);
    }
  }

 private:
  virtual bool _doesVaoUseThisResource(const GLVertexArrayShPtr& vao, Args... args) = 0;
  virtual void _setVaoDirtyFlag(GLVertexArrayShPtr& vao, const bool dirtyFlag) = 0;
  void _addVertexArray(GLVertexArrayShPtr& vao) { _vaoRefs.emplace(vao.get(), vao); }
  void _deleteVertexArray(GLVertexArray* vao) {
    _cleanupVaoRefs();
    _vaoRefs.erase(vao);
  }

  void _deleteAllVertexArrays() {
    GLVertexArrayShPtr vaoPtr;

    for (auto& vao : _vaoRefs) {
      vaoPtr = vao.second.lock();
      if (vaoPtr) {
        _setVaoDirtyFlag(vaoPtr, true);
      }
    }

    _vaoRefs.clear();
  }

  std::map<GLVertexArray*, GLVertexArrayWkPtr> _vaoRefs;

  friend class ::Rendering::GL::GLResourceManager;
  friend class ::Rendering::GL::Resources::GLVertexArray;
};

}  // namespace Resources
}  // namespace GL
}  // namespace Rendering

#endif  // RENDERING_GL_RESOURCES_GLVERTEXARRAYRESOURCE_H_

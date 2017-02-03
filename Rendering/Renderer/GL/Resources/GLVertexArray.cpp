#include "GLVertexArray.h"
#include "GLVertexBuffer.h"
#include "GLIndexBuffer.h"
#include "../GLResourceManager.h"

namespace Rendering {
namespace GL {
namespace Resources {

GLVertexArray::GLVertexArray(const RendererWkPtr& rendererPtr)
    : GLResource(rendererPtr, GLResourceType::VERTEXARRAY),
      _vao(0),
      _dirty(false),
      _numVertices(0),
      _useIbo(false),
      _usedVbos() {
  _initResource();
}

GLVertexArray::GLVertexArray(const RendererWkPtr& rendererPtr,
                             const VboAttrToShaderAttrMap& vboAttrToShaderAttrMap,
                             const GLIndexBufferShPtr& iboPtr)
    : GLVertexArray(rendererPtr) {
  _initialize(vboAttrToShaderAttrMap, iboPtr, getGLRenderer());
}

GLVertexArray::GLVertexArray(const RendererWkPtr& rendererPtr,
                             const VboLayoutAttrToShaderAttrMap& vboLayoutAttrToShaderAttrMap,
                             const GLIndexBufferShPtr& iboPtr)
    : GLVertexArray(rendererPtr) {
  _initialize(vboLayoutAttrToShaderAttrMap, iboPtr, getGLRenderer());
}

GLVertexArray::~GLVertexArray() {
  cleanupResource();
}

void GLVertexArray::_initResource() {
  validateRenderer(__FILE__, __LINE__);

  if (!_vao) {
    MAPD_CHECK_GL_ERROR(glGenVertexArrays(1, &_vao));
  }

  // NOTE: not setting usable yet. It should be set as usable
  // when vertex attributes are added/enabled.
}

void GLVertexArray::_cleanupResource() {
  if (_vao) {
    MAPD_CHECK_GL_ERROR(glDeleteVertexArrays(1, &_vao));
  }
}

void GLVertexArray::_makeEmpty() {
  _vao = 0;
  _clearCaches();
}

void GLVertexArray::_clearCaches() {
  _deleteAllBufferRsrcs();

  _numVertices = 0;
  _numVerticesVbo.first.reset();
  _numVerticesVbo.second.reset();

  _boundVboPtr.first.reset();
  _boundVboPtr.second.reset();

  _useIbo = false;
  _boundIboPtr.reset();
}

void GLVertexArray::validateBuffers() {
  if (!_dirty) {
    return;
  }

  bool sync = false;
  std::pair<size_t, size_t> layoutData;
  GLBufferLayoutShPtr layoutToUse;
  for (auto& vboDataItem : _usedVbos) {
    std::vector<std::pair<GLBufferLayoutShPtr, GLBufferLayoutShPtr>> bufferReplacements;
    auto vbo = vboDataItem.second.vboPtr.lock();
    if (!vbo) {
      THROW_RUNTIME_EX("A vbo used by vertex array " + std::to_string(getId()) +
                       " has been removed. The vertex array is now unusable. The vertex "
                       "array will need to be re-initialized to be usable.");

      setUnusable();
      break;
    }

    for (auto& boundLayoutData : vboDataItem.second.layouts) {
      if (!vbo->hasBufferLayout(boundLayoutData.first)) {
        layoutToUse = vbo->getBufferLayoutAtOffset(boundLayoutData.second.second);
        if (*layoutToUse != *boundLayoutData.first) {
          THROW_RUNTIME_EX("A buffer layout that is used by vertex array " + ::Rendering::GL::to_string(getRsrcId()) +
                           " is no longer attached to the vertex buffer associated with it. The vertex array is now "
                           "unusable. The vertex array will need to be re-initialized to be usable.");

          setUnusable();
          break;
        }
        bufferReplacements.emplace_back(std::make_pair(boundLayoutData.first, layoutToUse));
      } else {
        layoutToUse = boundLayoutData.first;
      }

      layoutData = vbo->getBufferLayoutData(layoutToUse);
      if (layoutData.second != boundLayoutData.second.second) {
        THROW_RUNTIME_EX("A buffer layout used by vertex array " + ::Rendering::GL::to_string(getRsrcId()) +
                         " has had its offset changed. The vertex array is now "
                         "unusable. The vertex array will need to be re-initialized to be usable.");

        setUnusable();
        break;
      }

      if (layoutData.first != boundLayoutData.second.first) {
        auto numVerts = vbo->numVertices(layoutToUse);

        // TODO(croot): these lock() calls can probably be improved by instead
        // doing owner_less calls as the lock call constructs a new object
        if (_numVerticesVbo.first.lock() == vbo && _numVerticesVbo.second.lock() == boundLayoutData.first) {
          if (numVerts < _numVertices) {
            _numVertices = numVerts;
          } else {
            sync = true;
          }
          _numVerticesVbo.second = layoutToUse;
        } else if (numVerts < _numVertices) {
          _numVertices = numVerts;
          _numVerticesVbo.first = vbo;
          _numVerticesVbo.second = layoutToUse;
        }

        boundLayoutData.second.first = layoutData.first;
      }
    }

    for (auto replacePair : bufferReplacements) {
      auto itr = vboDataItem.second.layouts.find(replacePair.first);
      vboDataItem.second.layouts.emplace(replacePair.second, itr->second);
      vboDataItem.second.layouts.erase(itr);
    }
  }

  if (_useIbo && _boundIboPtr.expired()) {
    THROW_RUNTIME_EX("The ibo used by vertex array " + std::to_string(getId()) +
                     " has been removed. The vertex array is now unusable. The vertex array will need to be "
                     "re-initialized to be usable.");

    setUnusable();
  }

  if (sync) {
    _syncWithVBOs();
  }

  _setDirtyFlag(false);
}

bool GLVertexArray::hasVbo(const GLVertexBufferShPtr& vbo, const GLBufferLayoutShPtr& layout) const {
  if (!vbo) {
    return false;
  }

  return hasVbo(vbo.get(), layout);
}

bool GLVertexArray::hasVbo(const GLVertexBuffer* vbo, const GLBufferLayoutShPtr& layout) const {
  auto itr = _usedVbos.find(vbo);
  if (itr != _usedVbos.end() && !itr->second.vboPtr.expired()) {
    if (!layout) {
      return true;
    }

    return (itr->second.layouts.find(layout) != itr->second.layouts.end());
  }
  return false;
}

bool GLVertexArray::hasIbo(const GLIndexBufferShPtr& ibo) const {
  if (!ibo) {
    return false;
  }

  return hasIbo(ibo.get());
}

bool GLVertexArray::hasIbo(const GLIndexBuffer* ibo) const {
  if (_boundIboPtr.expired()) {
    return false;
  }
  auto iboPtr = _boundIboPtr.lock();
  return iboPtr.get() == ibo;
}

size_t GLVertexArray::numIndices() const {
  auto ibo = _boundIboPtr.lock();
  return (ibo ? ibo->numItems() : 0);
}

void GLVertexArray::initialize(const VboAttrToShaderAttrMap& vboAttrToShaderAttrMap,
                               const GLIndexBufferShPtr& iboPtr,
                               GLRenderer* renderer) {
  GLRenderer* rendererToUse = renderer;
  GLRenderer* myRenderer = dynamic_cast<GLRenderer*>(_rendererPtr.lock().get());

  if (!rendererToUse) {
    rendererToUse = myRenderer;
  }

  CHECK(rendererToUse);

  _initialize(vboAttrToShaderAttrMap, iboPtr, rendererToUse);

  auto myRsrc = getSharedResourceFromTypePtr(this);
  CHECK(myRsrc && myRsrc.get() == this);

  for (auto& item : vboAttrToShaderAttrMap) {
    item.first->_addVertexArray(myRsrc);
  }

  if (iboPtr) {
    iboPtr->_addVertexArray(myRsrc);
  }
}

void GLVertexArray::initialize(const VboLayoutAttrToShaderAttrMap& vboLayoutAttrToShaderAttrMap,
                               const GLIndexBufferShPtr& iboPtr,
                               GLRenderer* renderer) {
  GLRenderer* rendererToUse = renderer;
  GLRenderer* myRenderer = dynamic_cast<GLRenderer*>(_rendererPtr.lock().get());

  if (!rendererToUse) {
    rendererToUse = myRenderer;
  }

  CHECK(rendererToUse);

  _initialize(vboLayoutAttrToShaderAttrMap, iboPtr, rendererToUse);

  auto myRsrc = getSharedResourceFromTypePtr(this);
  CHECK(myRsrc && myRsrc.get() == this);

  for (auto& item : vboLayoutAttrToShaderAttrMap) {
    item.first.first->_addVertexArray(myRsrc);
  }

  if (iboPtr) {
    iboPtr->_addVertexArray(myRsrc);
  }
}

void GLVertexArray::_setVboLayoutAttrs(GLShader* shader,
                                       const VboAndLayoutPair& vboLayout,
                                       const VboAttrToShaderAttrList& attrPairs) {
  auto vbo = vboLayout.first;
  auto layout = vboLayout.second;
  RUNTIME_EX_ASSERT(vboLayout.second != nullptr || vboLayout.first->getNumBufferLayouts() == 1,
                    "Cannot initialize vertex array - the vbo must have exactly 1 layout attached, but it has " +
                        std::to_string(vboLayout.first->getNumBufferLayouts()) + " buffer layouts");

  if (layout == nullptr) {
    layout = vbo->getBufferLayoutAtIndex(0);
  }

  if (!attrPairs.size()) {
    // going to attempt to add all attributes in the vbo's layout and
    // assuming that the name of the vbo attribute is the same name/type
    // as the vertex attr in the shader.
    vbo->_bindToShaderInternal(shader, layout);
  } else {
    for (auto& attrPair : attrPairs) {
      vbo->_bindToShaderInternal(shader, layout, attrPair.first, attrPair.second);
    }
  }

  _addVertexBuffer(vbo, layout);

  _boundVboPtr.first = vbo;
  _boundVboPtr.second = layout;
}

std::tuple<GLShaderShPtr, GLint, GLint, GLint> GLVertexArray::_preInitialize(GLRenderer* renderer) {
  // TODO(croot): make this thread safe?
  validateRenderer(__FILE__, __LINE__, renderer);

  RUNTIME_EX_ASSERT(renderer->hasBoundShader(),
                    "Cannot initialize vertex array object. A shader needs to be bound to the renderer for vertex "
                    "array initialization.");

  GLShaderShPtr shaderPtr = renderer->getBoundShader();

  _clearCaches();

  GLint currVao, currVbo, currIbo;
  MAPD_CHECK_GL_ERROR(glGetIntegerv(GL_VERTEX_ARRAY_BINDING, &currVao));
  MAPD_CHECK_GL_ERROR(glGetIntegerv(GL_ARRAY_BUFFER_BINDING, &currVbo));
  MAPD_CHECK_GL_ERROR(glGetIntegerv(GL_ELEMENT_ARRAY_BUFFER_BINDING, &currIbo));

  MAPD_CHECK_GL_ERROR(glBindVertexArray(_vao));

  return std::make_tuple(shaderPtr, currVao, currVbo, currIbo);
}

void GLVertexArray::_postInitialize(const GLIndexBufferShPtr& iboPtr, GLint currVao, GLint currVbo, GLint currIbo) {
  if (iboPtr) {
    _useIbo = true;
    MAPD_CHECK_GL_ERROR(glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, iboPtr->getId()));
  }
  _boundIboPtr = iboPtr;

  if (currVao != static_cast<GLint>(_vao)) {
    MAPD_CHECK_GL_ERROR(glBindVertexArray(currVao));
  }

  MAPD_CHECK_GL_ERROR(glBindBuffer(GL_ARRAY_BUFFER, currVbo));
  MAPD_CHECK_GL_ERROR(glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, currIbo));

  setUsable();
}

void GLVertexArray::_initialize(const VboAttrToShaderAttrMap& vboAttrToShaderAttrMap,
                                const GLIndexBufferShPtr& iboPtr,
                                GLRenderer* renderer) {
  auto preInitData = _preInitialize(renderer);
  auto shader = std::get<0>(preInitData);

  for (auto& item : vboAttrToShaderAttrMap) {
    const GLVertexBufferShPtr& vbo = item.first;
    const VboAttrToShaderAttrList& attrPairs = item.second;

    _setVboLayoutAttrs(shader.get(), std::make_pair(vbo, nullptr), attrPairs);
  }

  _postInitialize(iboPtr, std::get<1>(preInitData), std::get<2>(preInitData), std::get<3>(preInitData));
  _setDirtyFlag(false);
}

void GLVertexArray::_initialize(const VboLayoutAttrToShaderAttrMap& vboLayoutAttrToShaderAttrMap,
                                const GLIndexBufferShPtr& iboPtr,
                                GLRenderer* renderer) {
  auto preInitData = _preInitialize(renderer);
  auto shader = std::get<0>(preInitData);

  for (auto& item : vboLayoutAttrToShaderAttrMap) {
    _setVboLayoutAttrs(shader.get(), item.first, item.second);
  }

  _postInitialize(iboPtr, std::get<1>(preInitData), std::get<2>(preInitData), std::get<3>(preInitData));
  _setDirtyFlag(false);
}

void GLVertexArray::_addVertexBuffer(const GLVertexBufferShPtr& vbo, const GLBufferLayoutShPtr& layout) {
  auto itr = _usedVbos.find(vbo.get());
  auto data = vbo->getBufferLayoutData(layout);
  if (itr != _usedVbos.end()) {
    CHECK(!itr->second.vboPtr.expired());
    itr->second.layouts.insert({layout, std::move(data)});
  } else {
    auto emplacertn =
        _usedVbos.emplace(std::piecewise_construct, std::forward_as_tuple(vbo.get()), std::forward_as_tuple());
    CHECK(emplacertn.second);
    emplacertn.first->second.vboPtr = vbo;
    emplacertn.first->second.layouts.insert({layout, std::move(data)});
  }

  auto numVboVertices = vbo->numVertices(layout);
  if (_numVertices == 0 || numVboVertices < _numVertices) {
    _numVertices = numVboVertices;
    _numVerticesVbo.first = vbo;
    _numVerticesVbo.second = layout;
  }
}

void GLVertexArray::_deleteAllBufferRsrcs() {
  GLVertexBufferShPtr vboPtr;
  for (auto& vboInfo : _usedVbos) {
    vboPtr = vboInfo.second.vboPtr.lock();
    if (vboPtr) {
      vboPtr->_deleteVertexArray(this);
    }
  }
  _usedVbos.clear();
}

void GLVertexArray::_syncWithVBOs() {
  // TODO(croot): We currently don't maintain any synchronization
  // when VBOs change under the hood. This function needs to be
  // called to make sure the VAO stays up-to-date with the vbos
  // it uses. This is kinda ugly. We may want to keep a pointer
  // in the vbo to update any attached VAOs on the fly. That or
  // remove the # of items/vertices in the vao.
  // size_t numVboItems, numVboVertices;
  size_t numVboVertices;

  _numVertices = 0;
  _numVerticesVbo.first.reset();
  _numVerticesVbo.second.reset();

  GLVertexBufferShPtr vboPtr;

  std::vector<const GLVertexBuffer*> vbosToDelete;
  for (auto& vboData : _usedVbos) {
    vboPtr = vboData.second.vboPtr.lock();
    if (vboPtr) {
      for (auto& layoutData : vboData.second.layouts) {
        numVboVertices = vboPtr->numVertices(layoutData.first);

        if (_numVertices == 0 || numVboVertices < _numVertices) {
          _numVertices = numVboVertices;
          _numVerticesVbo.first = vboPtr;
          _numVerticesVbo.second = layoutData.first;
        }
      }
    } else {
      vbosToDelete.push_back(vboData.first);
    }
  }

  for (auto vboData : vbosToDelete) {
    _usedVbos.erase(vboData);
  }
}

}  // namespace Resources
}  // namespace GL
}  // namespace Rendering

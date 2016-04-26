#include "GLVertexArray.h"
#include "GLVertexBuffer.h"
#include "GLIndexBuffer.h"
#include "../GLResourceManager.h"

namespace Rendering {
namespace GL {
namespace Resources {

GLVertexArray::GLVertexArray(const RendererWkPtr& rendererPtr)
    : GLResource(rendererPtr, GLResourceType::VERTEXARRAY), _vao(0), _numItems(0), _numVertices(0), _useIbo(false) {
  _initResource();
}

GLVertexArray::GLVertexArray(const RendererWkPtr& rendererPtr,
                             const VboAttrToShaderAttrMap& vboAttrToShaderAttrMap,
                             const GLIndexBufferShPtr& iboPtr)
    : GLVertexArray(rendererPtr) {
  _initialize(vboAttrToShaderAttrMap, iboPtr, getGLRenderer());
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
  _deleteAllVertexBuffers();
  _numItems = 0;
  _numItemsVbo.reset();
  _numVertices = 0;
  _numVerticesVbo.reset();
}

void GLVertexArray::validateBuffers() {
  for (auto& vbo : _usedVbos) {
    if (vbo.expired()) {
      THROW_RUNTIME_EX("A vbo used by vertex array " + std::to_string(getId()) +
                       " has been removed. The vertex array is now unusable. The vertex "
                       "array will need to be re-initialized to be usable.");

      setUnusable();
      break;
    }
  }

  if (_useIbo && _boundIboPtr.expired()) {
    THROW_RUNTIME_EX("The ibo used by vertex array " + std::to_string(getId()) +
                     " has been removed. The vertex array is now unusable. The vertex array will need to be "
                     "re-initialized to be usable.");

    setUnusable();
  }
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
  CHECK(rendererToUse);

  if (!rendererToUse) {
    rendererToUse = myRenderer;
  }

  _initialize(vboAttrToShaderAttrMap, iboPtr, renderer);

  GLResourceManagerShPtr rsrcMgr = myRenderer->getResourceManager();
  GLVertexArrayShPtr myRsrc = std::dynamic_pointer_cast<GLVertexArray>(rsrcMgr->getResourcePtr(this));
  CHECK(myRsrc && myRsrc.get() == this);

  for (auto& item : vboAttrToShaderAttrMap) {
    item.first->_addVertexArray(myRsrc);
  }
}

void GLVertexArray::_initialize(const VboAttrToShaderAttrMap& vboAttrToShaderAttrMap,
                                const GLIndexBufferShPtr& iboPtr,
                                GLRenderer* renderer) {
  // TODO(croot): make this thread safe?
  validateRenderer(__FILE__, __LINE__, renderer);

  RUNTIME_EX_ASSERT(renderer->hasBoundShader(),
                    "Cannot initialize vertex array object. A shader needs to be bound to the renderer for vertex "
                    "array initialization.");

  GLShaderShPtr shaderPtr = renderer->getBoundShader();
  GLShader* shader = shaderPtr.get();

  _deleteAllVertexBuffers();
  _numItems = 0;
  _numItemsVbo.reset();
  _numVertices = 0;
  _numVerticesVbo.reset();

  _boundVboPtr.reset();

  _useIbo = false;
  _boundIboPtr.reset();

  GLint currVao, currVbo, currIbo;
  MAPD_CHECK_GL_ERROR(glGetIntegerv(GL_VERTEX_ARRAY_BINDING, &currVao));
  MAPD_CHECK_GL_ERROR(glGetIntegerv(GL_ARRAY_BUFFER_BINDING, &currVbo));
  MAPD_CHECK_GL_ERROR(glGetIntegerv(GL_ELEMENT_ARRAY_BUFFER_BINDING, &currIbo));

  MAPD_CHECK_GL_ERROR(glBindVertexArray(_vao));

  for (auto& item : vboAttrToShaderAttrMap) {
    const GLVertexBufferShPtr& vbo = item.first;
    const VboAttrToShaderAttrList& attrPairs = item.second;

    if (!attrPairs.size()) {
      // going to attempt to add all attributes in the vbo's layout and
      // assuming that the name of the vbo attribute is the same name/type
      // as the vertex attr in the shader.
      vbo->_bindToShaderInternal(shader);
    } else {
      for (auto& attrPair : attrPairs) {
        vbo->_bindToShaderInternal(shader, attrPair.first, attrPair.second);
      }
    }

    _addVertexBuffer(vbo);
    _boundVboPtr = vbo;
  }

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

void GLVertexArray::_addVertexBuffer(const GLVertexBufferShPtr& vbo) {
  size_t numVboItems, numVboVertices;
  _usedVbos.emplace(vbo);
  numVboItems = vbo->numItems();
  numVboVertices = vbo->numVertices();
  if (_numItems == 0 || numVboItems < _numItems) {
    _numItems = numVboItems;
    _numItemsVbo = vbo;
  }

  if (_numVertices == 0 || numVboVertices < _numVertices) {
    _numVertices = numVboVertices;
    _numVerticesVbo = vbo;
  }
}

void GLVertexArray::_deleteVertexBuffer(GLVertexBuffer* vbo) {
  GLVertexBufferShPtr vboPtr;
  std::vector<GLVertexBufferWkPtr> vbosToDelete;
  bool resync = false;
  for (auto& vboWkPtr : _usedVbos) {
    vboPtr = vboWkPtr.lock();
    if (vboPtr && vboPtr.get() == vbo) {
      vbosToDelete.push_back(vboWkPtr);
      if (vboPtr == _numItemsVbo.lock() || vboPtr == _numVerticesVbo.lock()) {
        resync = true;
      }
    } else if (!vboPtr) {
      vbosToDelete.push_back(vboWkPtr);
    }
  }

  for (auto& vboWkPtr : vbosToDelete) {
    _usedVbos.erase(vboWkPtr);
  }

  if (resync) {
    _syncWithVBOs();
  }
}

void GLVertexArray::_deleteAllVertexBuffers() {
  GLVertexBufferShPtr vboPtr;
  for (auto& vboWkPtr : _usedVbos) {
    vboPtr = vboWkPtr.lock();
    if (vboPtr) {
      vboPtr->_deleteVertexArray(this);
    }
  }
  _usedVbos.clear();
}

void GLVertexArray::_vboUpdated(GLVertexBuffer* vbo) {
  GLVertexBufferShPtr vboPtr;
  for (auto& vboWkPtr : _usedVbos) {
    vboPtr = vboWkPtr.lock();
    if (vboPtr && vboPtr.get() == vbo) {
      break;
    }
    vboPtr = nullptr;
  }

  CHECK(vboPtr);

  size_t numVboItems = vboPtr->numItems();
  size_t numVboVertices = vboPtr->numVertices();
  if ((vboPtr == _numItemsVbo.lock() && numVboItems != _numItems) ||
      (vboPtr == _numVerticesVbo.lock() && numVboVertices != _numVertices)) {
    _syncWithVBOs();
  } else {
    if (_numItems == 0 || numVboItems < _numItems) {
      _numItems = numVboItems;
      _numItemsVbo = vboPtr;
    }

    if (_numVertices == 0 || numVboVertices < _numVertices) {
      _numVertices = numVboVertices;
      _numVerticesVbo = vboPtr;
    }
  }
}

void GLVertexArray::_syncWithVBOs() {
  // TODO(croot): We currently don't maintain any synchronization
  // when VBOs change under the hood. This function needs to be
  // called to make sure the VAO stays up-to-date with the vbos
  // it uses. This is kinda ugly. We may want to keep a pointer
  // in the vbo to update any attached VAOs on the fly. That or
  // remove the # of items/vertices in the vao.
  size_t numVboItems, numVboVertices;
  _numItems = 0;
  _numVertices = 0;
  GLVertexBufferShPtr vboPtr;
  std::vector<GLVertexBufferWkPtr> vbosToDelete;
  for (auto& vbo : _usedVbos) {
    vboPtr = vbo.lock();
    if (vboPtr) {
      numVboItems = vboPtr->numItems();
      numVboVertices = vboPtr->numVertices();
      _numItems = (_numItems == 0 ? numVboItems : std::min(_numItems, numVboItems));
      _numVertices = (_numVertices == 0 ? numVboVertices : std::min(_numVertices, numVboVertices));
    } else {
      vbosToDelete.push_back(vbo);
    }
  }

  for (auto& vbo : vbosToDelete) {
    _usedVbos.erase(vbo);
  }
}

}  // namespace Resources
}  // namespace GL
}  // namespace Rendering

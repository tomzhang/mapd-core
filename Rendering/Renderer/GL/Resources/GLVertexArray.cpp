#include "GLVertexArray.h"
#include "GLVertexBuffer.h"
#include <iostream>

namespace Rendering {
namespace GL {
namespace Resources {

GLVertexArray::GLVertexArray(const RendererWkPtr& rendererPtr)
    : GLResource(rendererPtr), _vao(0), _numItems(0), _numVertices(0) {
  _initResource();
}

GLVertexArray::GLVertexArray(const RendererWkPtr& rendererPtr, const VboAttrToShaderAttrMap& vboAttrToShaderAttrMap)
    : GLVertexArray(rendererPtr) {
  initialize(vboAttrToShaderAttrMap);
}

GLVertexArray::~GLVertexArray() {
  std::cerr << "CROOT - GLVertexArray destructor" << std::endl;
  cleanupResource();
}

void GLVertexArray::_initResource() {
  validateRenderer();

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
  _usedVbos.clear();
  _numItems = 0;
  _numVertices = 0;
}

void GLVertexArray::validateVBOs() {
  for (auto& vbo : _usedVbos) {
    if (vbo.expired()) {
      THROW_RUNTIME_EX(
          "A vbo used by a vertex array has been removed. The vertex array is now unusable. The vertex "
          "array will need to be re-initialized to be usable.");

      setUnusable();
      break;
    }
  }
}

void GLVertexArray::initialize(const VboAttrToShaderAttrMap& vboAttrToShaderAttrMap, GLRenderer* renderer) {
  // TODO(croot): make this thread safe?

  size_t numVboItems, numVboVertices;

  validateRenderer(renderer);

  GLRenderer* rendererToUse = renderer;
  if (!rendererToUse) {
    rendererToUse = dynamic_cast<GLRenderer*>(_rendererPtr.lock().get());
    CHECK(rendererToUse);
  }

  RUNTIME_EX_ASSERT(rendererToUse->hasBoundShader(),
                    "Cannot initialize vertex array object. A shader needs to be bound to the renderer for vertex "
                    "array initialization.");

  GLShaderShPtr shaderPtr = rendererToUse->getBoundShader();
  GLShader* shader = shaderPtr.get();

  _usedVbos.clear();
  _numItems = 0;
  _numVertices = 0;

  GLint currVao, currVbo;
  MAPD_CHECK_GL_ERROR(glGetIntegerv(GL_VERTEX_ARRAY_BINDING, &currVao));
  MAPD_CHECK_GL_ERROR(glGetIntegerv(GL_ARRAY_BUFFER_BINDING, &currVbo));

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

    _usedVbos.emplace(vbo);
    numVboItems = vbo->numItems();
    numVboVertices = vbo->numVertices();
    _numItems = (_numItems == 0 ? numVboItems : std::min(_numItems, numVboItems));
    _numVertices = (_numVertices == 0 ? numVboVertices : std::min(_numVertices, numVboVertices));
  }

  if (currVao != static_cast<GLint>(_vao)) {
    MAPD_CHECK_GL_ERROR(glBindVertexArray(currVao));
  }

  MAPD_CHECK_GL_ERROR(glBindBuffer(GL_ARRAY_BUFFER, currVbo));

  setUsable();
}

}  // namespace Resources
}  // namespace GL
}  // namespace Rendering

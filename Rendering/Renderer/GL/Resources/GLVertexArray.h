#ifndef RENDERING_GL_RESOURCES_GLVERTEXARRAY_H_
#define RENDERING_GL_RESOURCES_GLVERTEXARRAY_H_

#include "Types.h"
#include "GLResource.h"
#include <map>
#include <vector>
#include <string>
#include <set>

namespace Rendering {
namespace GL {
namespace Resources {

typedef std::pair<std::string, std::string> VboAttrToShaderAttrPair;
typedef std::vector<VboAttrToShaderAttrPair> VboAttrToShaderAttrList;
typedef std::map<GLVertexBufferShPtr, VboAttrToShaderAttrList> VboAttrToShaderAttrMap;

class GLVertexArray : public GLResource {
 public:
  ~GLVertexArray();

  size_t numItems() const { return _numItems; }
  size_t numVertices() const { return _numVertices; }

  GLResourceType getResourceType() const final { return GLResourceType::VERTEXARRAY; }
  GLuint getId() const final { return _vao; }
  GLenum getTarget() const final { return GL_VERTEX_ARRAY; }

  void initialize(const VboAttrToShaderAttrMap& vboAttrToShaderAttrMap, GLRenderer* renderer = nullptr);

  void validateVBOs();

 private:
  explicit GLVertexArray(const RendererWkPtr& rendererPtr);
  explicit GLVertexArray(const RendererWkPtr& rendererPtr, const VboAttrToShaderAttrMap& vboAttrToShaderAttrMap);

  GLuint _vao;
  size_t _numItems;
  size_t _numVertices;

  void _initResource();
  void _cleanupResource() final;
  void _makeEmpty() final;

  std::set<GLVertexBufferWkPtr, std::owner_less<GLVertexBufferWkPtr>> _usedVbos;

  friend class ::Rendering::GL::GLResourceManager;
};

}  // namespace Resources
}  // namespace GL
}  // namespace Rendering

#endif  // RENDERING_GL_RESOURCES_GLVERTEXARRAY_H_

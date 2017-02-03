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
typedef std::pair<GLVertexBufferShPtr, GLBufferLayoutShPtr> VboAndLayoutPair;
typedef std::pair<GLVertexBufferWkPtr, GLBufferLayoutWkPtr> VboAndLayoutWkPair;
typedef std::map<VboAndLayoutPair, VboAttrToShaderAttrList> VboLayoutAttrToShaderAttrMap;

class GLVertexArray : public GLResource {
 public:
  ~GLVertexArray();

  size_t numVertices() const { return _numVertices; }
  size_t numIndices() const;

  GLuint getId() const final { return _vao; }
  GLenum getTarget() const final { return GL_VERTEX_ARRAY; }

  void initialize(const VboAttrToShaderAttrMap& vboAttrToShaderAttrMap,
                  const GLIndexBufferShPtr& iboPtr = nullptr,
                  GLRenderer* renderer = nullptr);

  void initialize(const VboLayoutAttrToShaderAttrMap& vboLayoutAttrToShaderAttrMap,
                  const GLIndexBufferShPtr& iboPtr = nullptr,
                  GLRenderer* renderer = nullptr);

  void validateBuffers();

  bool hasVbo(const GLVertexBufferShPtr& vbo, const GLBufferLayoutShPtr& layout) const;
  bool hasVbo(const GLVertexBuffer* vbo, const GLBufferLayoutShPtr& layout) const;
  bool hasIbo(const GLIndexBufferShPtr& ibo) const;
  bool hasIbo(const GLIndexBuffer* ibo) const;
  GLVertexBufferShPtr getLastBoundVbo() const { return _boundVboPtr.first.lock(); }
  GLIndexBufferShPtr getLastBoundIbo() const { return _boundIboPtr.lock(); }

  bool isDirty() const { return _dirty; }

 private:
  explicit GLVertexArray(const RendererWkPtr& rendererPtr);

  explicit GLVertexArray(const RendererWkPtr& rendererPtr,
                         const VboAttrToShaderAttrMap& vboAttrToShaderAttrMap,
                         const GLIndexBufferShPtr& iboPtr = nullptr);

  explicit GLVertexArray(const RendererWkPtr& rendererPtr,
                         const VboLayoutAttrToShaderAttrMap& vboLayoutAttrToShaderAttrMap,
                         const GLIndexBufferShPtr& iboPtr = nullptr);

  GLuint _vao;

  bool _dirty;
  size_t _numVertices;
  VboAndLayoutWkPair _numVerticesVbo;

  VboAndLayoutWkPair _boundVboPtr;

  bool _useIbo;
  GLIndexBufferWkPtr _boundIboPtr;

  void _initResource();
  void _cleanupResource() final;
  void _makeEmpty() final;
  void _clearCaches();

  void _setVboLayoutAttrs(GLShader* shader,
                          const VboAndLayoutPair& vboLayout,
                          const VboAttrToShaderAttrList& attrPairs);

  std::tuple<GLShaderShPtr, GLint, GLint, GLint> _preInitialize(GLRenderer* renderer);
  void _postInitialize(const GLIndexBufferShPtr& iboPtr, GLint currVao, GLint currVbo, GLint currIbo);
  void _initialize(const VboAttrToShaderAttrMap& vboAttrToShaderAttrMap,
                   const GLIndexBufferShPtr& iboPtr,
                   GLRenderer* renderer);

  void _initialize(const VboLayoutAttrToShaderAttrMap& vboLayoutAttrToShaderAttrMap,
                   const GLIndexBufferShPtr& iboPtr,
                   GLRenderer* renderer);

  void _setDirtyFlag(const bool dirtyFlag) { _dirty = dirtyFlag; }
  void _addVertexBuffer(const GLVertexBufferShPtr& vbo, const GLBufferLayoutShPtr& layout);
  void _deleteAllBufferRsrcs();
  void _syncWithVBOs();

  struct VboPtrs {
    GLVertexBufferWkPtr vboPtr;
    std::map<GLBufferLayoutShPtr, std::pair<size_t, size_t>> layouts;
  };

  std::map<const GLVertexBuffer*, VboPtrs> _usedVbos;

  friend class ::Rendering::GL::GLResourceManager;
  friend class ::Rendering::GL::Resources::GLVertexBuffer;
  friend class ::Rendering::GL::Resources::GLIndexBuffer;
};

}  // namespace Resources
}  // namespace GL
}  // namespace Rendering

#endif  // RENDERING_GL_RESOURCES_GLVERTEXARRAY_H_

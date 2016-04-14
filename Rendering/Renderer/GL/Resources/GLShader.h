#ifndef RENDERING_GL_RESOURCES_GLSHADER_H_
#define RENDERING_GL_RESOURCES_GLSHADER_H_

#include "../Types.h"
#include "../../../RenderError.h"
#include "GLResource.h"
#include "Types.h"
#include <GL/glew.h>
#include <string>
#include <unordered_map>
#include <memory>
#include <vector>

namespace Rendering {
namespace GL {

namespace Resources {

namespace detail {

struct AttrInfo {
  GLint type;
  GLint size;
  GLuint location;

  AttrInfo(GLint t, GLint s, GLuint l) : type(t), size(s), location(l) {}
  ~AttrInfo() {}
};

struct UniformAttrInfo : AttrInfo {
  UniformAttrInfo(GLint t, GLint s, GLuint l) : AttrInfo(t, s, l) {}
  virtual void setAttr(const void* data) = 0;
};

struct UniformSamplerAttr : UniformAttrInfo {
  UniformSamplerAttr(GLint t, GLint s, GLuint l, GLenum target, GLint startTxImgUnit = -1);

  void setAttr(const void* data);
  void setTexImgUnit(GLint texImgUnit);

  GLenum target;
  GLint startTexImgUnit;
};

}  // namespace detail

// enum { ATTR_TYPE = 0, ATTR_SIZE, ATTR_LOC };

class GLShader : public GLResource {
 public:
  ~GLShader();

  GLuint getId() const final { return _programId; }
  GLenum getTarget() const final { return GL_SHADER; }

  std::string getVertexSource() const;
  std::string getFragmentSource() const;

  bool hasUniformAttribute(const std::string& attrName);

  GLint getUniformAttributeGLType(const std::string& attrName);

  // TODO(croot): this doesn't work if called like so:
  // shader.setUniformAttribute("attrName", 1+2);
  // Need to find a way to deal with this
  template <typename T>
  void setUniformAttribute(const std::string& attrName, T attrValue) {
    detail::UniformAttrInfo* info = _validateAttr(attrName);

    GLint attrSz = info->size;
    RUNTIME_EX_ASSERT(attrSz == 1,
                      "Uniform attribute \"" + attrName +
                          "\" is not the appropriate size. It is size 1 but should be " + std::to_string(attrSz) + ".");

    // TODO(croot): check type mismatch?
    info->setAttr((void*)&attrValue);
  }

  template <typename T>
  void setUniformAttribute(const std::string& attrName, const std::vector<T>& attrValue) {
    detail::UniformAttrInfo* info = _validateAttr(attrName);

    GLuint attrSz = info->size;
    RUNTIME_EX_ASSERT(attrSz == attrValue.size(),
                      "Uniform attribute: " + attrName + " is not the appropriate size. It is size " +
                          std::to_string(attrValue.size()) + " but should be " + std::to_string(attrSz) + ".");

    // TODO(croot): check type mismatch?re2
    info->setAttr((void*)(&attrValue[0]));
  }

  template <typename T, size_t N>
  void setUniformAttribute(const std::string& attrName, const std::array<T, N>& attrValue) {
    detail::UniformAttrInfo* info = _validateAttr(attrName);

    GLuint attrSz = info->size;
    RUNTIME_EX_ASSERT(attrSz == N,
                      "Uniform attribute: " + attrName + " is not the appropriate size. It is size " +
                          std::to_string(N) + " but should be " + std::to_string(attrSz) + ".");

    // TODO(croot): check type mismatch?
    info->setAttr((void*)(&attrValue[0]));
  }

  void setSamplerAttribute(const std::string& attrName, const GLResourceShPtr& rsrc);

  void setSamplerTextureImageUnit(const std::string& attrName, GLenum startTexImageUnit);

  GLuint getVertexAttributeLocation(const std::string& attrName) const;

 private:
  GLShader(const RendererWkPtr& rendererPtr, const std::string& vertexShaderSrc, const std::string& fragmentShaderSrc);

  typedef std::unordered_map<std::string, std::unique_ptr<detail::AttrInfo>> AttrMap;
  typedef std::unordered_map<std::string, std::unique_ptr<detail::UniformAttrInfo>> UniformAttrMap;

  GLuint _vertShaderId;
  GLuint _fragShaderId;
  GLuint _programId;
  UniformAttrMap _uniformAttrs;
  AttrMap _vertexAttrs;

  void _initResource(const std::string& vertSrc, const std::string& fragSrc);
  void _cleanupResource() final;
  void _makeEmpty() final;
  detail::UniformAttrInfo* _validateAttr(const std::string& attrName);
  detail::UniformSamplerAttr* _validateSamplerAttr(const std::string& attrName);

  friend class ::Rendering::GL::GLResourceManager;
};

}  // namespace Resources

}  // namespace GL
}  // namespace Rendering

#endif  // RENDERING_GL_RESOURCES_GLSHADER_H_

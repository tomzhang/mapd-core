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
  virtual void setAttr(const void* data, GLint dataSize) = 0;
};

struct UniformSamplerAttr : UniformAttrInfo {
  UniformSamplerAttr(GLint t, GLint s, GLuint l, GLenum target, GLint startTxImgUnit = -1);

  void setAttr(const void* data, GLint dataSize) final;
  void setTexImgUnit(GLint texImgUnit);

  GLenum target;
  GLint startTexImgUnit;
};

struct UniformImageLoadStoreAttr : UniformAttrInfo {
  UniformImageLoadStoreAttr(GLint t,
                            GLint s,
                            GLuint l,
                            GLenum format,
                            GLenum access = GL_READ_ONLY,
                            bool isMultisampled = false,
                            GLint startImgUnit = -1);

  void setAttr(const void* data, GLint attrSize) final;
  void setAttr(const void* data, GLint attrSize, bool layered, int layerIdx);
  void setImgUnit(GLint imgUnit, GLint attrSize);

  GLenum format;
  GLenum access;
  bool isMultisampled;
  GLint startImgUnit;
};

struct UniformBlockAttrInfo {
  std::string blockName;
  GLint blockIndex;
  GLint bufferSize;
  GLint bufferBindingIndex;
  std::unordered_map<std::string, std::unique_ptr<UniformAttrInfo>> activeAttrs;
  GLShaderBlockLayoutShPtr blockLayoutPtr;

  UniformBlockAttrInfo(const GLShaderShPtr& shaderPtr,
                       const std::string& blockName,
                       GLint blockIndex,
                       GLint bufferSize,
                       GLint bufferBindingIndex,
                       ShaderBlockLayoutType layoutType = ShaderBlockLayoutType::PACKED);

  void setBufferBinding(GLuint programId, GLint bindingIndex);
  void bindBuffer(GLuint bufferId);
  void bindBuffer(GLuint bufferId, size_t offsetBytes, size_t sizeBytes);

  // void addActiveAttr(const std::string& attrName, GLint type, GLint size, GLuint location, GLuint idx);
  void addActiveAttr(const std::string& attrName, GLint type, GLint size, GLuint idx);
};

struct UniformSubroutineAttrInfo : UniformAttrInfo {
  UniformSubroutineAttrInfo(const std::string& subroutineName,
                            GLint sz,
                            GLuint loc,
                            GLint index,
                            GLenum shaderStage,
                            std::unordered_map<std::string, GLuint> compatibleSubroutines);

  void setAttr(const void* data, GLint attrSize) final;
  GLuint getCompatibleSubroutineIndex(const std::string& compatibleSubroutine);

  std::string subroutineName;
  GLuint index;
  GLenum shaderStage;

  std::unordered_map<std::string, GLuint> compatibleSubroutines;
};

}  // namespace detail

class GLShader : public GLResource {
 public:
  ~GLShader();

  GLuint getId() const final { return _programId; }
  GLenum getTarget() const final { return GL_SHADER; }

  std::string getVertexSource() const;
  std::string getGeometrySource() const;
  std::string getFragmentSource() const;

  bool hasUniformAttribute(const std::string& attrName);
  GLint getUniformAttributeGLType(const std::string& attrName);

  GLShaderBlockLayoutShPtr getBlockLayout(const std::string& blockName) const;

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
    info->setAttr((void*)&attrValue, 1);
  }

  template <typename T>
  void setUniformAttribute(const std::string& attrName, const std::vector<T>& attrValue, bool checkFullSize = true) {
    detail::UniformAttrInfo* info = _validateAttr(attrName);

    GLuint attrSz = info->size;

    RUNTIME_EX_ASSERT((checkFullSize && attrSz == attrValue.size()) || (!checkFullSize && attrValue.size() < attrSz),
                      "Uniform attribute: " + attrName + " is not the appropriate size. It is size " +
                          std::to_string(attrValue.size()) + " but should be " + std::to_string(attrSz) + ".");

    // TODO(croot): check type mismatch?re2
    info->setAttr((void*)(&attrValue[0]), attrValue.size());
  }

  template <typename T, size_t N>
  void setUniformAttribute(const std::string& attrName, const std::array<T, N>& attrValue) {
    detail::UniformAttrInfo* info = _validateAttr(attrName);

    GLuint attrSz = info->size;
    RUNTIME_EX_ASSERT(attrSz == N,
                      "Uniform attribute: " + attrName + " is not the appropriate size. It is size " +
                          std::to_string(N) + " but should be " + std::to_string(attrSz) + ".");

    // TODO(croot): check type mismatch?
    info->setAttr((void*)(&attrValue[0]), N);
  }

  void setSamplerAttribute(const std::string& attrName, const GLResource* rsrc);
  void setSamplerAttribute(const std::string& attrName, const GLResourceShPtr& rsrc);
  void setSamplerTextureImageUnit(const std::string& attrName, GLenum startTexImageUnit);

  // TODO(croot): support mip-maps
  void setImageLoadStoreAttribute(const std::string& attrName, const GLResource* rsrc, int layerIdx = 0);
  void setImageLoadStoreAttribute(const std::string& attrName, const GLResourceShPtr& rsrc, int layerIdx = 0);
  void setImageLoadStoreAttribute(const std::string& attrName, const std::vector<GLTexture2dShPtr>& rsrcs);
  // TODO(croot): match this api with the setSamplerTextureImageUnit??
  // That function uses a GLenum (i.e. GL_TEXTURE0, etc.) to create the
  // image unit. Should we do the same here?
  void setImageLoadStoreImageUnit(const std::string& attrName, int startImgUnit);

  void setSubroutine(const std::string& subroutineAttrName, const std::string& compatibleSubroutineName);
  void setSubroutines(const std::unordered_map<std::string, std::string>& subroutineVals);

  bool hasUniformBlockAttribute(const std::string& attrName);
  void bindUniformBufferToBlock(const std::string& blockName, const GLUniformBufferShPtr& ubo, int idx = -1);

  bool hasVertexAttribute(const std::string& attrName) const;
  GLuint getVertexAttributeLocation(const std::string& attrName) const;

 private:
  GLShader(const RendererWkPtr& rendererPtr);

  typedef std::unordered_map<std::string, std::unique_ptr<detail::AttrInfo>> AttrMap;
  typedef std::unordered_map<std::string, std::unique_ptr<detail::UniformAttrInfo>> UniformAttrMap;
  typedef std::unordered_map<std::string, std::unique_ptr<detail::UniformBlockAttrInfo>> UniformBlockAttrMap;
  typedef std::unordered_map<std::string, std::unique_ptr<detail::UniformSubroutineAttrInfo>> UniformSubroutineAttrMap;

  GLuint _vertShaderId;
  GLuint _geomShaderId;
  GLuint _fragShaderId;
  GLuint _programId;
  UniformAttrMap _uniformAttrs;
  UniformBlockAttrMap _uniformBlockAttrs;
  UniformSubroutineAttrMap _uniformSubroutineAttrs;
  AttrMap _vertexAttrs;

  void _initResource(const std::string& vertSrc, const std::string& fragSrc, const std::string& geomSrc = "");
  void _cleanupResource() final;
  void _makeEmpty() final;
  detail::UniformAttrInfo* _validateAttr(const std::string& attrName);
  detail::UniformSamplerAttr* _validateSamplerAttr(const std::string& attrName);
  detail::UniformImageLoadStoreAttr* _validateImageLoadStoreAttr(const std::string& attrName);
  detail::UniformBlockAttrInfo* _validateBlockAttr(const std::string& blockName,
                                                   const GLUniformBufferShPtr& ubo,
                                                   size_t idx);
  detail::UniformSubroutineAttrInfo* _validateSubroutineAttr(const std::string& attrName);
  std::unordered_map<GLenum, std::vector<GLuint>> _activeSubroutines;

  friend class ::Rendering::GL::GLResourceManager;
};

}  // namespace Resources
}  // namespace GL
}  // namespace Rendering

#endif  // RENDERING_GL_RESOURCES_GLSHADER_H_

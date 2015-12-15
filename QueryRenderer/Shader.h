#ifndef SHADER_H_
#define SHADER_H_

#include "MapDGL.h"
#include "QueryRendererError.h"

#include <GL/glew.h>

#include <string>
#include <unordered_map>
#include <vector>
#include <memory>
#include <iostream>

namespace MapD_Renderer {

// typedef vector<GLuint> AttrInfo;
// template <typename T>
// struct AttrInfo {
//     GLenum type;
//     GLint  size;
//     GLuint loc;
//     void (*setAttrFunc)(GLint, GLsizei, const T *);

//     AttrInfo(const GLenum& t, const GLint& s, const GLuint& l) : type(t), size(s), loc(l) {};
// };

struct AttrInfo {
  GLint type;
  GLint size;
  GLuint location;

  AttrInfo(GLint t, GLint s, GLuint l) : type(t), size(s), location(l){};
  ~AttrInfo() {
    // std::cout << "IN AttrInfo DESTRUCTOR" << std::endl;
  }
};

struct UniformAttrInfo : AttrInfo {
  UniformAttrInfo(GLint t, GLint s, GLuint l) : AttrInfo(t, s, l) {}
  virtual void setAttr(const void* data) = 0;
};

struct Uniform1uiAttr : UniformAttrInfo {
  Uniform1uiAttr(GLint t, GLint s, GLuint l) : UniformAttrInfo(t, s, l) {}
  void setAttr(const void* data) {
    MAPD_CHECK_GL_ERROR(glUniform1uiv(location, size, static_cast<const GLuint*>(data)));
  }
};

struct Uniform2uiAttr : UniformAttrInfo {
  Uniform2uiAttr(GLint t, GLint s, GLuint l) : UniformAttrInfo(t, s, l) {}
  void setAttr(const void* data) {
    MAPD_CHECK_GL_ERROR(glUniform2uiv(location, size, static_cast<const GLuint*>(data)));
  }
};

struct Uniform3uiAttr : UniformAttrInfo {
  Uniform3uiAttr(GLint t, GLint s, GLuint l) : UniformAttrInfo(t, s, l) {}
  void setAttr(const void* data) {
    MAPD_CHECK_GL_ERROR(glUniform3uiv(location, size, static_cast<const GLuint*>(data)));
  }
};

struct Uniform4uiAttr : UniformAttrInfo {
  Uniform4uiAttr(GLint t, GLint s, GLuint l) : UniformAttrInfo(t, s, l) {}
  void setAttr(const void* data) {
    MAPD_CHECK_GL_ERROR(glUniform4uiv(location, size, static_cast<const GLuint*>(data)));
  }
};

struct Uniform1iAttr : UniformAttrInfo {
  Uniform1iAttr(GLint t, GLint s, GLuint l) : UniformAttrInfo(t, s, l) {}
  void setAttr(const void* data) { MAPD_CHECK_GL_ERROR(glUniform1iv(location, size, static_cast<const GLint*>(data))); }
};

struct Uniform2iAttr : UniformAttrInfo {
  Uniform2iAttr(GLint t, GLint s, GLuint l) : UniformAttrInfo(t, s, l) {}
  void setAttr(const void* data) { MAPD_CHECK_GL_ERROR(glUniform2iv(location, size, static_cast<const GLint*>(data))); }
};

struct Uniform3iAttr : UniformAttrInfo {
  Uniform3iAttr(GLint t, GLint s, GLuint l) : UniformAttrInfo(t, s, l) {}
  void setAttr(const void* data) { MAPD_CHECK_GL_ERROR(glUniform3iv(location, size, static_cast<const GLint*>(data))); }
};

struct Uniform4iAttr : UniformAttrInfo {
  Uniform4iAttr(GLint t, GLint s, GLuint l) : UniformAttrInfo(t, s, l) {}
  void setAttr(const void* data) { MAPD_CHECK_GL_ERROR(glUniform4iv(location, size, static_cast<const GLint*>(data))); }
};

struct Uniform1fAttr : UniformAttrInfo {
  Uniform1fAttr(GLint t, GLint s, GLuint l) : UniformAttrInfo(t, s, l) {}
  void setAttr(const void* data) {
    MAPD_CHECK_GL_ERROR(glUniform1fv(location, size, static_cast<const GLfloat*>(data)));
  }
};

struct Uniform2fAttr : UniformAttrInfo {
  Uniform2fAttr(GLint t, GLint s, GLuint l) : UniformAttrInfo(t, s, l) {}
  void setAttr(const void* data) {
    MAPD_CHECK_GL_ERROR(glUniform2fv(location, size, static_cast<const GLfloat*>(data)));
  }
};

struct Uniform3fAttr : UniformAttrInfo {
  Uniform3fAttr(GLint t, GLint s, GLuint l) : UniformAttrInfo(t, s, l) {}
  void setAttr(const void* data) {
    MAPD_CHECK_GL_ERROR(glUniform3fv(location, size, static_cast<const GLfloat*>(data)));
  }
};

struct Uniform4fAttr : UniformAttrInfo {
  Uniform4fAttr(GLint t, GLint s, GLuint l) : UniformAttrInfo(t, s, l) {}
  void setAttr(const void* data) {
    MAPD_CHECK_GL_ERROR(glUniform4fv(location, size, static_cast<const GLfloat*>(data)));
  }
};

struct Uniform1dAttr : UniformAttrInfo {
  Uniform1dAttr(GLint t, GLint s, GLuint l) : UniformAttrInfo(t, s, l) {}
  void setAttr(const void* data) {
    MAPD_CHECK_GL_ERROR(glUniform1dv(location, size, static_cast<const GLdouble*>(data)));
  }
};

struct Uniform2dAttr : UniformAttrInfo {
  Uniform2dAttr(GLint t, GLint s, GLuint l) : UniformAttrInfo(t, s, l) {}
  void setAttr(const void* data) {
    MAPD_CHECK_GL_ERROR(glUniform2dv(location, size, static_cast<const GLdouble*>(data)));
  }
};

struct Uniform3dAttr : UniformAttrInfo {
  Uniform3dAttr(GLint t, GLint s, GLuint l) : UniformAttrInfo(t, s, l) {}
  void setAttr(const void* data) {
    MAPD_CHECK_GL_ERROR(glUniform3dv(location, size, static_cast<const GLdouble*>(data)));
  }
};

struct Uniform4dAttr : UniformAttrInfo {
  Uniform4dAttr(GLint t, GLint s, GLuint l) : UniformAttrInfo(t, s, l) {}
  void setAttr(const void* data) {
    MAPD_CHECK_GL_ERROR(glUniform4dv(location, size, static_cast<const GLdouble*>(data)));
  }
};

enum { ATTR_TYPE = 0, ATTR_SIZE, ATTR_LOC };

class Shader {
 private:
  typedef std::unordered_map<std::string, std::unique_ptr<AttrInfo>> AttrMap;
  typedef std::unordered_map<std::string, std::unique_ptr<UniformAttrInfo>> UniformAttrMap;

  GLuint _vertShaderId;
  GLuint _fragShaderId;
  GLuint _programId;
  UniformAttrMap _uniformAttrs;
  AttrMap _vertexAttrs;

  void _init(const std::string& vertSrc, const std::string& fragSrc);
  void _cleanupIds();

 public:
  Shader(const std::string& vertexShaderSrc, const std::string& fragmentShaderSrc);
  ~Shader();

  std::string getVertexSource() const;
  std::string getFragmentSource() const;

  bool hasUniformAttribute(const std::string& attrName) {
    UniformAttrMap::const_iterator iter = _uniformAttrs.find(attrName);
    return (iter != _uniformAttrs.end());
  }

  GLint getUniformAttributeGLType(const std::string& attrName) {
    UniformAttrMap::const_iterator iter = _uniformAttrs.find(attrName);

    RUNTIME_EX_ASSERT(
        iter != _uniformAttrs.end(),
        "Shader::getUniformAttributeGLType(): uniform attribute \"" + attrName + "\" does not exist in shader.");

    return iter->second->type;
  }

  template <typename T>
  void setUniformAttribute(const std::string& attrName, T attrValue) {
    UniformAttrMap::const_iterator iter = _uniformAttrs.find(attrName);

    // TODO: check if bound

    RUNTIME_EX_ASSERT(iter != _uniformAttrs.end(),
                      "Uniform attribute \"" + attrName + "\" is not defined in the shader.");

    UniformAttrInfo* info = iter->second.get();

    GLint attrSz = info->size;
    RUNTIME_EX_ASSERT(attrSz == 1,
                      "Uniform attribute \"" + attrName +
                          "\" is not the appropriate size. It is size 1 but should be " + std::to_string(attrSz) + ".");

    // TODO: check type mismatch?
    // setUniformByLocation(attrLoc, 1, &attrValue);
    // iter->(*second)();
    info->setAttr((void*)&attrValue);
  }

  template <typename T>
  void setUniformAttribute(const std::string& attrName, const std::vector<T>& attrValue) {
    UniformAttrMap::const_iterator iter = _uniformAttrs.find(attrName);

    // TODO: check if bound

    RUNTIME_EX_ASSERT(iter != _uniformAttrs.end(),
                      "Uniform attribute \"" + attrName + "\" is not defined in the shader.");

    UniformAttrInfo* info = iter->second.get();

    GLuint attrSz = info->size;
    RUNTIME_EX_ASSERT(attrSz == attrValue.size(),
                      "Uniform attribute: " + attrName + " is not the appropriate size. It is size " +
                          std::to_string(attrValue.size()) + " but should be " + std::to_string(attrSz) + ".");

    // TODO: check type mismatch?
    info->setAttr((void*)(&attrValue[0]));
    // setUniformByLocation(attrLoc, attrSz, &attrValue);
  }

  template <typename T, size_t N>
  void setUniformAttribute(const std::string& attrName, const std::array<T, N>& attrValue) {
    UniformAttrMap::const_iterator iter = _uniformAttrs.find(attrName);

    // TODO: check if bound

    RUNTIME_EX_ASSERT(iter != _uniformAttrs.end(),
                      "Uniform attribute \"" + attrName + "\" is not defined in the shader.");

    UniformAttrInfo* info = iter->second.get();

    GLuint attrSz = info->size;
    RUNTIME_EX_ASSERT(attrSz == N,
                      "Uniform attribute: " + attrName + " is not the appropriate size. It is size " +
                          std::to_string(N) + " but should be " + std::to_string(attrSz) + ".");

    // TODO: check type mismatch?
    info->setAttr((void*)(&attrValue[0]));
    // setUniformByLocation(attrLoc, attrSz, &attrValue);
  }

  GLuint getVertexAttributeLocation(const std::string& attrName) const;

  void bindToRenderer() const;
};

typedef std::unique_ptr<Shader> ShaderUqPtr;
typedef std::shared_ptr<Shader> ShaderShPtr;

}  // namespace MapD_Renderer

#endif  // SHADER_H_

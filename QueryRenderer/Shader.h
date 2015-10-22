#ifndef SHADER_H_
#define SHADER_H_

#include <string>
#include <unordered_map>
#include <vector>
#include <GL/glew.h>
#include <memory>
#include <iostream>

using namespace std;

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
  void setAttr(const void* data) { glUniform1uiv(location, size, static_cast<const GLuint*>(data)); }
};

struct Uniform2uiAttr : UniformAttrInfo {
  Uniform2uiAttr(GLint t, GLint s, GLuint l) : UniformAttrInfo(t, s, l) {}
  void setAttr(const void* data) { glUniform2uiv(location, size, static_cast<const GLuint*>(data)); }
};

struct Uniform3uiAttr : UniformAttrInfo {
  Uniform3uiAttr(GLint t, GLint s, GLuint l) : UniformAttrInfo(t, s, l) {}
  void setAttr(const void* data) { glUniform3uiv(location, size, static_cast<const GLuint*>(data)); }
};

struct Uniform4uiAttr : UniformAttrInfo {
  Uniform4uiAttr(GLint t, GLint s, GLuint l) : UniformAttrInfo(t, s, l) {}
  void setAttr(const void* data) { glUniform4uiv(location, size, static_cast<const GLuint*>(data)); }
};

struct Uniform1iAttr : UniformAttrInfo {
  Uniform1iAttr(GLint t, GLint s, GLuint l) : UniformAttrInfo(t, s, l) {}
  void setAttr(const void* data) { glUniform1iv(location, size, static_cast<const GLint*>(data)); }
};

struct Uniform2iAttr : UniformAttrInfo {
  Uniform2iAttr(GLint t, GLint s, GLuint l) : UniformAttrInfo(t, s, l) {}
  void setAttr(const void* data) { glUniform2iv(location, size, static_cast<const GLint*>(data)); }
};

struct Uniform3iAttr : UniformAttrInfo {
  Uniform3iAttr(GLint t, GLint s, GLuint l) : UniformAttrInfo(t, s, l) {}
  void setAttr(const void* data) { glUniform3iv(location, size, static_cast<const GLint*>(data)); }
};

struct Uniform4iAttr : UniformAttrInfo {
  Uniform4iAttr(GLint t, GLint s, GLuint l) : UniformAttrInfo(t, s, l) {}
  void setAttr(const void* data) { glUniform4iv(location, size, static_cast<const GLint*>(data)); }
};

struct Uniform1fAttr : UniformAttrInfo {
  Uniform1fAttr(GLint t, GLint s, GLuint l) : UniformAttrInfo(t, s, l) {}
  void setAttr(const void* data) { glUniform1fv(location, size, static_cast<const GLfloat*>(data)); }
};

struct Uniform2fAttr : UniformAttrInfo {
  Uniform2fAttr(GLint t, GLint s, GLuint l) : UniformAttrInfo(t, s, l) {}
  void setAttr(const void* data) { glUniform2fv(location, size, static_cast<const GLfloat*>(data)); }
};

struct Uniform3fAttr : UniformAttrInfo {
  Uniform3fAttr(GLint t, GLint s, GLuint l) : UniformAttrInfo(t, s, l) {}
  void setAttr(const void* data) { glUniform3fv(location, size, static_cast<const GLfloat*>(data)); }
};

struct Uniform4fAttr : UniformAttrInfo {
  Uniform4fAttr(GLint t, GLint s, GLuint l) : UniformAttrInfo(t, s, l) {}
  void setAttr(const void* data) { glUniform4fv(location, size, static_cast<const GLfloat*>(data)); }
};

enum { ATTR_TYPE = 0, ATTR_SIZE, ATTR_LOC };

class Shader {
 private:
  typedef unordered_map<string, unique_ptr<AttrInfo>> AttrMap;
  typedef unordered_map<string, unique_ptr<UniformAttrInfo>> UniformAttrMap;

  GLuint _vertShaderId;
  GLuint _fragShaderId;
  GLuint _programId;
  UniformAttrMap _uniformAttrs;
  AttrMap _vertexAttrs;

  void _init(const string& vertSrc, const string& fragSrc);
  void _cleanupIds();

 public:
  Shader(const string& vertexShaderSrc, const string& fragmentShaderSrc);
  ~Shader();

  string getVertexSource() const;
  string getFragmentSource() const;

  template <typename T>
  void setUniformAttribute(const string& attrName, T attrValue) {
    UniformAttrMap::const_iterator iter = _uniformAttrs.find(attrName);

    // TODO: check if bound

    if (iter == _uniformAttrs.end()) {
      // TODO: throw a warning/error?
      cerr << "Uniform attribute: " << attrName << " is not defined in the shader." << endl;
      return;
    }

    UniformAttrInfo* info = iter->second.get();

    GLint attrSz = info->size;
    if (attrSz != 1) {
      // TODO: throw a warning/error?
      cerr << "Uniform attribute: " << attrName << " is not the appropriate size. It is size 1 but should be " << attrSz
           << endl;
      return;
    }

    // TODO: check type mismatch?
    // setUniformByLocation(attrLoc, 1, &attrValue);
    // iter->(*second)();
    info->setAttr(attrValue);
  }

  template <typename T>
  void setUniformAttribute(const string& attrName, const vector<T>& attrValue) {
    UniformAttrMap::const_iterator iter = _uniformAttrs.find(attrName);

    // TODO: check if bound

    if (iter == _uniformAttrs.end()) {
      // TODO: throw a warning/error?
      cerr << "Uniform attribute: " << attrName << " is not defined in the shader." << endl;
      return;
    }

    UniformAttrInfo* info = iter->second.get();

    GLuint attrSz = info->size;
    if (attrSz != attrValue.size()) {
      // TODO: throw a warning/error?
      cerr << "Uniform attribute: " << attrName << " is not the appropriate size. It is size " << attrValue.size()
           << " but should be " << attrSz << endl;
      return;
    }

    // TODO: check type mismatch?
    info->setAttr((void*)(&attrValue[0]));
    // setUniformByLocation(attrLoc, attrSz, &attrValue);
  }

  GLuint getVertexAttributeLocation(const string& attrName) const;

  void bindToRenderer() const;
};

typedef std::unique_ptr<Shader> ShaderUqPtr;
typedef std::shared_ptr<Shader> ShaderShPtr;

}  // namespace MapD_Renderer

#endif  // SHADER_H_

#include "Shader.h"
#include <boost/algorithm/string/predicate.hpp>

using namespace MapD_Renderer;

GLint compileShader(const GLuint& shaderId, const std::string& shaderSrc, std::string& errStr) {
  GLint compiled;
  const GLchar* shaderSrcCode = shaderSrc.c_str();

  MAPD_CHECK_GL_ERROR(glShaderSource(shaderId, 1, &shaderSrcCode, NULL));
  MAPD_CHECK_GL_ERROR(glCompileShader(shaderId));
  MAPD_CHECK_GL_ERROR(glGetShaderiv(shaderId, GL_COMPILE_STATUS, &compiled));
  if (!compiled) {
    GLchar errLog[1024];
    MAPD_CHECK_GL_ERROR(glGetShaderInfoLog(shaderId, 1024, NULL, errLog));
    errStr.assign(std::string(errLog));

    // std::ofstream shadersrcstream;
    // shadersrcstream.open("shadersource.vert");
    // shadersrcstream << shadersrc;
    // shadersrcstream.close();
  }

  return compiled;
}

std::string getShaderSource(const GLuint& shaderId) {
  if (!shaderId) {
    return std::string();
  }

  GLint sourceLen;
  MAPD_CHECK_GL_ERROR(glGetShaderiv(shaderId, GL_SHADER_SOURCE_LENGTH, &sourceLen));

  std::unique_ptr<GLchar[]> source(new GLchar[sourceLen]);
  MAPD_CHECK_GL_ERROR(glGetShaderSource(shaderId, sourceLen, NULL, source.get()));

  return std::string(source.get());
}

GLint linkProgram(const GLuint& programId, std::string& errStr) {
  GLint linked;

  MAPD_CHECK_GL_ERROR(glLinkProgram(programId));
  MAPD_CHECK_GL_ERROR(glGetProgramiv(programId, GL_LINK_STATUS, &linked));
  if (!linked) {
    GLchar errLog[1024];
    MAPD_CHECK_GL_ERROR(glGetProgramInfoLog(programId, 1024, NULL, errLog));
    errStr.assign(std::string(errLog));
  }

  return linked;
}

UniformAttrInfo* createUniformAttrInfoPtr(GLint type, GLint size, GLuint location) {
  UniformAttrInfo* rtn = NULL;

  switch (type) {
    case GL_UNSIGNED_INT:
      rtn = new Uniform1uiAttr(type, size, location);
      break;
    case GL_UNSIGNED_INT_VEC2:
      rtn = new Uniform2uiAttr(type, size, location);
      break;
    case GL_UNSIGNED_INT_VEC3:
      rtn = new Uniform3uiAttr(type, size, location);
      break;
    case GL_UNSIGNED_INT_VEC4:
      rtn = new Uniform4uiAttr(type, size, location);
      break;

    case GL_INT:
      rtn = new Uniform1iAttr(type, size, location);
      break;
    case GL_INT_VEC2:
      rtn = new Uniform2iAttr(type, size, location);
      break;
    case GL_INT_VEC3:
      rtn = new Uniform3iAttr(type, size, location);
      break;
    case GL_INT_VEC4:
      rtn = new Uniform4iAttr(type, size, location);
      break;

    case GL_FLOAT:
      rtn = new Uniform1fAttr(type, size, location);
      break;
    case GL_FLOAT_VEC2:
      rtn = new Uniform2fAttr(type, size, location);
      break;
    case GL_FLOAT_VEC3:
      rtn = new Uniform3fAttr(type, size, location);
      break;
    case GL_FLOAT_VEC4:
      rtn = new Uniform4fAttr(type, size, location);
      break;

    case GL_DOUBLE:
      rtn = new Uniform1dAttr(type, size, location);
      break;
    case GL_DOUBLE_VEC2:
      rtn = new Uniform2dAttr(type, size, location);
      break;
    case GL_DOUBLE_VEC3:
      rtn = new Uniform3dAttr(type, size, location);
      break;
    case GL_DOUBLE_VEC4:
      rtn = new Uniform4dAttr(type, size, location);
      break;

    default:
      THROW_RUNTIME_EX("createUniformAttrPtr(): GL type " + std::to_string(type) + " is not yet a supported type.");
      break;
  }

  return rtn;
}

// void setUniformByLocation(const GLuint& loc, const GLuint& sz, const unsigned int *value)
// {
//  switch (sz)
//  {
//    case 1:
//      MAPD_CHECK_GL_ERROR(glUniform1uiv(loc, 1, value));
//      break;
//    case 2:
//      MAPD_CHECK_GL_ERROR(glUniform2uiv(loc, 2, value));
//      break;
//    case 3:
//      MAPD_CHECK_GL_ERROR(glUniform3uiv(loc, 3, value));
//      break;
//    case 4:
//      MAPD_CHECK_GL_ERROR(glUniform4uiv(loc, 4, value));
//      break;
//  }
// }

// void setUniformByLocation(const GLuint& loc, const GLuint& sz, const int *value)
// {
//  switch (sz)
//  {
//    case 1:
//      MAPD_CHECK_GL_ERROR(glUniform1iv(loc, 1, value));
//      break;
//    case 2:
//      MAPD_CHECK_GL_ERROR(glUniform2iv(loc, 2, value));
//      break;
//    case 3:
//      MAPD_CHECK_GL_ERROR(glUniform3iv(loc, 3, value));
//      break;
//    case 4:
//      MAPD_CHECK_GL_ERROR(glUniform4iv(loc, 4, value));
//      break;
//  }
// }

// void setUniformByLocation(const GLuint& loc, const GLuint& sz, const float *value)
// {
//  switch (sz)
//  {
//    case 1:
//      MAPD_CHECK_GL_ERROR(glUniform1fv(loc, 1, value));
//      break;
//    case 2:
//      MAPD_CHECK_GL_ERROR(glUniform2fv(loc, 2, value));
//      break;
//    case 3:
//      MAPD_CHECK_GL_ERROR(glUniform3fv(loc, 3, value));
//      break;
//    case 4:
//      MAPD_CHECK_GL_ERROR(glUniform4fv(loc, 4, value));
//      break;
//  }
// }

Shader::Shader(const std::string& vertexShaderSrc, const std::string& fragmentShaderSrc)
    : _vertShaderId(0), _fragShaderId(0), _programId(0) {
  _init(vertexShaderSrc, fragmentShaderSrc);
}

Shader::~Shader() {
  // std::cerr << "IN Shader DESTRUCTOR" << std::endl;
  _cleanupIds();
}

void Shader::_cleanupIds() {
  if (_vertShaderId) {
    MAPD_CHECK_GL_ERROR(glDeleteShader(_vertShaderId));
  }

  if (_fragShaderId) {
    MAPD_CHECK_GL_ERROR(glDeleteShader(_fragShaderId));
  }

  if (_programId) {
    MAPD_CHECK_GL_ERROR(glDeleteProgram(_programId));
  }
}

void Shader::_init(const std::string& vertSrc, const std::string& fragSrc) {
  std::string errStr;

  // build and compile the vertex shader
  MAPD_CHECK_GL_ERROR((_vertShaderId = glCreateShader(GL_VERTEX_SHADER)));

  if (!compileShader(_vertShaderId, vertSrc, errStr)) {
    _cleanupIds();
    THROW_RUNTIME_EX("Error compiling vertex shader: " + errStr + ".\n\nVertex shader src:\n\n" + vertSrc);
  }

  MAPD_CHECK_GL_ERROR((_fragShaderId = glCreateShader(GL_FRAGMENT_SHADER)));
  if (!compileShader(_fragShaderId, fragSrc, errStr)) {
    _cleanupIds();
    THROW_RUNTIME_EX("Error compiling fragment shader: " + errStr + ".\n\nFragment shader src:\n\n" + fragSrc);
  }

  MAPD_CHECK_GL_ERROR((_programId = glCreateProgram()));
  MAPD_CHECK_GL_ERROR(glAttachShader(_programId, _vertShaderId));
  MAPD_CHECK_GL_ERROR(glAttachShader(_programId, _fragShaderId));
  if (!linkProgram(_programId, errStr)) {
    // clean out the shader references
    _cleanupIds();
    THROW_RUNTIME_EX("Error linking the shader: " + errStr);
  }

  GLint numAttrs;
  GLchar attrName[512];
  GLint attrType;
  // GLint uAttrType;
  GLint attrSz;
  GLuint attrLoc;

  // NOTE: This needs to be improved to handle basic array types, structs,
  // arrays of structs, interface blocks (uniform & shader storage blocks),
  // subroutines, atomic counters, etc.
  // See: https://www.opengl.org/wiki/Program_Introspection

  // setup the uniform attributes
  _uniformAttrs.clear();
  MAPD_CHECK_GL_ERROR(glGetProgramiv(_programId, GL_ACTIVE_UNIFORMS, &numAttrs));
  for (GLuint i = 0; i < static_cast<GLuint>(numAttrs); ++i) {
    MAPD_CHECK_GL_ERROR(glGetActiveUniformName(_programId, i, 512, NULL, attrName));
    std::string attrNameStr(attrName);

    MAPD_CHECK_GL_ERROR(glGetActiveUniformsiv(_programId, 1, &i, GL_UNIFORM_TYPE, &attrType));
    MAPD_CHECK_GL_ERROR(glGetActiveUniformsiv(_programId, 1, &i, GL_UNIFORM_SIZE, &attrSz));
    MAPD_CHECK_GL_ERROR((attrLoc = glGetUniformLocation(_programId, attrName)));

    if (boost::algorithm::ends_with(attrNameStr, "[0]")) {
      attrNameStr.erase(attrNameStr.size() - 3, 3);
    }

    _uniformAttrs.insert(
        make_pair(attrNameStr, std::unique_ptr<UniformAttrInfo>(createUniformAttrInfoPtr(attrType, attrSz, attrLoc))));
  }

  // now setup the vertex attributes
  _vertexAttrs.clear();
  MAPD_CHECK_GL_ERROR(glGetProgramiv(_programId, GL_ACTIVE_ATTRIBUTES, &numAttrs));
  for (int i = 0; i < numAttrs; ++i) {
    MAPD_CHECK_GL_ERROR(glGetActiveAttrib(_programId, i, 512, NULL, &attrSz, (GLenum*)&attrType, attrName));
    MAPD_CHECK_GL_ERROR((attrLoc = glGetAttribLocation(_programId, attrName)));

    _vertexAttrs.insert(
        std::make_pair(std::string(attrName), std::unique_ptr<AttrInfo>(new AttrInfo(attrType, attrSz, attrLoc))));
  }
}

std::string Shader::getVertexSource() const {
  return getShaderSource(_vertShaderId);
}

std::string Shader::getFragmentSource() const {
  return getShaderSource(_fragShaderId);
}

GLuint Shader::getVertexAttributeLocation(const std::string& attrName) const {
  AttrMap::const_iterator iter = _vertexAttrs.find(attrName);

  RUNTIME_EX_ASSERT(iter != _vertexAttrs.end(),
                    "Attribute \"" + attrName + "\" does not exist in shader. Cannot get attribute location.");

  AttrInfo* info = iter->second.get();

  return info->location;
}

void Shader::bindToRenderer() const {
  // TODO: Throw an error or warning if the program
  // is invalid?

  if (_programId) {
    MAPD_CHECK_GL_ERROR(glUseProgram(_programId));
  }
}

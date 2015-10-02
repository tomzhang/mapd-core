#include <assert.h>
#include <iostream>
#include <boost/algorithm/string/predicate.hpp>
#include "Shader.h"

using namespace MapD_Renderer;

GLint compileShader(const GLuint& shaderId, const string& shaderSrc, string& errStr)
{
	GLint compiled;
	const GLchar* shaderSrcCode = shaderSrc.c_str();

	glShaderSource(shaderId, 1, &shaderSrcCode, NULL);
	glCompileShader(shaderId);
	glGetShaderiv(shaderId, GL_COMPILE_STATUS, &compiled);
	if (!compiled)
	{
		GLchar errLog[1024];
		glGetShaderInfoLog(shaderId, 1024, NULL, errLog);
		errStr.assign(string(errLog));
	}

	return compiled;
}

string getShaderSource(const GLuint& shaderId)
{
	if (!shaderId)
	{
		return string();
	}

	GLint sourceLen;
	glGetShaderiv(shaderId, GL_SHADER_SOURCE_LENGTH, &sourceLen);

	GLchar source[sourceLen];
	glGetShaderSource(shaderId, sourceLen, NULL, source);

	return string(source);
}

GLint linkProgram(const GLuint& programId, string& errStr)
{
	GLint linked;

	glLinkProgram(programId);
	glGetProgramiv(programId, GL_LINK_STATUS, &linked);
	if (!linked)
	{
		GLchar errLog[1024];
		glGetProgramInfoLog(programId, 1024, NULL, errLog);
		errStr.assign(string(errLog));
	}

	return linked;
}


AttrInfo* createUniformAttrInfoPtr(GLint type, GLint size, GLuint location)
{
	AttrInfo *rtn = NULL;

	switch (type)
	{
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

		default:
			// TODO: throw error?
			assert(1);
			break;
	}

	return rtn;
}


// void setUniformByLocation(const GLuint& loc, const GLuint& sz, const unsigned int *value)
// {
// 	switch (sz)
// 	{
// 		case 1:
// 			glUniform1uiv(loc, 1, value);
// 			break;
// 		case 2:
// 			glUniform2uiv(loc, 2, value);
// 			break;
// 		case 3:
// 			glUniform3uiv(loc, 3, value);
// 			break;
// 		case 4:
// 			glUniform4uiv(loc, 4, value);
// 			break;
// 	}
// }

// void setUniformByLocation(const GLuint& loc, const GLuint& sz, const int *value)
// {
// 	switch (sz)
// 	{
// 		case 1:
// 			glUniform1iv(loc, 1, value);
// 			break;
// 		case 2:
// 			glUniform2iv(loc, 2, value);
// 			break;
// 		case 3:
// 			glUniform3iv(loc, 3, value);
// 			break;
// 		case 4:
// 			glUniform4iv(loc, 4, value);
// 			break;
// 	}
// }

// void setUniformByLocation(const GLuint& loc, const GLuint& sz, const float *value)
// {
// 	switch (sz)
// 	{
// 		case 1:
// 			glUniform1fv(loc, 1, value);
// 			break;
// 		case 2:
// 			glUniform2fv(loc, 2, value);
// 			break;
// 		case 3:
// 			glUniform3fv(loc, 3, value);
// 			break;
// 		case 4:
// 			glUniform4fv(loc, 4, value);
// 			break;
// 	}
// }


Shader::Shader(
	const string& vertexShaderSrc,
	const string& fragmentShaderSrc)
	: _vertShaderId(0), _fragShaderId(0), _programId(0)
{
	_init(vertexShaderSrc, fragmentShaderSrc);
}

Shader::~Shader()
{
	std::cerr << "IN Shader DESTRUCTOR" << std::endl;
	_cleanupIds();
}

void Shader::_cleanupIds()
{
	if (_vertShaderId)
	{
		glDeleteShader(_vertShaderId);
	}

	if (_fragShaderId)
	{
		glDeleteShader(_fragShaderId);
	}

	if (_programId)
	{
		glDeleteProgram(_programId);
	}
}

void Shader::_init(const string& vertSrc, const string& fragSrc)
{
	string errStr;

	// build and compile the vertex shader
	_vertShaderId = glCreateShader(GL_VERTEX_SHADER);
	if (!compileShader(_vertShaderId, vertSrc, errStr))
	{
		// TODO: throw an error
		_cleanupIds();
		printf("Error compiling vertex shader: %s\n", errStr.c_str());
		assert(1);
	}

	_fragShaderId = glCreateShader(GL_FRAGMENT_SHADER);
	if (!compileShader(_fragShaderId, fragSrc, errStr))
	{
		// TODO: throw an error
		_cleanupIds();
		printf("Error compiling fragment shader: %s\n", errStr.c_str());
		assert(1);
	}

	_programId = glCreateProgram();
	glAttachShader(_programId, _vertShaderId);
	glAttachShader(_programId, _fragShaderId);
	if (!linkProgram(_programId, errStr))
	{
		// TODO: throw an error

		// clean out the shader references
		_cleanupIds();
		printf("Error linking the shader: %s\n", errStr.c_str());
		assert(1);
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
    glGetProgramiv(_programId, GL_ACTIVE_UNIFORMS, &numAttrs);
    for (GLuint i=0; i<numAttrs; ++i)
    {
        glGetActiveUniformName(_programId, i, 512, NULL, attrName);
        std::string attrNameStr(attrName);

        glGetActiveUniformsiv(_programId, 1, &i, GL_UNIFORM_TYPE, &attrType);
        glGetActiveUniformsiv(_programId, 1, &i, GL_UNIFORM_SIZE, &attrSz);
        attrLoc = glGetUniformLocation(_programId, attrName);

        if (boost::algorithm::ends_with(attrNameStr, "[0]")) {
            attrNameStr.erase(attrNameStr.size() - 3, 3);
        }

        _uniformAttrs.insert(
			make_pair(
				attrNameStr,
				unique_ptr<AttrInfo>(createUniformAttrInfoPtr(attrType, attrSz, attrLoc))
			));
    }

    // now setup the vertex attributes
    _vertexAttrs.clear();
    glGetProgramiv(_programId, GL_ACTIVE_ATTRIBUTES, &numAttrs);
    for (int i=0; i<numAttrs; ++i)
    {
        glGetActiveAttrib(_programId, i, 512, NULL, &attrSz, (GLenum *)&attrType, attrName);
        attrLoc = glGetAttribLocation(_programId, attrName);

        // _vertexAttrs.insert(
        //     make_pair(
        //         std::string(attrName),
        //         unique_ptr<AttrInfo>()
        //     ));
    }
}

string Shader::getVertexSource() const
{
	return getShaderSource(_vertShaderId);
}

string Shader::getFragmentSource() const
{
	return getShaderSource(_fragShaderId);
}


template <typename T>
void Shader::setUniformAttribute(const string& attrName, T attrValue) const
{
	AttrMap::const_iterator iter = _uniformAttrs.find(attrName);

	// TODO: check if bound

	if (iter == _uniformAttrs.end())
	{
		// TODO: throw a warning/error?
		cerr << "Uniform attribute: " << attrName << " is not defined in the shader." << endl;
		return;
	}

	AttrInfo *info = iter->second.get();

	GLenum attrType = info->type;
	GLint attrSz = info->size;
	GLuint attrLoc = info->location;
	if (attrSz != 1)
	{
		// TODO: throw a warning/error?
		cerr << "Uniform attribute: " << attrName << " is not the appropriate size. It is size 1 but should be " << attrSz << endl;
		return;
	}

	// TODO: check type mismatch?
	// setUniformByLocation(attrLoc, 1, &attrValue);
	// iter->(*second)();
	info->setAttr(attrValue);
}

template <typename T>
void Shader::setUniformAttribute(const string& attrName, vector<T> attrValue) const
{
	AttrMap::const_iterator iter = _uniformAttrs.find(attrName);

	// TODO: check if bound

	if (iter == _uniformAttrs.end())
	{
		// TODO: throw a warning/error?
		cerr << "Uniform attribute: " << attrName << " is not defined in the shader." << endl;
		return;
	}

	AttrInfo *info = iter->second.get();

	GLuint attrType = info->type;
	GLuint attrSz = info->size;
	GLuint attrLoc = info->location;
	if (attrSz != attrValue.length())
	{
		// TODO: throw a warning/error?
		cerr << "Uniform attribute: " << attrName << " is not the appropriate size. It is size " << attrValue.length() << " but should be " << attrSz << endl;
		return;
	}

	// TODO: check type mismatch?
	info->setAttr(attrValue);
	// setUniformByLocation(attrLoc, attrSz, &attrValue);
}

void Shader::bind() const
{
	// TODO: Throw an error or warning if the program
	// is invalid?

	if (_programId)
	{
		glUseProgram(_programId);
	}
}



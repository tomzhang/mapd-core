#ifndef SHADER_H_
#define SHADER_H_

#include <string>
#include <unordered_map>
#include <vector>
#include <GL/glew.h>

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

    AttrInfo(GLint t, GLint s, GLuint l) : type(t), size(s), location(l) {};
    virtual void setAttr(const void *data) = 0;
};

struct Uniform1uiAttr : AttrInfo {
    Uniform1uiAttr(GLint t, GLint s, GLuint l) : AttrInfo(t, s, l) {};
    void setAttr(const void *data) {
        glUniform1uiv(location, 1, static_cast<const GLuint *>(data));
    }
};

struct Uniform2uiAttr : AttrInfo {
    Uniform2uiAttr(GLint t, GLint s, GLuint l) : AttrInfo(t, s, l) {};
    void setAttr(const void *data) {
        glUniform2uiv(location, 1, static_cast<const GLuint *>(data));
    }
};

struct Uniform3uiAttr : AttrInfo {
    Uniform3uiAttr(GLint t, GLint s, GLuint l) : AttrInfo(t, s, l) {};
    void setAttr(const void *data) {
        glUniform3uiv(location, 1, static_cast<const GLuint *>(data));
    }
};

struct Uniform4uiAttr : AttrInfo {
    Uniform4uiAttr(GLint t, GLint s, GLuint l) : AttrInfo(t, s, l) {};
    void setAttr(const void *data) {
        glUniform4uiv(location, 1, static_cast<const GLuint *>(data));
    }
};

struct Uniform1iAttr : AttrInfo {
    Uniform1iAttr(GLint t, GLint s, GLuint l) : AttrInfo(t, s, l) {};
    void setAttr(const void *data) {
        glUniform1iv(location, 1, static_cast<const GLint *>(data));
    }
};

struct Uniform2iAttr : AttrInfo {
    Uniform2iAttr(GLint t, GLint s, GLuint l) : AttrInfo(t, s, l) {};
    void setAttr(const void *data) {
        glUniform2iv(location, 1, static_cast<const GLint *>(data));
    }
};

struct Uniform3iAttr : AttrInfo {
    Uniform3iAttr(GLint t, GLint s, GLuint l) : AttrInfo(t, s, l) {};
    void setAttr(const void *data) {
        glUniform3iv(location, 1, static_cast<const GLint *>(data));
    }
};

struct Uniform4iAttr : AttrInfo {
    Uniform4iAttr(GLint t, GLint s, GLuint l) : AttrInfo(t, s, l) {};
    void setAttr(const void *data) {
        glUniform4iv(location, 1, static_cast<const GLint *>(data));
    }
};

struct Uniform1fAttr : AttrInfo {
    Uniform1fAttr(GLint t, GLint s, GLuint l) : AttrInfo(t, s, l) {};
    void setAttr(const void *data) {
        glUniform1fv(location, 1, static_cast<const GLfloat *>(data));
    }
};

struct Uniform2fAttr : AttrInfo {
    Uniform2fAttr(GLint t, GLint s, GLuint l) : AttrInfo(t, s, l) {};
    void setAttr(const void *data) {
        glUniform2fv(location, 1, static_cast<const GLfloat *>(data));
    }
};

struct Uniform3fAttr : AttrInfo {
    Uniform3fAttr(GLint t, GLint s, GLuint l) : AttrInfo(t, s, l) {};
    void setAttr(const void *data) {
        glUniform3fv(location, 1, static_cast<const GLfloat *>(data));
    }
};

struct Uniform4fAttr : AttrInfo {
    Uniform4fAttr(GLint t, GLint s, GLuint l) : AttrInfo(t, s, l) {};
    void setAttr(const void *data) {
        glUniform4fv(location, 1, static_cast<const GLfloat *>(data));
    }
};

typedef unordered_map<string, unique_ptr<AttrInfo>> AttrMap;
enum {ATTR_TYPE = 0, ATTR_SIZE, ATTR_LOC};


class Shader
{
    private:
        GLuint _vertShaderId;
        GLuint _fragShaderId;
        GLuint _programId;
        AttrMap _uniformAttrs;
        AttrMap _vertexAttrs;

        void _init(const string& vertSrc, const string& fragSrc);
        void _cleanupIds();

    public:
        Shader(const string& vertexShaderSrc, const string& fragmentShaderSrc);
        ~Shader();

        string getVertexSource() const;
        string getFragmentSource() const;

        template <typename T>
        void setUniformAttribute(const string& attrName, T attrValue) const;

        template <typename T>
        void setUniformAttribute(const string& attrName, vector<T> attrValue) const;

        void bind() const;
};

} // namespace MapD_Renderer

#endif // SHADER_H_

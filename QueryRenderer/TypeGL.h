#ifndef TYPE_GL_H_
#define TYPE_GL_H_

#include <string>
#include <memory>
#include <GL/glew.h>
#include <functional>
#include <cstdint>

#define BUFFER_OFFSET(i) ((char *)NULL + (i))

namespace MapD_Renderer {

struct BaseTypeGL {
    BaseTypeGL() {}
    BaseTypeGL(const BaseTypeGL& baseTypeGL) {}
    virtual ~BaseTypeGL() {}

    virtual std::shared_ptr<BaseTypeGL> clone() = 0;

    virtual int numComponents() = 0;
    virtual int numBytes() = 0;
    virtual int baseGLType() = 0;
    virtual std::string glslType() = 0;

    virtual void bind(GLuint shaderAttrLoc, int stride, int offset) = 0;

    virtual GLboolean normalize() {
        return GL_FALSE;
    }
};

typedef std::unique_ptr<BaseTypeGL> TypeGLUqPtr;
typedef std::shared_ptr<BaseTypeGL> TypeGLShPtr;

template <typename T, int componentCnt=1>
struct TypeGL : BaseTypeGL {

    TypeGL(bool useAsFloat = false, bool normalize = false) {
        setUseAsFloat(useAsFloat, normalize);
    }

    TypeGL(const TypeGL<T, componentCnt>& typeGL) {
        setUseAsFloat(typeGL._useAsFloat, typeGL._normalize);
    }

    ~TypeGL() {}

    TypeGLShPtr clone() {
        return TypeGLShPtr(new TypeGL<T, componentCnt>(*this));
    }

    int numComponents() {
        return componentCnt;
    }

    int numBytes() {
        return sizeof(T)*numComponents();
    }

    int baseGLType();
    std::string glslType();

    void setUseAsFloat(bool useAsFloat, bool normalize = false) {
        _useAsFloat = useAsFloat;
        _normalize = normalize;
        // if (useAsFloat) {
        //     _bindFunc = std::bind(glVertexAttribPointer, std::placeholders::_1, numComponents(), baseGLType(), (normalize ? GL_TRUE : GL_FALSE), std::placeholders::_2, std::placeholders::_3);

        //     _useAsFloat = useAsFloat;
        //     _normalize = normalize;
        // } else {
        //     _bindFunc = std::bind(_getAltVertexAttribPointerFunc(), std::placeholders::_1, numComponents(), baseGLType(), std::placeholders::_2, std::placeholders::_3);
        //     _normalize = false;
        // }
    }

    void bind(GLuint shaderAttrLoc, int stride, int offset) {
        if (_useAsFloat) {
            glVertexAttribPointer(shaderAttrLoc, numComponents(), baseGLType(), (_normalize ? GL_TRUE : GL_FALSE), stride, (GLvoid *) offset);
        } else {
            _getAltVertexAttribPointerFunc()(shaderAttrLoc, numComponents(), baseGLType(), stride, (GLvoid*) offset);
        }
    }

    // void bind(GLuint shaderAttrLoc, int stride, int offset) {
    //     _bindFunc(shaderAttrLoc, stride, BUFFER_OFFSET(offset));
    // }

private:
    typedef std::function<void(GLuint, GLint, GLenum, GLsizei, const GLvoid *)> VertexAttribPtrFunc;

    bool _useAsFloat;  // when binding the attr type, this determines whether the attr
                       // will be bound as a float (via glVertexAttribPointer) or
                       // as another type (either glVertexAttribIPointer or glVertexAttribLPointer
                       // depending on the type).

    bool _normalize;    // if binding as a float, this determines whether the value when converted
                       // to a float should be normalized. If true, if the values is unsigned
                       // normalization results in a value from 0 to 1. If signed, then it
                       // results in a value from -1 to 1.

    // std::function<void(GLuint, GLsizei, const GLvoid *)> _bindFunc;

    VertexAttribPtrFunc _getAltVertexAttribPointerFunc() {
        return glVertexAttribIPointer;
    }
};



} // namespace MapD_renderer


#endif // TYPE_GL_H_

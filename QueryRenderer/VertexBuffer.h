#ifndef VERTEX_BUFFER_H_
#define VERTEX_BUFFER_H_

// #include "Buffer.h"
#include "BufferLayout.h"
#include "Shader.h"
#include <vector>
#include <memory>
#include <iostream>
#include <GL/glew.h>
#include <unordered_map>

namespace MapD_Renderer {

class VertexBuffer {
    public:
        typedef std::shared_ptr<BaseBufferLayout> BufferLayoutShPtr;

        VertexBuffer(const BufferLayoutShPtr& layoutPtr) : _size(0), _bufferId(0), _layoutPtr(layoutPtr) {}

        template <typename T>
        VertexBuffer(const std::vector<T>& data, const BufferLayoutShPtr& layoutPtr) : _size(0), _bufferId(0), _layoutPtr(layoutPtr) {
            // TODO: validate that the data and the layout align

            // _size will be set in the bufferData() call
            bufferData((void *)&data[0], data.size(), sizeof(T));
        }

        ~VertexBuffer() {
            // std::cout << "IN VertexBuffer DESTRUCTOR" << std::endl;
            if (_bufferId) {
                glDeleteBuffers(1, &_bufferId);
            }
        }

        BufferLayoutShPtr getBufferLayout() const {
            return BufferLayoutShPtr(_layoutPtr);
        }

        bool hasAttribute(const std::string& attrName) {
            return _layoutPtr->hasAttribute(attrName);
        }

        TypeGLShPtr getAttributeTypeGL(const std::string& attrName) {
            return _layoutPtr->getAttributeTypeGL(attrName);
        }

        // void setBufferLayout(const std::shared_ptr<BaseBufferLayout>& layoutPtr) {
        //     _layoutPtr = layoutPtr;
        // }

        void bufferData(void *data, int numItems, int numBytesPerItem, GLenum target=GL_ARRAY_BUFFER, GLenum usage=GL_STATIC_DRAW) {
            _initBuffer();
            // glGenBuffers(1, &_bufferId);

            // don't mess with the current state
            // TODO: Apply some kind of push-pop state system
            GLint currArrayBuf;
            glGetIntegerv(_getBufferBinding(target), &currArrayBuf);

            glBindBuffer(target, _bufferId);

            // TODO: what about the last usage parameter? Should the buffer's constructor
            // handle the different usage types?
            glBufferData(target, numItems*numBytesPerItem, data, usage);

            glBindBuffer(target, currArrayBuf);

            _size = numItems;
        }

        int size() const {
            return _size;
        }

        void bindToRenderer(Shader *activeShader, const std::string& attr = "", const std::string& shaderAttr = "") {
            assert(_bufferId && _layoutPtr);
            glBindBuffer(GL_ARRAY_BUFFER, _bufferId);
            _layoutPtr->bindToRenderer(activeShader, size(), attr, shaderAttr);
        }

    private:

        static GLenum _getBufferBinding(GLenum target) {
            typedef std::unordered_map<GLenum, GLenum> BufferBindingMap;
            static const BufferBindingMap bufferBindings = {
                {GL_ARRAY_BUFFER, GL_ARRAY_BUFFER_BINDING},
                {GL_PIXEL_UNPACK_BUFFER, GL_PIXEL_UNPACK_BUFFER_BINDING}
            };

            BufferBindingMap::const_iterator itr;

            if ((itr = bufferBindings.find(target)) == bufferBindings.end()) {
                // TODO: throw exception
                assert(false);
            }

            return itr->first;
        }


        int _size;
        GLuint _bufferId;

        std::shared_ptr<BaseBufferLayout> _layoutPtr;

        void _initBuffer() {
            if (!_bufferId) {
                glGenBuffers(1, &_bufferId);
            }
        }
};

typedef std::unique_ptr<VertexBuffer> VertexBufferUqPtr;
typedef std::shared_ptr<VertexBuffer> VertexBufferShPtr;

} // namespace MapD_Renderer

#endif  // VERTEX_BUFFER_H_

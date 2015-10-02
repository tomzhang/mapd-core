#ifndef VERTEX_BUFFER_H_
#define VERTEX_BUFFER_H_

#include "Buffer.h"
#include "BufferLayout.h"
#include <vector>
#include <memory>
#include <iostream>
#include <GL/glew.h>

namespace MapD_Renderer {

class VertexBuffer : Buffer {
    public:
        typedef std::shared_ptr<BaseBufferLayout> BufferLayoutPtr;

        VertexBuffer() : _size(0) {};

        VertexBuffer(const BufferLayoutPtr& layoutPtr) : _size(0), _layoutPtr(layoutPtr) {}

        template <typename T>
        VertexBuffer(const std::vector<T>& data, const BufferLayoutPtr& layoutPtr) : _size(0), _layoutPtr(layoutPtr) {
            // TODO: validate that the data and the layout align

            // _size will be set in the bufferData() call
            bufferData((void *)&data[0], data.size(), sizeof(T));
        }

        template <typename T>
        VertexBuffer(std::vector<T>);

        VertexBuffer(GLuint bufferId);

        BufferLayoutPtr getBufferLayout() const {
            return BufferLayoutPtr(_layoutPtr);
        }

        void setBufferLayout(const std::shared_ptr<BaseBufferLayout>& layoutPtr) {
            _layoutPtr = layoutPtr;
        }

        void bufferData(void *data, int numItems, int numBytesPerItem) {
            _initBuffer();

            // don't mess with the current state
            // TODO: Apply some kind of push-pop state system
            GLint currArrayBuf;
            glGetIntegerv(GL_ARRAY_BUFFER_BINDING, &currArrayBuf);

            glBindBuffer(GL_ARRAY_BUFFER, _bufferId);

            // TODO: what about the last usage parameter? Should the buffer's constructor
            // handle the different usage types?
            glBufferData(GL_ARRAY_BUFFER, numItems*numBytesPerItem, data, GL_STATIC_DRAW);

            glBindBuffer(GL_ARRAY_BUFFER, currArrayBuf);

            _size = numItems;
        }

        int size() const {
            return _size;
        }

    private:
        int _size;
        std::shared_ptr<BaseBufferLayout> _layoutPtr;
        GLuint _bufferId;
};

} // namespace MapD_Renderer

#endif  // VERTEX_BUFFER_H_

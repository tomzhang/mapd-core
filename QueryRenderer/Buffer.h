#ifndef BUFFER_H_
#define BUFFER_H_

#include <vector>
#include <GL/glew.h>

namespace MapD_Renderer {

class Buffer {
    public:
        Buffer() : _bufferId(0) {}

        // template <typename T>
        // Buffer(std::vector<T>);

        // TODO: Should we make bufferId a shared resource?
        Buffer(GLuint bufferId) : _bufferId(bufferId) {}

        virtual ~Buffer() {
            if (_bufferId) {
                glDeleteBuffers(1, &_bufferId);
            }
        }

        virtual void bufferData(void *data, int numItems, int numBytesPerItem) = 0;
        virtual int size() const = 0;

    private:
        GLuint _bufferId;

    protected:
        void _initBuffer() {
            if (!_bufferId) {
                glGenBuffers(1, &_bufferId);
            }
        }
};

} // namespace MapD_Renderer

#endif  // BUFFER_H_

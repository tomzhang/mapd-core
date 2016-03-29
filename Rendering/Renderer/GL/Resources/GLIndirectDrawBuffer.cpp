#include "GLIndirectDrawBuffer.h"

namespace Rendering {
namespace GL {
namespace Resources {

GLIndirectDrawVertexBuffer::GLIndirectDrawVertexBuffer(const RendererWkPtr& rendererPtr,
                                                       BufferAccessType accessType,
                                                       BufferAccessFreq accessFreq)
    : GLBaseIndirectDrawBuffer(rendererPtr,
                               GLResourceType::INDIRECT_DRAW_VERTEX_BUFFER,
                               GLBufferType::INDIRECT_DRAW_VERTEX_BUFFER,
                               accessType,
                               accessFreq) {
}

GLIndirectDrawVertexBuffer::GLIndirectDrawVertexBuffer(const RendererWkPtr& rendererPtr,
                                                       size_t numBytes,
                                                       BufferAccessType accessType,
                                                       BufferAccessFreq accessFreq)
    : GLBaseIndirectDrawBuffer(rendererPtr,
                               GLResourceType::INDIRECT_DRAW_VERTEX_BUFFER,
                               GLBufferType::INDIRECT_DRAW_VERTEX_BUFFER,
                               numBytes,
                               accessType,
                               accessFreq) {
}

GLIndirectDrawVertexBuffer::GLIndirectDrawVertexBuffer(const RendererWkPtr& rendererPtr,
                                                       const std::vector<IndirectDrawVertexData>& items,
                                                       BufferAccessType accessType,
                                                       BufferAccessFreq accessFreq)
    : GLBaseIndirectDrawBuffer(rendererPtr,
                               GLResourceType::INDIRECT_DRAW_VERTEX_BUFFER,
                               GLBufferType::INDIRECT_DRAW_VERTEX_BUFFER,
                               items,
                               accessType,
                               accessFreq) {
}

GLIndirectDrawIndexBuffer::GLIndirectDrawIndexBuffer(const RendererWkPtr& rendererPtr,
                                                     BufferAccessType accessType,
                                                     BufferAccessFreq accessFreq)
    : GLBaseIndirectDrawBuffer(rendererPtr,
                               GLResourceType::INDIRECT_DRAW_INDEX_BUFFER,
                               GLBufferType::INDIRECT_DRAW_INDEX_BUFFER,
                               accessType,
                               accessFreq) {
}

GLIndirectDrawIndexBuffer::GLIndirectDrawIndexBuffer(const RendererWkPtr& rendererPtr,
                                                     size_t numBytes,
                                                     BufferAccessType accessType,
                                                     BufferAccessFreq accessFreq)
    : GLBaseIndirectDrawBuffer(rendererPtr,
                               GLResourceType::INDIRECT_DRAW_INDEX_BUFFER,
                               GLBufferType::INDIRECT_DRAW_INDEX_BUFFER,
                               numBytes,
                               accessType,
                               accessFreq) {
}

GLIndirectDrawIndexBuffer::GLIndirectDrawIndexBuffer(const RendererWkPtr& rendererPtr,
                                                     const std::vector<IndirectDrawIndexData>& items,
                                                     BufferAccessType accessType,
                                                     BufferAccessFreq accessFreq)
    : GLBaseIndirectDrawBuffer(rendererPtr,
                               GLResourceType::INDIRECT_DRAW_INDEX_BUFFER,
                               GLBufferType::INDIRECT_DRAW_INDEX_BUFFER,
                               items,
                               accessType,
                               accessFreq) {
}

}  // namespace Resources
}  // namespace GL
}  // namespace Rendering

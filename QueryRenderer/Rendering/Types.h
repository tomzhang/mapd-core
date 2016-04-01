#ifndef QUERYRENDERER_RENDERING_TYPES_H_
#define QUERYRENDERER_RENDERING_TYPES_H_

#include <memory>

namespace QueryRenderer {

class QueryFramebuffer;
typedef std::unique_ptr<QueryFramebuffer> QueryFramebufferUqPtr;

class QueryIdMapPixelBuffer;
typedef std::shared_ptr<QueryIdMapPixelBuffer> QueryIdMapPixelBufferShPtr;

class QueryRenderCompositor;

}  // namespace QueryRenderer

#endif  // QUERYRENDERER_RENDERING_TYPES_H_

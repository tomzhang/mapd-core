#ifndef QUERYRENDERER_RENDERING_TYPES_H_
#define QUERYRENDERER_RENDERING_TYPES_H_

#include <memory>

namespace QueryRenderer {

class QueryFramebuffer;
typedef std::unique_ptr<QueryFramebuffer> QueryFramebufferUqPtr;

class QueryIdMapPixelBuffer;
typedef std::shared_ptr<QueryIdMapPixelBuffer> QueryIdMapPixelBufferShPtr;

class QueryIdMapPboPool;
typedef std::unique_ptr<QueryIdMapPboPool> QueryIdMapPboPoolUqPtr;

class QueryRenderCompositor;
typedef std::shared_ptr<QueryRenderCompositor> QueryRenderCompositorShPtr;

}  // namespace QueryRenderer

#endif  // QUERYRENDERER_RENDERING_TYPES_H_

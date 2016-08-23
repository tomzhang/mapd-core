#ifndef QUERYRENDERER_RENDERING_TYPES_H_
#define QUERYRENDERER_RENDERING_TYPES_H_

#include <memory>

namespace QueryRenderer {

class QueryFramebuffer;
typedef std::unique_ptr<QueryFramebuffer> QueryFramebufferUqPtr;
typedef std::shared_ptr<QueryFramebuffer> QueryFramebufferShPtr;

class QueryIdMapPixelBuffer;
typedef std::shared_ptr<QueryIdMapPixelBuffer> QueryIdMapPixelBufferShPtr;

class QueryIdMapPboPool;
typedef std::unique_ptr<QueryIdMapPboPool> QueryIdMapPboPoolUqPtr;

class QueryAccumTxPool;
typedef std::unique_ptr<QueryAccumTxPool> QueryAccumTxPoolUqPtr;

class QueryRenderCompositor;
typedef std::shared_ptr<QueryRenderCompositor> QueryRenderCompositorShPtr;

class QueryRenderSMAAPass;
typedef std::shared_ptr<QueryRenderSMAAPass> QueryRenderSMAAPassShPtr;

}  // namespace QueryRenderer

#endif  // QUERYRENDERER_RENDERING_TYPES_H_

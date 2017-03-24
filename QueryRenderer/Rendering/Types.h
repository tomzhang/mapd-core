#ifndef QUERYRENDERER_RENDERING_TYPES_H_
#define QUERYRENDERER_RENDERING_TYPES_H_

#include <memory>

namespace QueryRenderer {

class QueryFramebuffer;
typedef std::unique_ptr<QueryFramebuffer> QueryFramebufferUqPtr;
typedef std::shared_ptr<QueryFramebuffer> QueryFramebufferShPtr;

template <typename T>
class QueryIdMapPixelBuffer;

typedef std::shared_ptr<QueryIdMapPixelBuffer<unsigned int>> QueryIdMapPixelBufferUIntShPtr;
typedef std::weak_ptr<QueryIdMapPixelBuffer<unsigned int>> QueryIdMapPixelBufferUIntWkPtr;
typedef std::shared_ptr<QueryIdMapPixelBuffer<int>> QueryIdMapPixelBufferIntShPtr;
typedef std::weak_ptr<QueryIdMapPixelBuffer<int>> QueryIdMapPixelBufferIntWkPtr;

template <typename T>
class QueryIdMapPboPool;

typedef std::unique_ptr<QueryIdMapPboPool<unsigned int>> QueryIdMapPboPoolUIntUqPtr;
typedef std::unique_ptr<QueryIdMapPboPool<int>> QueryIdMapPboPoolIntUqPtr;

class QueryAccumTxPool;
typedef std::unique_ptr<QueryAccumTxPool> QueryAccumTxPoolUqPtr;

class QueryRenderCompositor;
typedef std::shared_ptr<QueryRenderCompositor> QueryRenderCompositorShPtr;

class QueryRenderSMAAPass;
typedef std::shared_ptr<QueryRenderSMAAPass> QueryRenderSMAAPassShPtr;

class DistributedRenderBufferCompositor;
typedef std::shared_ptr<DistributedRenderBufferCompositor> DistributedRenderBufferCompositorShPtr;

}  // namespace QueryRenderer

#endif  // QUERYRENDERER_RENDERING_TYPES_H_

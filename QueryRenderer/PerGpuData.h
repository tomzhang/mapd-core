#ifndef QUERYRENDERER_PERGPUDATA_H_
#define QUERYRENDERER_PERGPUDATA_H_

#include "Types.h"
#include "Rendering/Types.h"
#include "Interop/Types.h"
#include "Interop/QueryBuffer.h"

#include <Rendering/Types.h>
#include <Rendering/Renderer/GL/Types.h>

#include <boost/multi_index_container.hpp>
#include <boost/multi_index/random_access_index.hpp>
#include <boost/multi_index/ordered_index.hpp>
#include <boost/multi_index/tag.hpp>

namespace QueryRenderer {

struct RootPerGpuData {
  GpuId gpuId;
  QueryResultVertexBufferShPtr queryResultBufferPtr;
  Rendering::WindowShPtr windowPtr;
  Rendering::RendererShPtr rendererPtr;

  // TODO(croot): make a pool of framebuffers?
  // This would be necessary if we ever support asynchronous
  // queries, even partially (i.e. the cuda part of the query
  // may be serialized, but the render part could run asynchronously.)
  QueryFramebufferUqPtr msFramebufferPtr;
  QueryFramebufferShPtr blitFramebufferPtr;
  QueryRenderCompositorShPtr compositorPtr;
  QueryIdMapPboPoolUqPtr pboPoolPtr;

  RootPerGpuData(GpuId gpuId);
  ~RootPerGpuData();

  void makeActiveOnCurrentThread() const;
  void makeInactive() const;

  ::Rendering::Renderer* getRenderer();
  ::Rendering::GL::GLRenderer* getGLRenderer();

  void resize(size_t width, size_t height, bool resizeCompositor = true);

  QueryFramebufferUqPtr& getRenderFramebuffer() { return msFramebufferPtr; }
  QueryFramebufferShPtr& getBlitFramebuffer() { return blitFramebufferPtr; }
  QueryRenderCompositorShPtr& getCompositor() { return compositorPtr; }
  GpuId getCompositorGpuId();

  QueryIdMapPboPoolUqPtr& getIdMapPboPool() { return pboPoolPtr; }

  QueryIdMapPixelBufferShPtr getInactiveIdMapPbo(size_t width, size_t height);

  void setIdMapPboInactive(QueryIdMapPixelBufferShPtr& pbo);
};

struct inorder {};
struct RootPerGpuDataId {
  typedef GpuId result_type;
  result_type operator()(const RootPerGpuDataShPtr& perGpuData) const { return perGpuData->gpuId; }
};

typedef ::boost::multi_index_container<RootPerGpuDataShPtr,
                                       ::boost::multi_index::indexed_by<
                                           // hashed on gpuId
                                           ::boost::multi_index::ordered_unique<RootPerGpuDataId>,
                                           ::boost::multi_index::random_access<::boost::multi_index::tag<inorder>>>>
    RootPerGpuDataMap;

typedef RootPerGpuDataMap::index<inorder>::type RootPerGpuDataMap_in_order;

struct BasePerGpuData {
  RootPerGpuDataWkPtr rootPerGpuData;

  BasePerGpuData() : rootPerGpuData() {}
  BasePerGpuData(const RootPerGpuDataShPtr& rootPerGpuData) : rootPerGpuData(rootPerGpuData) {}
  BasePerGpuData(const BasePerGpuData& gpuData) : rootPerGpuData(gpuData.rootPerGpuData) {}
  BasePerGpuData(BasePerGpuData&& data) noexcept : rootPerGpuData(std::move(data.rootPerGpuData)) {}
  ~BasePerGpuData() {
    // need to make active to properly destroy gpu resources
    // TODO(croot): uncomment this if we have GL resources at
    // this level (i.e. a framebuffer or a compositor per gpu)
    // TODO(croot): reset to previously active renderer?
    // makeActiveOnCurrentThread();
  }

  RootPerGpuDataShPtr getRootPerGpuData() const { return rootPerGpuData.lock(); }

  void makeActiveOnCurrentThread() const;
  void makeInactive() const;

  ::Rendering::Renderer* getRenderer();
  ::Rendering::GL::GLRenderer* getGLRenderer();

  virtual void resize(size_t width, size_t height);

  QueryFramebufferUqPtr& getRenderFramebuffer();
  QueryFramebufferShPtr& getBlitFramebuffer();
  std::shared_ptr<QueryRenderCompositor>& getCompositor();
  GpuId getCompositorGpuId();

  QueryIdMapPixelBufferShPtr getInactiveIdMapPbo(size_t width, size_t height);
  void setIdMapPboInactive(QueryIdMapPixelBufferShPtr& pbo);
};

}  // namespace QueryRenderer

#endif  // QUERYRENDERER_PERGPUDATA_H_

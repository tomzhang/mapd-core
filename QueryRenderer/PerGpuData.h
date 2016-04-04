#ifndef QUERYRENDERER_PERGPUDATA_H_
#define QUERYRENDERER_PERGPUDATA_H_

#include "Types.h"
#include "Rendering/Types.h"
#include "Interop/Types.h"

#include <Rendering/Types.h>
#include <Rendering/Renderer/GL/Types.h>

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
  QueryFramebufferUqPtr framebufferPtr;
  QueryRenderCompositorShPtr compositorPtr;
  QueryIdMapPboPoolUqPtr pboPoolPtr;

  RootPerGpuData(GpuId gpuId);
  ~RootPerGpuData();

  void makeActiveOnCurrentThread() const;
  void makeInactive() const;

  ::Rendering::Renderer* getRenderer();
  ::Rendering::GL::GLRenderer* getGLRenderer();

  void resize(size_t width, size_t height);

  QueryFramebufferUqPtr& getFramebuffer() { return framebufferPtr; }
  QueryRenderCompositorShPtr& getCompositor() { return compositorPtr; }
  QueryIdMapPboPoolUqPtr& getIdMapPboPool() { return pboPoolPtr; }

  QueryIdMapPixelBufferShPtr getInactiveIdMapPbo(size_t width, size_t height);

  void setIdMapPboInactive(QueryIdMapPixelBufferShPtr& pbo);
};

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

  QueryFramebufferUqPtr& getFramebuffer();
  std::shared_ptr<QueryRenderCompositor>& getCompositor();
  GpuId getCompositorGpuId();

  QueryIdMapPixelBufferShPtr getInactiveIdMapPbo(size_t width, size_t height);
  void setIdMapPboInactive(QueryIdMapPixelBufferShPtr& pbo);
};

}  // namespace QueryRenderer

#endif  // QUERYRENDERER_PERGPUDATA_H_

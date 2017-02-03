#ifndef QUERYRENDERER_QUERYRENDERER_H_
#define QUERYRENDERER_QUERYRENDERER_H_

#include "Types.h"
#include "Rendering/Types.h"
#include "Interop/Types.h"
#include "Marks/Types.h"
#include "Scales/Types.h"
#include "Data/Types.h"
#include "RootCache.h"

#include "Rendering/QueryRenderCompositor.h"
#include "Rendering/QueryIdMapPixelBuffer.h"
#include "Utils/RapidJSONUtils.h"

#include <Rendering/Renderer/GL/Resources/Types.h>
#include <Rendering/Objects/Array2d.h>

#include <string>
#include <map>
#include <unordered_map>
#include <set>
#include <vector>

class Executor;

namespace QueryRenderer {

class QueryRenderer {
 public:
  explicit QueryRenderer(int userId,
                         int widgetId,
                         const std::shared_ptr<RootCache>& qrmGpuCache,
                         bool doHitTest = false,
                         bool doDepthTest = false) noexcept;

  explicit QueryRenderer(int userId,
                         int widgetId,
                         const std::shared_ptr<RootCache>& qrmGpuCache,
                         const std::shared_ptr<rapidjson::Document>& jsonDocumentPtr,
                         bool doHitTest = false,
                         bool doDepthTest = false);

  explicit QueryRenderer(int userId,
                         int widgetId,
                         const std::shared_ptr<RootCache>& qrmGpuCache,
                         const std::string& configJSON,
                         bool doHitTest = false,
                         bool doDepthTest = false);

  ~QueryRenderer();

  size_t getWidth();
  size_t getHeight();
  void setWidthHeight(size_t width, size_t height);

  void setJSONConfig(const std::string& configJSON, bool forceUpdate = false);
  void setJSONDocument(const std::shared_ptr<rapidjson::Document>& jsonDocumentPtr, bool forceUpdate = false);

  void updateResultsPostQuery(Executor* executor);
  void setQueryExecutionParams(Executor* executor,
                               QueryExecCB execFunc,
                               std::shared_ptr<RenderQueryExecuteTimer>& renderTimer);
  void unsetQueryExecutionParams();

  void render(bool inactivateRendererOnThread = true);
  PngData renderToPng(int compressionLevel = -1);

  TableIdRowIdPair getIdAt(size_t x, size_t y, size_t pixelRadius = 0);
  std::string getVegaTableNameWithTableId(const TableId tableId) const;
  QueryDataTableBaseType getVegaTableTypeWithTableId(const TableId tableId) const;

  QueryRendererContext* getContext() { return _ctx.get(); }

  static void renderPasses(
      const std::shared_ptr<RootPerGpuDataMap>& qrmPerGpuData,
      QueryRendererContext* ctx,
      const std::set<GpuId>& usedGpus,
      bool clearFboEveryPass,
      std::function<
          void(::Rendering::GL::GLRenderer*, QueryFramebufferUqPtr&, size_t, size_t, bool, bool, int, ScaleShPtr&, int)>
          perPassGpuCB,
      std::function<void(const std::set<GpuId>&, size_t, size_t, bool, bool, int, ScaleShPtr&)> passCompleteCB);

  static QueryFramebufferUqPtr& renderGpu(GpuId gpuId,
                                          const std::shared_ptr<RootPerGpuDataMap>& qrmPerGpuData,
                                          QueryRendererContext* ctx,
                                          int r = -1,
                                          int g = -1,
                                          int b = -1,
                                          int a = -1);

 private:
  std::shared_ptr<QueryRendererContext> _ctx;

  GpuId _pboGpu;
  QueryIdMapPixelBufferUIntShPtr _pbo1A;  // need a pbo for each of the id buffers
  QueryIdMapPixelBufferUIntWkPtr _pbo1Awk;
  QueryIdMapPixelBufferUIntShPtr _pbo1B;    // The row id is a 64-bit int, so its packed into 2 32-bit textures
  QueryIdMapPixelBufferUIntWkPtr _pbo1Bwk;  // The row id is a 64-bit int, so its packed into 2 32-bit textures
  QueryIdMapPixelBufferIntShPtr _pbo2;      // 2 id buffers - one for the row id, the other for table id
  QueryIdMapPixelBufferIntWkPtr _pbo2wk;    // 2 id buffers - one for the row id, the other for table id

  typedef ::Rendering::Objects::Array2d<unsigned int> Array2dui;
  typedef ::Rendering::Objects::Array2d<int> Array2di;
  bool _idPixelsDirty;
  std::shared_ptr<Array2dui> _id1APixels;
  std::shared_ptr<Array2dui> _id1BPixels;
  std::shared_ptr<Array2di> _id2Pixels;

  void _clear(bool preserveDimensions = false);
  void _clearGpuResources();
  void _clearAll(bool preserveDimensions = false);

  // std::unordered_set<GpuId> _initUnusedGpus();
  // void _initGpuResources(RootPerGpuDataMap* qrmPerGpuData,
  //                        const std::vector<GpuId>& gpuIds,
  //                        std::unordered_set<GpuId>& unusedGpus);
  // void _updateGpuData(const GpuId& gpuId,
  //                     RootPerGpuDataMap* qrmPerGpuData,
  //                     std::unordered_set<GpuId>& unusedGpus,
  //                     size_t width,
  //                     size_t height);

  void _initFromJSON(const std::shared_ptr<rapidjson::Document>& jsonDocumentPtr, bool forceUpdate = false);
  void _initFromJSON(const std::string& configJSON, bool forceUpdate = false);
  void _resizeFramebuffers(int width, int height);

  void _update();
  void _createPbo(const std::set<GpuId>& usedGpus, int width = -1, int height = -1, bool makeContextInactive = false);
  void _releasePbo(bool makeContextInactive = false);
  void _updatePbo();

  void _render(const std::set<GpuId>& usedGpus, bool inactivateRendererOnThread = true);
  QueryFramebufferShPtr& _runAntialiasingPass(const RootPerGpuDataShPtr& gpuData, QueryFramebuffer* rndrFbo);

  friend class QueryRendererContext;
  friend class QueryRenderCompositor;
};

};  // QueryRenderer namespace

#endif  // QUERYRENDERER_QUERYRENDERER_H_

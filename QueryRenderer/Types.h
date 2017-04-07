#ifndef QUERYRENDERER_TYPES_H_
#define QUERYRENDERER_TYPES_H_

#include <Catalog/TableDescriptor.h>
#include <QueryEngine/ResultRows.h>
#include <QueryEngine/TargetMetaInfo.h>
#include "rapidjson/document.h"
#include "rapidjson/pointer.h"
#include <memory>
#include <vector>
#include <Rendering/Objects/Array2d.h>

class Executor;

namespace QueryRenderer {

struct RenderQueryExecuteTimer {
  int64_t queue_time_ms;
  int64_t execution_time_ms;
  int64_t render_time_ms;
};

enum class RefType { DATA = 0, SCALE };
class JSONRefObject {
 public:
  virtual ~JSONRefObject() {}

  RefType getRefType() const { return _refType; }
  std::string getName() const { return _name; }
  const std::string& getNameRef() const { return _name; }

 protected:
  // JSONRefObject(const RefType refType, const std::string& name) : _name(name), _refType(refType) {}

  JSONRefObject(const RefType refType, const std::string& name, const rapidjson::Pointer& jsonPath)
      : _name(name), _jsonPath(jsonPath), _refType(refType) {}

  std::string _name;
  rapidjson::Pointer _jsonPath;

 private:
  RefType _refType;
};
typedef std::shared_ptr<JSONRefObject> RefObjWkPtr;
typedef std::shared_ptr<JSONRefObject> RefObjShPtr;
enum class RefEventType { UPDATE = 0, REMOVE, REPLACE, ALL };
typedef std::function<void(RefEventType, const RefObjShPtr&)> RefEventCallback;
typedef uint32_t RefCallbackId;

typedef size_t GpuId;
typedef decltype(TableDescriptor::tableId) TableId;

struct PngData;

struct UserWidgetIdPair {
  const std::string& userId;
  const int widgetId;

  UserWidgetIdPair(const std::string& userId, int widgetId) : userId(userId), widgetId(widgetId) {}
};

struct RootPerGpuData;
typedef std::weak_ptr<RootPerGpuData> RootPerGpuDataWkPtr;
typedef std::shared_ptr<RootPerGpuData> RootPerGpuDataShPtr;

class QueryRenderManager;

struct QueryDataLayout;
typedef std::shared_ptr<QueryDataLayout> QueryDataLayoutShPtr;

class QueryRenderer;
typedef std::unique_ptr<QueryRenderer> QueryRendererUqPtr;

class QueryRendererContext;
typedef std::shared_ptr<QueryRendererContext> QueryRendererContextShPtr;

struct RootCache;
typedef std::shared_ptr<RootCache> RootCacheShPtr;

struct NonProjectionRenderQueryCache;
typedef std::shared_ptr<NonProjectionRenderQueryCache> NPRQueryCacheShPtr;

struct RawPixelData {
  int width;
  int height;
  int numChannels;
  const std::shared_ptr<unsigned char> pixels;
  const std::shared_ptr<uint32_t> rowIdsA;
  const std::shared_ptr<uint32_t> rowIdsB;
  const std::shared_ptr<int32_t> tableIds;

  RawPixelData()
      : width(0), height(0), numChannels(4), pixels(nullptr), rowIdsA(nullptr), rowIdsB(nullptr), tableIds(nullptr) {}
  RawPixelData(const int width, const int height)
      : width(width), height(height), numChannels(4), rowIdsA(nullptr), rowIdsB(nullptr), tableIds(nullptr) {}
  RawPixelData(const int width,
               const int height,
               const int numChannels,
               const std::shared_ptr<unsigned char> pixels,
               const std::shared_ptr<uint32_t> rowIdsA,
               const std::shared_ptr<uint32_t> rowIdsB,
               const std::shared_ptr<int32_t> tableIds)
      : width(width),
        height(height),
        numChannels(numChannels),
        pixels(pixels),
        rowIdsA(rowIdsA),
        rowIdsB(rowIdsB),
        tableIds(tableIds) {}

  bool isEmpty() const { return (!pixels || width == 0 || height == 0); }
};

struct HitInfo {
  TableId tableId;
  uint64_t rowId;
  uint8_t vegaDataId;

  HitInfo(const TableId tableId, const uint64_t rowId, const uint8_t vegaDataId)
      : tableId(tableId), rowId(rowId), vegaDataId(vegaDataId) {}
};
typedef std::pair<TableId, decltype(TableDescriptor::tableName)> TableIdNamePair;

struct RenderQueryExecuteData {
  std::shared_ptr<ResultRows> result_rows;
  std::vector<TargetMetaInfo> target_meta_info;
  std::vector<TableIdNamePair> referenced_tables;
  int64_t execute_time_ms;
  bool in_situ_data;  // true if the results of the query are written directly to opengl buffers without going to CPU.
  std::shared_ptr<QueryDataLayout> vbo_data_layout;
  std::shared_ptr<QueryDataLayout> ubo_data_layout;
};

typedef std::function<
    RenderQueryExecuteData(RenderQueryExecuteTimer&, Executor*, const std::string&, const rapidjson::Value*, bool)>
    QueryExecCB;

std::string to_string(const UserWidgetIdPair& value);
std::string to_string(const RefType refType);

}  // namespace QueryRenderer

std::ostream& operator<<(std::ostream& os, const QueryRenderer::UserWidgetIdPair& value);
std::ostream& operator<<(std::ostream& os, const QueryRenderer::RefType refType);

#endif  // QUERYRENDERER_TYPES_H_

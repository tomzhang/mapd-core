#ifndef QUERYRENDERER_TYPES_H_
#define QUERYRENDERER_TYPES_H_

#include <memory>

namespace QueryRenderer {

typedef size_t GpuId;

struct PngData;

struct UserWidgetIdPair {
  const int userId;
  const int widgetId;

  UserWidgetIdPair(int userId, int widgetId) : userId(userId), widgetId(widgetId) {}
};

struct RootPerGpuData;
typedef std::weak_ptr<RootPerGpuData> RootPerGpuDataWkPtr;
typedef std::shared_ptr<RootPerGpuData> RootPerGpuDataShPtr;

class QueryRenderManager;

struct QueryDataLayout;
typedef std::shared_ptr<QueryDataLayout> QueryDataLayoutShPtr;

class QueryRenderer;
typedef std::unique_ptr<QueryRenderer> QueryRendererUqPtr;

enum class RefEventType { UPDATE = 0, REMOVE, REPLACE, ALL };
class QueryRendererContext;
typedef std::shared_ptr<QueryRendererContext> QueryRendererContextShPtr;

std::string to_string(const UserWidgetIdPair& value);

}  // namespace QueryRenderer

std::ostream& operator<<(std::ostream& os, const QueryRenderer::UserWidgetIdPair& value);

#endif  // QUERYRENDERER_TYPES_H_

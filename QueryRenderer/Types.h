#ifndef QUERYRENDERER_TYPES_H_
#define QUERYRENDERER_TYPES_H_

#include <memory>

namespace QueryRenderer {

typedef size_t GpuId;

struct PngData;
class QueryRenderManager;

class QueryRenderer;
typedef std::unique_ptr<QueryRenderer> QueryRendererUqPtr;

class QueryVertexBuffer;
typedef std::shared_ptr<QueryVertexBuffer> QueryVertexBufferShPtr;

class QueryResultVertexBuffer;
typedef std::shared_ptr<QueryResultVertexBuffer> QueryResultVertexBufferShPtr;

enum class RefEventType { UPDATE = 0, REMOVE, REPLACE, ALL };
class QueryRendererContext;
typedef std::shared_ptr<QueryRendererContext> QueryRendererContextShPtr;

class QueryFramebuffer;
typedef std::unique_ptr<QueryFramebuffer> QueryFramebufferUqPtr;

enum class QueryDataType { UINT = 0, INT, FLOAT, DOUBLE, COLOR, STRING };
enum class QueryDataTableType { SQLQUERY = 0, EMBEDDED, URL, UNSUPPORTED };

class BaseQueryDataTableVBO;
typedef std::unique_ptr<BaseQueryDataTableVBO> QueryDataTableVBOUqPtr;
typedef std::shared_ptr<BaseQueryDataTableVBO> QueryDataTableVBOShPtr;

class BaseMark;
typedef std::shared_ptr<BaseMark> MarkShPtr;

class BaseScale;
typedef std::shared_ptr<BaseScale> ScaleShPtr;

}  // namespace QueryRenderer

#endif  // QUERYRENDERER_TYPES_H_

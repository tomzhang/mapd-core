#ifndef QUERYRENDERER_INTEROP_TYPES_H_
#define QUERYRENDERER_INTEROP_TYPES_H_

#include <memory>

namespace QueryRenderer {

class QueryVertexBuffer;
typedef std::shared_ptr<QueryVertexBuffer> QueryVertexBufferShPtr;

class QueryResultVertexBuffer;
typedef std::shared_ptr<QueryResultVertexBuffer> QueryResultVertexBufferShPtr;

class QueryIndexBuffer;
typedef std::shared_ptr<QueryIndexBuffer> QueryIndexBufferShPtr;

class QueryResultIndexBuffer;
typedef std::shared_ptr<QueryResultIndexBuffer> QueryResultIndexBufferShPtr;

class QueryUniformBuffer;
typedef std::shared_ptr<QueryUniformBuffer> QueryUniformBufferShPtr;

class QueryResultUniformBuffer;
typedef std::shared_ptr<QueryResultUniformBuffer> QueryResultUniformBufferShPtr;

class QueryIndirectVbo;
typedef std::shared_ptr<QueryIndirectVbo> QueryIndirectVboShPtr;

class QueryResultIndirectVbo;
typedef std::shared_ptr<QueryResultIndirectVbo> QueryResultIndirectVboShPtr;

class QueryIndirectIbo;
typedef std::shared_ptr<QueryIndirectIbo> QueryIndirectIboShPtr;

class QueryResultIndirectIbo;
typedef std::shared_ptr<QueryResultIndirectIbo> QueryResultIndirectIboShPtr;

}  // namespace QueryRenderer

#endif  // QUERYRENDERER_INTEROP_TYPES_H_

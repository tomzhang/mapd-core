#ifndef QUERYRENDERER_INTEROP_TYPES_H_
#define QUERYRENDERER_INTEROP_TYPES_H_

#include <memory>

namespace QueryRenderer {

class QueryBuffer;
typedef std::shared_ptr<QueryBuffer> QueryBufferShPtr;
typedef std::weak_ptr<QueryBuffer> QueryBufferWkPtr;

class QueryVertexBuffer;
typedef std::shared_ptr<QueryVertexBuffer> QueryVertexBufferShPtr;
typedef std::weak_ptr<QueryVertexBuffer> QueryVertexBufferWkPtr;

class QueryResultVertexBuffer;
typedef std::shared_ptr<QueryResultVertexBuffer> QueryResultVertexBufferShPtr;
typedef std::weak_ptr<QueryResultVertexBuffer> QueryResultVertexBufferWkPtr;

class QueryIndexBuffer;
typedef std::shared_ptr<QueryIndexBuffer> QueryIndexBufferShPtr;
typedef std::weak_ptr<QueryIndexBuffer> QueryIndexBufferWkPtr;

class QueryResultIndexBuffer;
typedef std::shared_ptr<QueryResultIndexBuffer> QueryResultIndexBufferShPtr;
typedef std::weak_ptr<QueryResultIndexBuffer> QueryResultIndexBufferWkPtr;

class QueryUniformBuffer;
typedef std::shared_ptr<QueryUniformBuffer> QueryUniformBufferShPtr;
typedef std::weak_ptr<QueryUniformBuffer> QueryUniformBufferWkPtr;

class QueryResultUniformBuffer;
typedef std::shared_ptr<QueryResultUniformBuffer> QueryResultUniformBufferShPtr;
typedef std::weak_ptr<QueryResultUniformBuffer> QueryResultUniformBufferWkPtr;

class QueryIndirectVbo;
typedef std::shared_ptr<QueryIndirectVbo> QueryIndirectVboShPtr;
typedef std::weak_ptr<QueryIndirectVbo> QueryIndirectVboWkPtr;

class QueryResultIndirectVbo;
typedef std::shared_ptr<QueryResultIndirectVbo> QueryResultIndirectVboShPtr;
typedef std::weak_ptr<QueryResultIndirectVbo> QueryResultIndirectVboWkPtr;

class QueryIndirectIbo;
typedef std::shared_ptr<QueryIndirectIbo> QueryIndirectIboShPtr;
typedef std::weak_ptr<QueryIndirectIbo> QueryIndirectIboWkPtr;

class QueryResultIndirectIbo;
typedef std::shared_ptr<QueryResultIndirectIbo> QueryResultIndirectIboShPtr;
typedef std::weak_ptr<QueryResultIndirectIbo> QueryResultIndirectIboWkPtr;

}  // namespace QueryRenderer

#endif  // QUERYRENDERER_INTEROP_TYPES_H_

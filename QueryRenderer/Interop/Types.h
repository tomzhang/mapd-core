#ifndef QUERYRENDERER_INTEROP_TYPES_H_
#define QUERYRENDERER_INTEROP_TYPES_H_

#include <memory>

namespace QueryRenderer {

class QueryResultVertexBuffer;
typedef std::shared_ptr<QueryResultVertexBuffer> QueryResultVertexBufferShPtr;

}  // namespace QueryRenderer

#endif  // QUERYRENDERER_INTEROP_TYPES_H_

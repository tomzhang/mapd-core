#ifndef QUERYRENDERER_DATA_TYPES_H_
#define QUERYRENDERER_DATA_TYPES_H_

#include <memory>

namespace QueryRenderer {

enum class QueryDataType { UINT = 0, INT, FLOAT, DOUBLE, COLOR, STRING };
enum class QueryDataTableType { SQLQUERY = 0, EMBEDDED, URL, UNSUPPORTED };

class BaseQueryDataTableVBO;
typedef std::unique_ptr<BaseQueryDataTableVBO> QueryDataTableVBOUqPtr;
typedef std::shared_ptr<BaseQueryDataTableVBO> QueryDataTableVBOShPtr;

}  // namespace QueryRenderer

#endif  // QUERYRENDERER_DATA_TYPES_H_

#ifndef QUERYRENDERER_DATA_TYPES_H_
#define QUERYRENDERER_DATA_TYPES_H_

#include <memory>

namespace QueryRenderer {

enum class QueryDataType { UINT = 0, INT, FLOAT, DOUBLE, UINT64, INT64, COLOR, STRING, BOOL, POLYGON_DOUBLE };
enum class QueryDataTableType { SQLQUERY = 0, EMBEDDED, URL, UNSUPPORTED };
enum class QueryDataTableBaseType { BASIC_VBO = 0, POLY, UNSUPPORTED };

class BaseQueryDataTable;
typedef std::unique_ptr<BaseQueryDataTable> QueryDataTableUqPtr;
typedef std::shared_ptr<BaseQueryDataTable> QueryDataTableShPtr;

class BaseQueryDataTableJSON;
typedef std::unique_ptr<BaseQueryDataTableJSON> QueryDataTableJSONUqPtr;
typedef std::shared_ptr<BaseQueryDataTableJSON> QueryDataTableJSONShPtr;

class BaseQueryDataTableSQL;
typedef std::shared_ptr<BaseQueryDataTableSQL> QueryDataTableSqlShPtr;

class BaseQueryDataTableVBO;
typedef std::unique_ptr<BaseQueryDataTableVBO> QueryDataTableVBOUqPtr;
typedef std::shared_ptr<BaseQueryDataTableVBO> QueryDataTableVBOShPtr;

class BaseQueryPolyDataTable;
typedef std::unique_ptr<BaseQueryPolyDataTable> QueryPolyDataTableUqPtr;
typedef std::shared_ptr<BaseQueryPolyDataTable> QueryPolyDataTableShPtr;

std::string to_string(QueryDataType dataType);
std::string to_string(QueryDataTableType dataTableType);
std::string to_string(QueryDataTableBaseType dataTableBaseType);

}  // namespace QueryRenderer

std::ostream& operator<<(std::ostream& os, QueryRenderer::QueryDataType value);
std::ostream& operator<<(std::ostream& os, QueryRenderer::QueryDataTableType value);
std::ostream& operator<<(std::ostream& os, QueryRenderer::QueryDataTableBaseType value);

#endif  // QUERYRENDERER_DATA_TYPES_H_

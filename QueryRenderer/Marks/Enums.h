#ifndef QUERYRENDERER_MARKS_ENUMS_H_
#define QUERYRENDERER_MARKS_ENUMS_H_

#include <string>

namespace QueryRenderer {

enum class GeomType { POINTS = 0, POLYS };  // LINES
enum class LineJoinType { BEVEL = 0, ROUND, MITER };

std::string to_string(const LineJoinType value);
int convertStringToLineJoinEnum(const std::string& val);

}  // namespace QueryRenderer

std::ostream& operator<<(std::ostream& os, const QueryRenderer::LineJoinType value);

#endif  // QUERYRENDERER_MARKS_ENUMS_H_

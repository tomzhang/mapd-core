#ifndef QUERYRENDERER_MARKS_TYPES_H_
#define QUERYRENDERER_MARKS_TYPES_H_

#include <memory>

namespace QueryRenderer {

class BaseMark;
typedef std::shared_ptr<BaseMark> GeomConfigShPtr;

class BaseRenderProperty;

}  // namespace QueryRenderer

#endif  // QUERYRENDERER_MARKS_TYPES_H_

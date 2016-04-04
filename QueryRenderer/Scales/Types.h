#ifndef QUERYRENDERER_SCALES_TYPES_H_
#define QUERYRENDERER_SCALES_TYPES_H_

#include <memory>

namespace QueryRenderer {

enum class ScaleType { LINEAR = 0, ORDINAL, UNDEFINED };

class BaseScaleDomainRangeData;

class BaseScale;
typedef std::shared_ptr<BaseScale> ScaleShPtr;

class BaseScaleRef;
typedef std::shared_ptr<BaseScaleRef> ScaleRefShPtr;

std::string to_string(ScaleType scaleType);

}  // namespace QueryRenderer

#endif  // QUERYRENDERER_SCALES_TYPES_H_

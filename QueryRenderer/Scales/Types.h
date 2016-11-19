#ifndef QUERYRENDERER_SCALES_TYPES_H_
#define QUERYRENDERER_SCALES_TYPES_H_

#include <memory>

namespace QueryRenderer {

enum class ScaleType { LINEAR = 0, LOG, POW, SQRT, ORDINAL, QUANTIZE, UNDEFINED };
enum class AccumulatorType : uint8_t {
  MIN = 0x1,
  MAX = 0x2,
  BLEND = 0x4,
  DENSITY = 0x8,
  UNDEFINED = DENSITY + 1,
  ALL = 0xF
};

class BaseScaleDomainRangeData;
typedef std::shared_ptr<BaseScaleDomainRangeData> ScaleDomainRangeDataShPtr;

class BaseScale;
typedef std::shared_ptr<BaseScale> ScaleShPtr;

class BaseScaleRef;
typedef std::shared_ptr<BaseScaleRef> ScaleRefShPtr;

std::string to_string(ScaleType scaleType);
std::string to_string(AccumulatorType accumType);

bool isQuantitativeScale(const ScaleType type);

}  // namespace QueryRenderer

#endif  // QUERYRENDERER_SCALES_TYPES_H_

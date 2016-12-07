#include "BaseColor4f.h"

namespace Rendering {
namespace Colors {

const Validators::Clamp0to1f BaseOpacityValidator::opacityValidator = Validators::Clamp0to1f();

const PackedFloatConverters::ConvertUInt8To0to1 BaseOpacityConvertToFloat::opacityConvertToFloatChannel =
    PackedFloatConverters::ConvertUInt8To0to1();

}  // namespace Colors
}  // namespace Rendering

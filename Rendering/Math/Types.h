#ifndef RENDERING_MATH_TYPES_H
#define RENDERING_MATH_TYPES_H

namespace Rendering {

namespace Math {

template <typename T, int DIMS = 2>
class AABox;

typedef AABox<float, 2> AABox2f;
typedef AABox<double, 2> AABox2d;

}  // namespace Math

}  // namespace Rendering

#endif  // RENDERING_MATH_TYPES_H

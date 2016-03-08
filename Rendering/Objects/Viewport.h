#ifndef RENDERING_OBJECTS_VIEWPORT_H_
#define RENDERING_OBJECTS_VIEWPORT_H_

#include "../Math/AABox.h"

namespace Rendering {

namespace Objects {

class Viewport : public Math::AABox<int, 2> {
 public:
  enum Index { MIN_X = 0, MIN_Y, MAX_X, MAX_Y };

  Viewport() : Math::AABox<int, 2>() {}
  explicit Viewport(int x, int y, int w, int h) : Math::AABox<int, 2>({x, y, x + w, y + h}) {}
  ~Viewport() {}

  int getXPos() const { return _data[MIN_X]; }
  void setXPos(int x) {
    int currWidth = getWidth();
    _data[MIN_X] = x;
    _data[MAX_X] = x + currWidth;
  }

  int getYPos() const { return _data[1]; }
  void setYPos(int y) {
    int currHeight = getHeight();
    _data[MIN_X] = y;
    _data[MAX_X] = y + currHeight;
  }

  int getWidth() const { return _data[MAX_X] - _data[MIN_X]; }
  void setWidth(int width) { _data[MAX_X] = _data[MIN_X] + width; }

  int getHeight() const { return _data[MAX_Y] - _data[MIN_Y]; }
  void setHeight(int height) { _data[MAX_Y] = _data[MIN_Y] + height; }

  /**
   * Placeholder for when zooming becomes necessary
   */
  Viewport& zoom() { return *this; }
};

}  // namespace Objects

}  // namespace Rendering

#endif  // RENDERING_OBJECTS_VIEWPORT_H_

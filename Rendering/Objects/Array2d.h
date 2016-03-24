#ifndef RENDERING_OBJECTS_ARRAY2D_H_
#define RENDERING_OBJECTS_ARRAY2D_H_

#include "../RenderError.h"
#include <cstring>
#include <memory>

namespace Rendering {

namespace Objects {

enum class WrapType { CLAMP, REPEAT, INVALIDATE, USE_DEFAULT };

template <typename T>
class Array2d {
 public:
  Array2d(size_t width, size_t height, T defaultVal = T())
      : defaultVal(defaultVal), width(0), height(0), data(), rows() {
    _initialize(width, height);
  }

  size_t getWidth() const { return width; }
  size_t getHeight() const { return height; }

  void resize(size_t width, size_t height) { _initialize(width, height); }

  void copyFrom(const Array2d<T>& src,
                size_t srcX = 0,
                size_t srcY = 0,
                size_t dstX = 0,
                size_t dstY = 0,
                int copyWidth = -1,
                int copyHeight = -1,
                WrapType wrapType = WrapType::INVALIDATE) {
    size_t widthToUse = (copyWidth < 0 ? width : copyWidth);
    size_t heightToUse = (copyHeight < 0 ? height : copyHeight);
    size_t endX = srcX + widthToUse;
    size_t endY = srcY + heightToUse;
    size_t srcWidth = src.getWidth();
    size_t srcHeight = src.getHeight();

    RUNTIME_EX_ASSERT(dstX < width && dstX + widthToUse <= width && dstY < height && dstY + heightToUse <= height,
                      "The copy area: " + std::to_string(widthToUse) + "x" + std::to_string(heightToUse) +
                          " starting at [" + std::to_string(dstX) + ", " + std::to_string(dstY) +
                          "] extends beyond the bounds of the array: " + std::to_string(width) + "x" +
                          std::to_string(height) + ".");

    RUNTIME_EX_ASSERT(wrapType != WrapType::INVALIDATE || (endX <= srcWidth && endY <= srcHeight),
                      "Cannot copy array data. The copy area: " + std::to_string(widthToUse) + "x" +
                          std::to_string(heightToUse) + " starting at [" + std::to_string(srcX) + ", " +
                          std::to_string(srcY) + "] extends beyond the bounds of the src 2d array: " +
                          std::to_string(srcWidth) + "x" + std::to_string(srcHeight) + ".");

    // get the in-bounds pixels, in both dimensions
    size_t inStartX = (srcX < srcWidth ? srcX : 0);
    size_t inEndX = (endX <= srcWidth ? endX : srcWidth);
    size_t inStartY = (srcY < srcHeight ? srcY : 0);
    size_t inEndY = (endY <= srcHeight ? endY : srcHeight);
    int inWidth = static_cast<int>(inEndX) - static_cast<int>(inStartX);
    int inHeight = static_cast<int>(inEndY) - static_cast<int>(inStartY);

    const std::vector<T*>& srcRows = src.rows;
    std::vector<T*>& dstRows = rows;

    if (inStartX < inEndX && inStartY < inEndY) {
      for (size_t sy = inStartY, dy = dstY; sy < inEndY; ++sy, ++dy) {
        std::copy(&srcRows[sy][inStartX], &srcRows[sy][inStartX + inWidth], &dstRows[dy][dstX]);
      }
    }

    int outWidth = static_cast<int>(endX) - static_cast<int>(srcWidth);
    int outHeight = static_cast<int>(endY) - static_cast<int>(srcHeight);
    bool wrapX = outWidth > 0;
    bool wrapY = outHeight > 0;
    if (wrapX || wrapY) {
      size_t xidx;
      switch (wrapType) {
        // case WrapType::CLAMP:
        // case WrapType::REPEAT:
        case WrapType::USE_DEFAULT:
          if (wrapX) {
            xidx = dstX + inWidth;
            for (size_t dy = dstY; dy < dstY + inHeight; ++dy) {
              std::fill(&dstRows[dy][xidx], &dstRows[dy][xidx + outWidth], defaultVal);
            }
          }
          if (wrapY) {
            for (size_t dy = dstY + inHeight; dy < outHeight + dstY + inHeight; ++dy) {
              std::fill(&dstRows[dy][dstX], &dstRows[dy][dstX + inWidth], defaultVal);
            }
          }
          if (wrapX && wrapY) {
            xidx = dstX + inWidth;
            for (size_t dy = dstY + inHeight; dy < outHeight + dstY + inHeight; ++dy) {
              std::fill(&dstRows[dy][xidx], &dstRows[dy][xidx + outWidth], defaultVal);
            }
          }
          break;
        default:
          THROW_RUNTIME_EX("Unsupported wrap type: " + std::to_string(static_cast<int>(wrapType)));
          break;
      }
    }
  }

  void copyFromPixelCenter(const Array2d<T>& src,
                           size_t srcCtrX,
                           size_t srcCtrY,
                           size_t dstCtrX,
                           size_t dstCtrY,
                           size_t radiusWidth,
                           size_t radiusHeight,
                           WrapType wrapType = WrapType::INVALIDATE) {
    size_t widthToUse = radiusWidth * 2 + 1;
    size_t heightToUse = radiusHeight * 2 + 1;
    int srcX = static_cast<int>(srcCtrX) - static_cast<int>(radiusWidth);
    int srcY = static_cast<int>(srcCtrY) - static_cast<int>(radiusHeight);
    int dstX = static_cast<int>(dstCtrX) - static_cast<int>(radiusWidth);
    int dstY = static_cast<int>(dstCtrY) - static_cast<int>(radiusWidth);
    size_t endX = srcX + widthToUse;
    size_t endY = srcY + heightToUse;
    size_t srcWidth = src.getWidth();
    size_t srcHeight = src.getHeight();

    RUNTIME_EX_ASSERT(dstX >= 0 && dstX < static_cast<int>(width) && dstX + widthToUse <= width && dstY >= 0 &&
                          dstY < static_cast<int>(height) && dstY + heightToUse <= height,
                      "The copy area: " + std::to_string(widthToUse) + "x" + std::to_string(heightToUse) +
                          " starting at [" + std::to_string(dstX) + ", " + std::to_string(dstY) +
                          "] extends beyond the bounds of the array: " + std::to_string(width) + "x" +
                          std::to_string(height) + ".");

    RUNTIME_EX_ASSERT(
        wrapType != WrapType::INVALIDATE || (srcX >= 0 && endX <= srcWidth && srcY >= 0 && endY <= srcHeight),
        "Cannot copy array data. The copy area: " + std::to_string(widthToUse) + "x" + std::to_string(heightToUse) +
            " starting at [" + std::to_string(srcX) + ", " + std::to_string(srcY) +
            "] extends beyond the bounds of the src 2d array: " + std::to_string(srcWidth) + "x" +
            std::to_string(srcHeight) + ".");

    // get the in-bounds pixels, in both dimensions
    size_t xidx, yidx;
    int inStartX = std::max((srcX < static_cast<int>(srcWidth) ? srcX : 0), 0);
    int inEndX = (endX <= srcWidth ? endX : srcWidth);
    int inStartY = std::max((srcY < static_cast<int>(srcHeight) ? srcY : 0), 0);
    int inEndY = (endY <= srcHeight ? endY : srcHeight);
    int inWidth = inEndX - inStartX;
    int inHeight = inEndY - inStartY;

    int outNegWidth = std::max(-srcX, 0);
    int outNegHeight = std::max(-srcY, 0);

    const std::vector<T*>& srcRows = src.rows;
    std::vector<T*>& dstRows = rows;

    if (inStartX < inEndX && inStartY < inEndY) {
      xidx = dstX + outNegWidth;
      for (int sy = inStartY, dy = dstY + outNegHeight; sy < inEndY; ++sy, ++dy) {
        std::copy(&srcRows[sy][inStartX], &srcRows[sy][inStartX + inWidth], &dstRows[dy][xidx]);
      }
    }

    int outPosWidth = static_cast<int>(endX) - static_cast<int>(srcWidth);
    int outPosHeight = static_cast<int>(endY) - static_cast<int>(srcHeight);

    bool wrapNegX = outNegWidth > 0;
    bool wrapNegY = outNegHeight > 0;
    bool wrapPosX = outPosWidth > 0;
    bool wrapPosY = outPosHeight > 0;

    if (wrapNegX || wrapNegY || wrapPosX || wrapPosY) {
      switch (wrapType) {
        // case WrapType::CLAMP:
        // case WrapType::REPEAT:
        case WrapType::USE_DEFAULT:
          if (wrapNegX) {
            xidx = dstX + outNegWidth;
            yidx = dstY + outNegHeight;
            for (size_t dy = yidx; dy < yidx + inHeight; ++dy) {
              std::fill(&dstRows[dy][dstX], &dstRows[dy][xidx], defaultVal);
            }
          }
          if (wrapNegY) {
            xidx = dstX + outNegWidth;
            for (size_t dy = dstY; dy < dstY + static_cast<size_t>(outNegHeight); ++dy) {
              std::fill(&dstRows[dy][xidx], &dstRows[dy][xidx + inWidth], defaultVal);
            }
          }
          if (wrapNegX && wrapNegY) {
            xidx = dstX + outNegWidth;
            for (size_t dy = dstY; dy < dstY + static_cast<size_t>(outNegHeight); ++dy) {
              std::fill(&dstRows[dy][dstX], &dstRows[dy][xidx], defaultVal);
            }
          }
          if (wrapPosX) {
            xidx = dstX + outNegWidth + inWidth;
            yidx = dstY + outNegHeight;
            for (size_t dy = yidx; dy < yidx + static_cast<size_t>(inHeight); ++dy) {
              std::fill(&dstRows[dy][xidx], &dstRows[dy][xidx + outPosWidth], defaultVal);
            }
          }
          if (wrapPosY) {
            xidx = dstX + outNegWidth;
            yidx = dstY + outNegHeight + inHeight;
            for (size_t dy = yidx; dy < static_cast<size_t>(yidx + outPosHeight); ++dy) {
              std::fill(&dstRows[dy][xidx], &dstRows[dy][xidx + inWidth], defaultVal);
            }
          }
          if (wrapPosX && wrapPosY) {
            xidx = dstX + outNegWidth + inWidth;
            yidx = dstY + outNegHeight + inHeight;
            for (size_t dy = yidx; dy < static_cast<size_t>(yidx + outPosHeight); ++dy) {
              std::fill(&dstRows[dy][xidx], &dstRows[dy][xidx + outPosWidth], defaultVal);
            }
          }
          if (wrapPosX && wrapNegY) {
            xidx = dstX + outNegWidth + inWidth;
            for (size_t dy = dstY; dy < dstY + static_cast<size_t>(outNegHeight); ++dy) {
              std::fill(&dstRows[dy][xidx], &dstRows[dy][xidx + outPosWidth], defaultVal);
            }
          }
          if (wrapNegX && wrapPosY) {
            xidx = dstX + outNegWidth;
            yidx = dstY + outNegHeight + inHeight;
            for (size_t dy = yidx; dy < yidx + static_cast<size_t>(outPosHeight); ++dy) {
              std::fill(&dstRows[dy][dstX], &dstRows[dy][xidx], defaultVal);
            }
          }
          break;
        default:
          THROW_RUNTIME_EX("Unsupported wrap type: " + std::to_string(static_cast<int>(wrapType)));
          break;
      }
    }
  }

  T get(size_t xidx, size_t yidx) {
    RUNTIME_EX_ASSERT(xidx < width,
                      "Invalid x index: " + std::to_string(xidx) + ". It must be < " + std::to_string(width) + ".");
    RUNTIME_EX_ASSERT(yidx < height,
                      "Invalid y index: " + std::to_string(yidx) + ". It must be < " + std::to_string(height) + ".");
    return rows[yidx][xidx];
  }

  // TODO(croot): this class is intended to be used for standard types, hence no use of a reference for 'val'
  // but what if this class is used with classes/structs? This would result in an undesirable copy for 'val'
  void set(size_t xidx, size_t yidx, T val) {
    RUNTIME_EX_ASSERT(
        xidx < width,
        "Invalid x index: " + std::to_string(xidx) + ". Width of 2d array is " + std::to_string(width) + ".");
    RUNTIME_EX_ASSERT(
        yidx < height,
        "Invalid y index: " + std::to_string(yidx) + ". Height of 2d array is " + std::to_string(height) + ".");
    rows[yidx][xidx] = val;
  }

  T* operator[](size_t rowidx) {
    RUNTIME_EX_ASSERT(
        rowidx < height,
        "Invalid height index: " + std::to_string(rowidx) + ". Height of 2d array is " + std::to_string(height) + ".");
    return rows[rowidx];
  }

  const T* operator[](size_t rowidx) const {
    RUNTIME_EX_ASSERT(
        rowidx < height,
        "Invalid height index: " + std::to_string(rowidx) + ". Height of 2d array is " + std::to_string(height) + ".");
    return rows[rowidx];
  }

  T* getDataPtr() {
    RUNTIME_EX_ASSERT(
        data.size() > 0,
        "The 2d array is empty (" + std::to_string(width) + "x" + std::to_string(height) + "). Cannot retrieve data.");
    return &data[0];
  }
  const T* getDataPtr() const {
    RUNTIME_EX_ASSERT(
        data.size() > 0,
        "The 2d array is empty (" + std::to_string(width) + "x" + std::to_string(height) + "). Cannot retrieve data.");
    return &data[0];
  }

 private:
  T defaultVal;
  size_t width;
  size_t height;

  std::vector<T> data;
  std::vector<T*> rows;

  void _initialize(size_t newWidth, size_t newHeight) {
    if (newWidth != width || newHeight != height) {
      size_t numElems = newHeight * newWidth;

      newWidth = (numElems > 0 ? newWidth : 0);
      newHeight = (numElems > 0 ? newHeight : 0);

      data.resize(numElems);
      rows.resize(newHeight);

      // TODO(croot): Should we do a shrink_to_fit() to clear out unused memory
      // when necessary? Would this be costly? If so, perhaps find a metric by
      // which shrink_to_fit is called?

      if (numElems) {
        // TODO(croot): preserve as much of the original data
        // as possible on a resize or just reset everything?
        // Note: right now we're just resetting everything
        std::fill(data.begin(), data.end(), defaultVal);

        for (size_t i = 0; i < newHeight; ++i) {
          // TODO(croot): expose a simple Row class that will handle the [] operators,
          // and thus handle any out-of-bounds indices
          rows[i] = &data[i * newWidth];
        }
      }

      width = newWidth;
      height = newHeight;
    }
  }
};

}  // namespace Objects

}  // namespace Rendering

#endif  // RENDERING_OBJECTS_ARRAY2D_H_

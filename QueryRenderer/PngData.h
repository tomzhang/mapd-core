#ifndef QUERYRENDERER_PNGDATA_H_
#define QUERYRENDERER_PNGDATA_H_

#include <memory>

namespace QueryRenderer {

struct PngData {
  std::shared_ptr<char> pngDataPtr;
  int pngSize;

  // PngData(const std::shared_ptr<char>& pngDataPtr, int pngSize) : pngDataPtr(pngDataPtr), pngSize(pngSize) {}

  // TODO(croot): Create a Pixels struct which will havve the width, height, & num channels self-contained.
  PngData();
  explicit PngData(int width, int height, const std::shared_ptr<unsigned char>& pixelsPtr, int compressionLevel = -1);

  void writeToFile(const std::string& filename);
};

}  // namespace QueryRenderer

#endif  // QUERYRENDERER_PNGDATA_H_

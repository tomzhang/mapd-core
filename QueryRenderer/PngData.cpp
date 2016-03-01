#include "PngData.h"
#include <png.h>
#include <vector>
#include <fstream>
#include <cstring>
#include <assert.h>

namespace QueryRenderer {

std::shared_ptr<char> pngDataPtr;
int pngSize;

static void writePngData(png_structp png_ptr, png_bytep data, png_size_t length) {
  std::vector<char>* pngData = reinterpret_cast<std::vector<char>*>(png_get_io_ptr(png_ptr));
  size_t currSz = pngData->size();
  pngData->resize(currSz + length);
  std::memcpy(&(*pngData)[0] + currSz, data, length);
}

static void flushPngData(png_structp) {
}

PngData::PngData() : pngDataPtr(nullptr), pngSize(0) {
}

PngData::PngData(int width, int height, const std::shared_ptr<unsigned char>& pixelsPtr, int compressionLevel)
    : pngDataPtr(nullptr), pngSize(0) {
  if (!pixelsPtr) {
    throw std::runtime_error("Cannot create PngData(). The pixels are empty.");
  }

  if (compressionLevel < -1 || compressionLevel > 9) {
    throw std::runtime_error("Invalid compression level argument " + std::to_string(compressionLevel) +
                             ". Must be a value between 0 and 9 and -1 would mean use default.");
  }

  png_structp png_ptr = png_create_write_struct(PNG_LIBPNG_VER_STRING, nullptr, nullptr, nullptr);
  assert(png_ptr != nullptr);

  png_infop info_ptr = png_create_info_struct(png_ptr);
  assert(info_ptr != nullptr);

  // TODO(croot) - rather than going the setjmp route, you can enable the
  // PNG_SETJMP_NOT_SUPPORTED compiler flag which would result in asserts
  // when libpng errors, according to its docs.
  // if (setjmp(png_jmpbuf(png_ptr))) {
  //   std::cerr << "Got a libpng error" << std::endl;
  //   // png_destroy_info_struct(png_ptr, &info_ptr);
  //   png_destroy_write_struct(&png_ptr, &info_ptr);
  //   assert(false);
  // }

  // using a vector to store the png bytes. I'm doing this to take advantage of the
  // optimized allocation vectors do when resizing. The only downside of this approach
  // is that the vector maintains the memory, so I have to copy the vector's internal
  // memory to my own buffer
  // TODO(croot) - I could just use a vector of bytes/chars instead of
  // a shared_ptr<char>(new char[]), but I'd have to be sure to do a "shrink-to-fit" on
  // the vector if I did this to deallocate any unused memory. This might be just
  // as costly as a full memcpy --- or maybe not since the vector's memory itself is
  // also fully deallocated -- this might be a better approach.
  std::vector<char> pngData;

  png_set_write_fn(png_ptr, &pngData, writePngData, flushPngData);

  // set filtering?
  png_set_filter(png_ptr, PNG_FILTER_TYPE_BASE, PNG_FILTER_NONE);
  // png_set_filter(png_ptr, PNG_FILTER_TYPE_BASE, PNG_FILTER_SUB);
  // png_set_filter(png_ptr, PNG_FILTER_TYPE_BASE, PNG_FILTER_UP);
  // png_set_filter(png_ptr, PNG_FILTER_TYPE_BASE, PNG_FILTER_AVG);
  // png_set_filter(png_ptr, PNG_FILTER_TYPE_BASE, PNG_FILTER_PAETH);
  // png_set_filter(png_ptr, PNG_FILTER_TYPE_BASE, PNG_ALL_FILTERS);

  // set filter weights/preferences? I can't seem to get this
  // to make a difference
  // double weights[3] = {2.0, 1.5, 1.1};
  // double costs[PNG_FILTER_VALUE_LAST] = {2.0, 2.0, 1.0, 2.0, 2.0};
  // png_set_filter_heuristics(png_ptr, PNG_FILTER_HEURISTIC_WEIGHTED, 3, weights, costs);

  // set zlib compression level
  // if (compressionLevel >= 0) {
  //  png_set_compression_level(png_ptr, compressionLevel);
  //}
  png_set_compression_level(png_ptr, compressionLevel);

  // other zlib params?
  // png_set_compression_mem_level(png_ptr, 8);
  // png_set_compression_strategy(png_ptr, PNG_Z_DEFAULT_STRATEGY);
  // png_set_compression_window_bits(png_ptr, 15);
  // png_set_compression_method(png_ptr, 8);
  // png_set_compression_buffer_size(png_ptr, 8192);

  // skip the 8 bytes signature?
  // png_set_sig_bytes(png_ptr, 8);

  int interlace_type = PNG_INTERLACE_NONE;  // or PNG_INTERLACE_ADAM7 if we ever want interlacing
  png_set_IHDR(png_ptr,
               info_ptr,
               width,
               height,
               8,
               PNG_COLOR_TYPE_RGB_ALPHA,
               interlace_type,
               PNG_COMPRESSION_TYPE_DEFAULT,
               PNG_FILTER_TYPE_DEFAULT);

  /* write out the PNG header info (everything up to first IDAT) */
  png_write_info(png_ptr, info_ptr);

  // make sure < 8-bit images are packed into pixels as tightly as possible - only necessary
  // for palette images, which we're not doing yet
  // png_set_packing(png_ptr);

  unsigned char* pixels = pixelsPtr.get();
  png_byte* row_pointers[height];

  for (int j = 0; j < height; ++j) {
    // invert j -- input pixel rows go bottom up, where pngs are
    // defined top-down.
    row_pointers[j] = &pixels[(height - j - 1) * width * 4];
  }

  png_write_image(png_ptr, row_pointers);

  // can alternatively write per-row, but this didn't
  // seem to make a difference. I thought that perhaps
  // this could be parallelized, but png_write_row() doesn't
  // appear to be a fixed-function call.
  // for (j = 0; j < height; ++j) {
  //   png_write_row(png_ptr, row_pointers[j]);
  // }

  png_write_end(png_ptr, info_ptr);

  pngSize = pngData.size();
  pngDataPtr.reset(new char[pngSize], std::default_delete<char[]>());
  char* pngRawData = pngDataPtr.get();
  std::memcpy(pngRawData, &pngData[0], pngSize);

  png_destroy_write_struct(&png_ptr, &info_ptr);
}

void PngData::writeToFile(const std::string& filename) {
  if (!pngDataPtr) {
    throw std::runtime_error("Cannot write file " + filename + ". The pixels are empty.");
  }
  std::ofstream pngFile(filename, std::ios::binary);
  pngFile.write(pngDataPtr.get(), pngSize);
  pngFile.close();
}

}  // namespace QueryRenderer

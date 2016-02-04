#include "EglUtils.h"
#include <EGL/egl.h>
#include "../../../Settings/BaseSettings.h"
#include "../../../RenderError.h"

namespace Rendering {

using Settings::IntConstant;
using Settings::IntSetting;
using Settings::BaseSettings;

namespace GL {
namespace EGL {

void appendFbAttrs(EglGLRenderer* renderer,
                   IntConstant drawableType,
                   const BaseSettings& settings,
                   std::vector<int>& attributes) {
  // Check for special framebuffer configurations and add the appropiate
  // attributes for choosing an fb config

  // Setup the color profile

  // for now, all buffers will use an RGB color buffer. Perhaps we'll support a luminance buffer
  // at some point

  attributes.push_back(EGL_COLOR_BUFFER_TYPE);
  attributes.push_back(EGL_RGB_BUFFER);

  int32_t colorSize = settings.getIntSetting(IntSetting::BITS_RGBA);  // number of bits per color channel
  IntConstant colorConstant = Settings::convertToIntConstant(colorSize);
  if (colorConstant != IntConstant::OFF) {
    switch (colorConstant) {
      case IntConstant::RGBA16F:
      case IntConstant::RGBA32F:
        // TODO(croot): Would we use a Luminance buffer here?
        THROW_RUNTIME_EX("EGL float framebuffers are not currently supported.");
        // attributes.push_back(GLX_RENDER_TYPE);
        // attributes.push_back(GLX_RGBA_FLOAT_BIT);
        colorSize = (colorConstant == IntConstant::RGBA16F ? 16 : 32);
        break;
      case IntConstant::DEFAULT:
      case IntConstant::AUTO:
      case IntConstant::ON:
        colorSize = 8;
        break;
      default:
        break;
    }

    RUNTIME_EX_ASSERT(colorSize > 0,
                      "EGL Setting: " + std::to_string(colorSize) + " is an invalid value for the BITS_RGBA setting");

    attributes.push_back(EGL_RED_SIZE);
    attributes.push_back(colorSize);
    attributes.push_back(EGL_GREEN_SIZE);
    attributes.push_back(colorSize);
    attributes.push_back(EGL_BLUE_SIZE);
    attributes.push_back(colorSize);

    LOG(INFO) << "EGL Setting: " << IntSetting::BITS_RGBA << " = " << colorSize << ".";

    // number of bits in the alpha channel. Expose this in settings info?
    int32_t alphaSize = settings.getIntSetting(IntSetting::BITS_ALPHA);
    IntConstant alphaConstant = Settings::convertToIntConstant(alphaSize);
    switch (alphaConstant) {
      case IntConstant::OFF:
        break;

      case IntConstant::DEFAULT:
      case IntConstant::AUTO:
      case IntConstant::UNDEFINED:
      case IntConstant::ON:
        alphaSize = colorSize;  // do not break. Let pass thru
      default:
        if (alphaSize < 0) {
          LOG(WARNING) << "EGL Setting: " << alphaSize << " is an invalid value for BITS_ALPHA setting. Using "
                       << colorSize << ".";
          alphaSize = colorSize;
        }
        attributes.push_back(EGL_ALPHA_SIZE);
        attributes.push_back(alphaSize);

        LOG(INFO) << "EGL Setting: " << IntSetting::BITS_ALPHA << " = " << alphaSize << ".";
        break;
    }
  }

  // Number of bits in the depth channel.
  int32_t depthSize = settings.getIntSetting(IntSetting::BITS_DEPTH);
  IntConstant depthConstant = Settings::convertToIntConstant(depthSize);
  if (depthSize > 0 || depthConstant == IntConstant::AUTO) {
    if (depthConstant == IntConstant::AUTO || depthConstant == IntConstant::ON) {
      depthSize = 1;  // TODO(croot): expose a default depth size
    }
    attributes.push_back(EGL_DEPTH_SIZE);
    attributes.push_back(depthSize);
    LOG(INFO) << "EGL Setting: " << IntSetting::BITS_DEPTH << " = " << depthSize << ".";
  }

  // number of bits in the stencil channel
  int32_t stencilSize = settings.getIntSetting(IntSetting::BITS_STENCIL);
  IntConstant stencilConstant = Settings::convertToIntConstant(stencilSize);
  if (stencilSize > 0 || stencilConstant == IntConstant::AUTO) {
    if (stencilConstant == IntConstant::AUTO || stencilConstant == IntConstant::ON) {
      stencilSize = 1;  // TODO(croot): expose a default stencil size
    }
    attributes.push_back(EGL_STENCIL_SIZE);
    attributes.push_back(stencilSize);  // TODO(croot): expose a default stencil size
    LOG(INFO) << "EGL Setting: " << IntSetting::BITS_STENCIL << " = " << stencilSize << ".";
  }

  // handle accum buffer
  int32_t accumSize = settings.getIntSetting(IntSetting::BITS_ACCUM_RGBA);
  IntConstant accumConstant = Settings::convertToIntConstant(accumSize);

  int32_t accumAlphaSize = settings.getIntSetting(IntSetting::BITS_ACCUM_ALPHA);
  // IntConstant accumAlphaConstant = Settings::convertToIntConstant(accumAlphaSize);

  if (accumSize > 0 || accumConstant == IntConstant::AUTO) {
    THROW_RUNTIME_EX("EGL accumulation buffers are not currently supported.");
  } else if (accumAlphaSize > 0) {
    THROW_RUNTIME_EX("EGL accumulation buffers are not currently supported.");
  }

  // handle multi-sampling
  int32_t sampleSize = settings.getIntSetting(IntSetting::NUM_SAMPLES_PER_PIXEL);
  IntConstant sampleConstant = Settings::convertToIntConstant(sampleSize);
  if (sampleSize > 0 || sampleConstant == IntConstant::AUTO) {
    if (sampleConstant == IntConstant::AUTO || sampleConstant == IntConstant::ON) {
      sampleSize = 1;  // TODO(croot): expose a default stencil size
    }
    attributes.push_back(EGL_SAMPLE_BUFFERS);
    attributes.push_back(1);
    attributes.push_back(EGL_SAMPLES);
    attributes.push_back(sampleSize);

    LOG(INFO) << "EGL Setting: " << IntSetting::NUM_SAMPLES_PER_PIXEL << " = " << sampleSize << ".";
  }
}

}  // namespace EGL
}  // namespace GL
}  // namespace Rendering

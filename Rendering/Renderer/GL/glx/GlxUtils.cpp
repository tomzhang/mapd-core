#include "GlxUtils.h"
#include "GlxGLRenderer.h"
#include "../../../RenderError.h"
#include <vector>

namespace Rendering {

using Settings::BaseSettings;
using Settings::IntSetting;
using Settings::IntConstant;

namespace GL {
namespace GLX {

/**
 * NOTE: a GlxGLRenderer is passed in as an argument to chooseFbConfigFromSettings
 * to appease the glxewGetContext requirement for GLEW_MX-related extension/version
 * checks.
 */

#ifdef GLEW_MX
#define glxewGetContext renderer->glxewGetContext
#endif

void appendFbBufferAttrs(GlxGLRenderer* renderer,
                         IntConstant drawableType,
                         const BaseSettings& settings,
                         std::vector<int>& attributes) {
  // Check for special framebuffer configurations and add the appropiate
  // attributes for choosing an fb config

  // Setup the color profile
  int32_t colorSize = settings.getIntSetting(IntSetting::BITS_RGBA);  // number of bits per color channel
  IntConstant colorConstant = Settings::convertToIntConstant(colorSize);
  if (colorConstant != IntConstant::OFF) {
    if (drawableType == IntConstant::FBO || drawableType == IntConstant::UNDEFINED) {
      LOG(INFO) << "GLX Setting: " << IntSetting::DRAWABLE_TYPE << " = " << drawableType
                << ". Number of bits per channel is forced to be 8.";
      colorSize = 8;
    } else {
      switch (colorConstant) {
        case IntConstant::RGBA16F:
        case IntConstant::RGBA32F:
          // Need GlxGLRenderer* here, which should have called glxewInit()
          // if GLEW_MX is enabled.
          RUNTIME_EX_ASSERT(GLXEW_ARB_fbconfig_float, "Error: GlxGLWindow: GLX framebuffer doesn't support floats.");
          attributes.push_back(GLX_RENDER_TYPE);
#ifdef GLX_RGBA_FLOAT_BIT_ARB
          attributes.push_back(GLX_RGBA_FLOAT_BIT_ARB);
#else
          attributes.push_back(GLX_RGBA_FLOAT_BIT);
#endif  // GLX_RGBA_FLOAT_BIT_ARB
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
    }

    RUNTIME_EX_ASSERT(colorSize > 0,
                      "GLX Setting: " + std::to_string(colorSize) + " is an invalid value for the BITS_RGBA setting");

    attributes.push_back(GLX_RED_SIZE);
    attributes.push_back(colorSize);
    attributes.push_back(GLX_GREEN_SIZE);
    attributes.push_back(colorSize);
    attributes.push_back(GLX_BLUE_SIZE);
    attributes.push_back(colorSize);

    LOG(INFO) << "GLX Setting: " << IntSetting::BITS_RGBA << " = " << colorSize << ".";

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
          LOG(WARNING) << "GLX Setting: " << alphaSize << " is an invalid value for BITS_ALPHA setting. Using "
                       << colorSize << ".";
          alphaSize = colorSize;
        }
        attributes.push_back(GLX_ALPHA_SIZE);
        attributes.push_back(alphaSize);

        LOG(INFO) << "GLX Setting: " << IntSetting::BITS_ALPHA << " = " << alphaSize << ".";
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
    attributes.push_back(GLX_DEPTH_SIZE);
    attributes.push_back(depthSize);
    LOG(INFO) << "GLX Setting: " << IntSetting::BITS_DEPTH << " = " << depthSize << ".";
  }

  // number of bits in the stencil channel
  int32_t stencilSize = settings.getIntSetting(IntSetting::BITS_STENCIL);
  IntConstant stencilConstant = Settings::convertToIntConstant(stencilSize);
  if (stencilSize > 0 || stencilConstant == IntConstant::AUTO) {
    if (stencilConstant == IntConstant::AUTO || stencilConstant == IntConstant::ON) {
      stencilSize = 1;  // TODO(croot): expose a default stencil size
    }
    attributes.push_back(GLX_STENCIL_SIZE);
    attributes.push_back(stencilSize);  // TODO(croot): expose a default stencil size
    LOG(INFO) << "GLX Setting: " << IntSetting::BITS_STENCIL << " = " << stencilSize << ".";
  }

  // handle accum buffer
  int32_t accumSize = settings.getIntSetting(IntSetting::BITS_ACCUM_RGBA);
  IntConstant accumConstant = Settings::convertToIntConstant(accumSize);

  int32_t accumAlphaSize = settings.getIntSetting(IntSetting::BITS_ACCUM_ALPHA);
  IntConstant accumAlphaConstant = Settings::convertToIntConstant(accumAlphaSize);

  if (accumSize > 0 || accumConstant == IntConstant::AUTO) {
    if (accumConstant == IntConstant::AUTO || accumConstant == IntConstant::ON) {
      accumSize = 1;  // TODO(croot): expose a default accum size
    }

    attributes.push_back(GLX_ACCUM_RED_SIZE);
    attributes.push_back(accumSize);
    attributes.push_back(GLX_ACCUM_GREEN_SIZE);
    attributes.push_back(accumSize);
    attributes.push_back(GLX_ACCUM_BLUE_SIZE);
    attributes.push_back(accumSize);

    LOG(INFO) << "GLX Setting: " << IntSetting::BITS_ACCUM_RGBA << " = " << accumSize << ".";

    if (accumAlphaConstant == IntConstant::AUTO || accumAlphaConstant == IntConstant::ON ||
        accumAlphaConstant == IntConstant::DEFAULT) {
      accumAlphaSize = accumSize;
    } else if (accumAlphaSize < 0) {
      LOG(WARNING) << accumAlphaSize << " is an invalid value for " << IntSetting::BITS_ACCUM_ALPHA
                   << " window setting. Using default.";
      accumAlphaSize = accumSize;
    }

    attributes.push_back(GLX_ACCUM_ALPHA_SIZE);
    attributes.push_back(accumAlphaSize);

    LOG(INFO) << "GLX Setting: " << IntSetting::BITS_ACCUM_ALPHA << " = " << accumAlphaSize << ".";
  } else if (accumAlphaSize > 0) {
    attributes.push_back(GLX_ACCUM_ALPHA_SIZE);
    attributes.push_back(accumAlphaSize);

    LOG(INFO) << "GLX Setting: " << IntSetting::BITS_ACCUM_ALPHA << " = " << accumAlphaSize << ".";
  }

  // handle multi-sampling
  int32_t sampleSize = settings.getIntSetting(IntSetting::NUM_SAMPLES_PER_PIXEL);
  IntConstant sampleConstant = Settings::convertToIntConstant(sampleSize);
  if (sampleSize > 0 || sampleConstant == IntConstant::AUTO) {
    if (sampleConstant == IntConstant::AUTO || sampleConstant == IntConstant::ON) {
      sampleSize = 1;  // TODO(croot): expose a default stencil size
    }
    attributes.push_back(GLX_SAMPLE_BUFFERS);
    attributes.push_back(1);
    attributes.push_back(GLX_SAMPLES);
    attributes.push_back(sampleSize);

    LOG(INFO) << "GLX Setting: " << IntSetting::NUM_SAMPLES_PER_PIXEL << " = " << sampleSize << ".";
  }
}

}  // namespace GLX
}  // namespace GL
}  // namespace Rendering

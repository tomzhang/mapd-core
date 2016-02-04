#include "RendererSettings.h"
#include "WindowSettings.h"
#include "../RenderError.h"

namespace Rendering {

namespace Settings {

RendererSettings::RendererSettings()
    : BaseSettings({IntSetting::OPENGL_MAJOR,
                    IntSetting::OPENGL_MINOR,
                    IntSetting::USE_CORE_PROFILE,

                    // only need to color and ancillary buffer info for renderers
                    IntSetting::BITS_RGBA,
                    IntSetting::BITS_ALPHA,
                    IntSetting::BITS_DEPTH,
                    IntSetting::BITS_STENCIL,
                    IntSetting::BITS_ACCUM_RGBA,
                    IntSetting::BITS_ACCUM_ALPHA,
                    IntSetting::NUM_SAMPLES_PER_PIXEL},

                   {StrSetting::DISPLAY}) {
}

RendererSettings::RendererSettings(const RendererSettings& rendererSettings) : BaseSettings(rendererSettings) {
}

RendererSettings::RendererSettings(const WindowSettings& windowSettings) : RendererSettings() {
  copyFrom(windowSettings);
}

RendererSettings::~RendererSettings() {
}

RendererSettings& RendererSettings::operator=(const RendererSettings& rhs) {
  BaseSettings::operator=(rhs);
  return *this;
}

bool RendererSettings::operator==(const RendererSettings& rhs) const {
  return BaseSettings::operator==(rhs);
}

bool RendererSettings::isCompatible(const WindowSettings& rhs) const {
  return rhs.isCompatible(*this);
}

}  // namespace Settings

}  // namespace Rendering

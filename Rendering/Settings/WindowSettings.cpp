#include "WindowSettings.h"

namespace Rendering {

namespace Settings {

WindowSettings::WindowSettings()
    : BaseSettings({IntSetting::GPU_ID,
                    IntSetting::USE_DOUBLEBUFFER,
                    IntSetting::USE_FULLSCREEN,
                    IntSetting::USE_DECORATION,
                    // IntSetting::USE_STEREO,

                    IntSetting::DRAWABLE_TYPE,

                    IntSetting::X_POS,
                    IntSetting::Y_POS,
                    IntSetting::WIDTH,
                    IntSetting::HEIGHT,

                    IntSetting::BITS_RGBA,
                    IntSetting::BITS_ALPHA,
                    IntSetting::BITS_DEPTH,
                    IntSetting::BITS_STENCIL,
                    IntSetting::BITS_ACCUM_RGBA,
                    IntSetting::BITS_ACCUM_ALPHA,
                    IntSetting::NUM_SAMPLES_PER_PIXEL},

                   {StrSetting::NAME, StrSetting::DISPLAY}),
      _viewport() {
}

WindowSettings::WindowSettings(const WindowSettings& windowSettings) : BaseSettings(windowSettings) {
  _viewport = windowSettings._viewport;
}

WindowSettings::~WindowSettings() {
}

WindowSettings& WindowSettings::operator=(const WindowSettings& rhs) {
  BaseSettings::operator=(rhs);
  _viewport = rhs._viewport;

  return *this;
}

bool WindowSettings::operator==(const WindowSettings& rhs) const {
  return (BaseSettings::operator==(rhs) && _viewport == rhs._viewport);
}

bool WindowSettings::isCompatible(const RendererSettings& rendererSettings) const {
  // TODO(croot): fill this method out
  return false;
}

void WindowSettings::_postSetIntSetting(const std::string& setting, int32_t val) {
  // TODO(croot): make a static version of the IntSetting enums as strings to avoid
  // recreating them all the time.
  if (setting == Rendering::Settings::to_string(IntSetting::X_POS)) {
    _viewport.setXPos(val);
  } else if (setting == Rendering::Settings::to_string(IntSetting::Y_POS)) {
    _viewport.setYPos(val);
  } else if (setting == Rendering::Settings::to_string(IntSetting::WIDTH)) {
    _viewport.setWidth(val);
  } else if (setting == Rendering::Settings::to_string(IntSetting::HEIGHT)) {
    _viewport.setHeight(val);
  }
}

Objects::Viewport WindowSettings::getViewport() const {
  return _viewport;
}

void WindowSettings::setViewport(int x, int y, int w, int h) {
  _viewport.initSizeFromLocation({x, y}, {w, h});

  // NOTE: calling setSetting() here instead of setIntSetting() method
  // in order to avoid recalling a viewport.set...() method

  // TODO(croot): make a static version of the IntSetting enums as strings to avoid
  // recreating them all the time.
  setSetting(_intSettings, Rendering::Settings::to_string(IntSetting::X_POS), x);
  setSetting(_intSettings, Rendering::Settings::to_string(IntSetting::Y_POS), y);
  setSetting(_intSettings, Rendering::Settings::to_string(IntSetting::WIDTH), w);
  setSetting(_intSettings, Rendering::Settings::to_string(IntSetting::HEIGHT), h);
}

void WindowSettings::setViewport(const Objects::Viewport& viewport) {
  setViewport(_viewport.getXPos(), _viewport.getYPos(), _viewport.getWidth(), _viewport.getHeight());
}

std::string WindowSettings::getName() const {
  return getStrSetting(StrSetting::NAME);
}

int32_t WindowSettings::getViewportSetting(const WindowSettings& settings,
                                           IntSetting viewportSetting,
                                           int32_t defaultVal) {
  int32_t rtn = settings.getIntSetting(viewportSetting);
  IntConstant val = convertToIntConstant(rtn);
  switch (val) {
    case IntConstant::OFF:
      LOG(WARNING) << val << " is an invalid value for " << viewportSetting << " window setting. Using default ("
                   << defaultVal << ").";

    case IntConstant::UNDEFINED:
    case IntConstant::DEFAULT:
    case IntConstant::AUTO:
    case IntConstant::ON:
      rtn = defaultVal;
      break;
    default:
      break;
  }

  return rtn;
}

}  // namespace Settings

}  // namespace Rendering

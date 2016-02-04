#include "IntSettings.h"
#include "../RenderError.h"

namespace Rendering {

namespace Settings {

void setSetting(IntSettingsMap& settings, const std::string& setting, int32_t val) {
  auto itr = settings.find(setting);
  RUNTIME_EX_ASSERT(itr != settings.end(), setting + " is not a valid window int setting.");

  itr->second = val;
}

std::string to_string(const IntConstant value) {
  switch (value) {
    case IntConstant::UNDEFINED:
      return "UNDEFINED";
    case IntConstant::RGBA32F:
      return "RGBA32F";
    case IntConstant::RGBA16F:
      return "RGBA16F";
    case IntConstant::FBO:
      return "FBO";
    case IntConstant::PBUFFER:
      return "PBUFFER";
    case IntConstant::WINDOW:
      return "WINDOW";
    case IntConstant::DEFAULT:
      return "DEFAULT";
    case IntConstant::AUTO:
      return "AUTO";
    case IntConstant::OFF:
      return "OFF";
    case IntConstant::ON:
      return "ON";
    default:
      return std::to_string(static_cast<int>(value));
  }

  return "";
}

std::string to_string(const IntSetting value) {
  switch (value) {
    case IntSetting::OPENGL_MAJOR:
      return "OPENGL_MAJOR";
    case IntSetting::OPENGL_MINOR:
      return "OPENGL_MINOR";
    case IntSetting::USE_CORE_PROFILE:
      return "USE_CORE_PROFILE";
    case IntSetting::GPU_ID:
      return "GPU_ID";
    case IntSetting::USE_DOUBLEBUFFER:
      return "USE_DOUBLEBUFFER";
    case IntSetting::USE_FULLSCREEN:
      return "USE_FULLSCREEN";
    case IntSetting::USE_DECORATION:
      return "USE_DECORATION";
    // case  IntSetting::USE_STEREO:
    //     return "USE_STEREO";
    case IntSetting::DRAWABLE_TYPE:
      return "DRAWABLE_TYPE";
    case IntSetting::X_POS:
      return "X_POS";
    case IntSetting::Y_POS:
      return "Y_POS";
    case IntSetting::WIDTH:
      return "WIDTH";
    case IntSetting::HEIGHT:
      return "HEIGHT";
    case IntSetting::BITS_RGBA:
      return "BITS_RGBA";
    case IntSetting::BITS_ALPHA:
      return "BITS_ALPHA";
    case IntSetting::BITS_DEPTH:
      return "BITS_DEPTH";
    case IntSetting::BITS_STENCIL:
      return "BITS_STENCIL";
    case IntSetting::BITS_ACCUM_RGBA:
      return "BITS_ACCUM_RGBA";
    case IntSetting::BITS_ACCUM_ALPHA:
      return "BITS_ACCUM_ALPHA";
    case IntSetting::NUM_SAMPLES_PER_PIXEL:
      return "NUM_SAMPLES_PER_PIXEL";
    case IntSetting::ALL:
      return "ALL";
    default:
      return std::to_string(static_cast<int>(value));
  }

  return "";
}

}  // namespace Settings

}  // namespace Rendering

std::ostream& operator<<(std::ostream& os, const Rendering::Settings::IntSetting value) {
  os << to_string(value);
  return os;
}

std::ostream& operator<<(std::ostream& os, const Rendering::Settings::IntConstant value) {
  os << to_string(value);
  return os;
}

#include "StrSettings.h"
#include "../RenderError.h"

namespace Rendering {

namespace Settings {

void setSetting(StrSettingsMap& settings, const std::string& setting, const std::string& val) {
  auto itr = settings.find(setting);
  RUNTIME_EX_ASSERT(itr != settings.end(), setting + " is not a valid window string setting.");

  itr->second = val;
}

std::string to_string(const StrSetting value) {
  switch (value) {
    case StrSetting::NAME:
      return "NAME";
    case StrSetting::DISPLAY:
      return "DISPLAY";
    case StrSetting::ALL:
      return "ALL";
    default:
      return std::to_string(static_cast<int>(value));
  }

  return "";
}

std::ostream& operator<<(std::ostream& os, const StrSetting value) {
  os << to_string(value);
  return os;
}

}  // namespace Settings

}  // namespace Rendering

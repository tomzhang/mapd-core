#ifndef RENDERING_SETTINGS_STRSETTINGS_H_
#define RENDERING_SETTINGS_STRSETTINGS_H_

#include <string>
#include <ostream>
#include <set>
#include <unordered_map>

namespace Rendering {

namespace Settings {

/**
 * Enum describing all possible string settings for windows/renderers.
 */
enum class StrSetting {
  /**
   * Renderer-specific settings
   */

  /**
   * Window-specific settings
   */

  /**
   * Settings applicable to both windows and renderers
   */
  NAME,  // The name of the window/renderer.
         // If IntSetting::UseDecoration is activated for a window,
         // then this will be the title of the window.

  DISPLAY,  // The X11 display server to use.

  ALL  // LEAVE AS LAST ITEM
};

typedef std::set<StrSetting> StrSettings;
typedef std::unordered_map<std::string, std::string> StrSettingsMap;

void setSetting(StrSettingsMap& settings, const std::string& setting, const std::string& val);

std::string to_string(const StrSetting value);

}  // namespace Settings

}  // namespace Rendering

std::ostream& operator<<(std::ostream& os, const Rendering::Settings::StrSetting value);

#endif  // RENDERING_SETTINGS_STRSETTINGS_H_

#ifndef RENDERING_SETTINGS_INTSETTINGS_H_
#define RENDERING_SETTINGS_INTSETTINGS_H_

#include <cstdint>
#include <string>
#include <ostream>
#include <set>
#include <unordered_map>

namespace Rendering {

namespace Settings {
/**
 * Enum describing constants that can be used with any IntSetting
 */
enum class IntConstant : int32_t {
  UNDEFINED = -0xfffffff,  // Undefined value
  RGBA32F = -7,            // Float32 framebuffer (WindowSettings::BITS_RGBA)
  RGBA16F = -6,            // Float16 framebuffer (WindowSettings::BITS_RGBA)

  FBO = -5,      // FBO drawable (WindowSettings::DRAWABLE_TYPE)
  PBUFFER = -4,  // PBuffer drawable (WindowSettings::DRAWABLE_TYPE)
  WINDOW = -3,   // Window drawable (WindowSettings::DRAWABLE_TYPE)

  /**
   * Nicest statistics gathering (WindowSettings::HINT_STATISTICS,
   * Channel::HINT_STATISTICS)
   */
  // NICEST = -2,

  DEFAULT = -2,
  AUTO = -1,  // Automatic selection (various attributes)
  OFF = 0,    // disabled (various attributes)
  ON = 1,     // enabled (various attributes)

  /**
   * Fastest statistics gathering (WindowSettings::HINT_STATISTICS,
   * Channel::HINT_STATISTICS)
   */
  // FASTEST = ON,
  // HORIZONTAL = ON  // Horizontal load-balancing
};

/**
 * Enum describing all possible int settings for windows/renderers.
 */
enum class IntSetting {
  /**
   * Renderer-specific settings
   */
  OPENGL_MAJOR,      // Major version for GL context creation
  OPENGL_MINOR,      // Minor version for GL context creation
  USE_CORE_PROFILE,  // Core profile context if possible

  /**
   * Window-specific settings
   */
  GPU_ID,            // The id of the GPU to use: 0 - max # gpus
  USE_DOUBLEBUFFER,  // Front and back buffer
  USE_FULLSCREEN,    // Fullscreen drawable
  USE_DECORATION,    // Window decorations
  // USE_STEREO,         // Active stereo

  // Window, pbuffer, FBO or OFF
  DRAWABLE_TYPE,

  // viewport-related attrs
  X_POS,
  Y_POS,
  WIDTH,
  HEIGHT,

  /**
   * Int settings used in both windows and renderers.
   * TODO(croot): verify these
   */
  BITS_RGBA,              // No of per-component color planes
  BITS_ALPHA,             // No of alpha planes
  BITS_DEPTH,             // No of z-buffer planes
  BITS_STENCIL,           // No of stencil planes
  BITS_ACCUM_RGBA,        // No of accumulation buffer planes
  BITS_ACCUM_ALPHA,       // No of alpha accum buffer planes
  NUM_SAMPLES_PER_PIXEL,  // No of multisample (AA) planes

  // TODO(croot): What are the following settings? Look them up
  // and determine if they should be exposed.
  // HINT_SWAPSYNC,       // Swap sync on vertical retrace
  // HINT_STATISTICS,     // Statistics gathering hint
  // HINT_SCREENSAVER,    // Screensaver (de)activation (WGL)
  // HINT_GRAB_POINTER,   // Capture mouse outside window

  ALL  // LEAVE AS LAST ITEM
};

typedef std::set<IntSetting> IntSettings;
typedef std::unordered_map<std::string, int32_t> IntSettingsMap;

inline IntConstant convertToIntConstant(int32_t val) {
  return static_cast<IntConstant>(val);
}

void setSetting(IntSettingsMap& settings, const std::string& setting, int32_t val);

std::string to_string(const IntSetting value);
std::string to_string(const IntConstant value);

}  // namespace Settings

}  // namespace Rendering

std::ostream& operator<<(std::ostream& os, const Rendering::Settings::IntSetting value);
std::ostream& operator<<(std::ostream& os, const Rendering::Settings::IntConstant value);

#endif  // RENDERING_SETTINGS_INTSETTINGS_H_

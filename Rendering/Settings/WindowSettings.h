#ifndef RENDERING_SETTINGS_WINDOWSETTINGS_H_
#define RENDERING_SETTINGS_WINDOWSETTINGS_H_

#include "BaseSettings.h"
#include "Types.h"
#include "../Objects/Viewport.h"

namespace Rendering {

namespace Settings {

class WindowSettings : public BaseSettings {
 public:
  WindowSettings();
  WindowSettings(const WindowSettings& windowSettings);

  ~WindowSettings();

  WindowSettings& operator=(const WindowSettings& rhs);
  bool operator==(const WindowSettings& rhs) const;

  bool isCompatible(const RendererSettings& rendererSettings) const;

  /**
   * Viewport-related funcs.
   */
  Objects::Viewport getViewport() const;  // A copy of the settings' viewport
  void setViewport(int x, int y, int w, int h);
  void setViewport(const Objects::Viewport& viewport);

  std::string getName() const;

  /**
   * Helper funcs
   */
  // static bool isIntConstant(int32_t val);

  static int32_t getViewportSetting(const WindowSettings& settings, IntSetting viewportSetting, int32_t defaultVal);

  // friend class Window;

 private:
  Objects::Viewport _viewport;

  void _postSetIntSetting(const std::string& setting, int32_t val) final;
};

}  // namespace Settings

}  // namespace Rendering

#endif  // RENDERING_SETTINGS_WINDOWSETTINGS_H_

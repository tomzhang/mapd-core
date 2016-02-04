#ifndef RENDERING_SETTINGS_RENDERERSETTINGS_H_
#define RENDERING_SETTINGS_RENDERERSETTINGS_H_

#include "BaseSettings.h"
#include "Types.h"

namespace Rendering {

namespace Settings {

class RendererSettings : public BaseSettings {
 public:
  RendererSettings();
  RendererSettings(const RendererSettings& rendererSettings);
  RendererSettings(const WindowSettings& windowSettings);
  ~RendererSettings();

  RendererSettings& operator=(const RendererSettings& rhs);
  bool operator==(const RendererSettings& rhs) const;

  bool isCompatible(const WindowSettings& rhs) const;

  // friend class Window;
};

}  // namespace Settings

}  // namespace Rendering

#endif  // RENDERING_SETTINGS_RENDERERSETTINGS_H_

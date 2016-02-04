#ifndef RENDERING_SETTINGS_BASESETTINGS_H_
#define RENDERING_SETTINGS_BASESETTINGS_H_

#include "IntSettings.h"
#include "StrSettings.h"
#include <set>
#include <cstdint>
#include <unordered_map>
#include <string>

namespace Rendering {

namespace Settings {

class BaseSettings {
 public:
  virtual ~BaseSettings();

  void setIntSetting(const std::string& setting, int32_t val);
  void setIntSetting(const std::string& setting, IntConstant val);
  void setIntSetting(IntSetting setting, int32_t val);
  void setIntSetting(IntSetting setting, IntConstant val);

  int32_t getIntSetting(const std::string& setting) const;
  int32_t getIntSetting(IntSetting setting) const;

  IntConstant getIntSettingAsConstant(const std::string& setting) const;
  IntConstant getIntSettingAsConstant(IntSetting setting) const;

  void setStrSetting(const std::string& setting, const std::string& val);
  void setStrSetting(StrSetting setting, const std::string& val);

  std::string getStrSetting(const std::string& setting) const;
  std::string getStrSetting(StrSetting setting) const;

  void copyFrom(const BaseSettings& settings);

 protected:
  BaseSettings(IntSettings intSettingsToUse = IntSettings(), StrSettings strSettingsToUse = StrSettings());
  BaseSettings(const BaseSettings& baseSettings);

  BaseSettings& operator=(const BaseSettings& rhs);
  bool operator==(const BaseSettings& rhs) const;

  IntSettingsMap _intSettings;
  StrSettingsMap _strSettings;

 private:
  virtual void _postSetIntSetting(const std::string& setting, int32_t val) {}
  virtual void _postSetStrSetting(const std::string& setting, const std::string& val) {}
};

bool isSettingActive(const BaseSettings& settings, IntSetting isActiveSetting, bool defaultVal);

}  // namespace Settings

}  // namespace Rendering

#endif  // RENDERING_SETTINGS_BASESETTINGS_H_

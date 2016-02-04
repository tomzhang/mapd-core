#include "BaseSettings.h"
#include "../RenderError.h"
#include <algorithm>
#include <iostream>

namespace Rendering {

namespace Settings {

BaseSettings::BaseSettings(IntSettings intSettingsToUse, StrSettings strSettingsToUse)
    : _intSettings(), _strSettings() {
  if (intSettingsToUse.size()) {
    for (const auto& itr : intSettingsToUse) {
      IntSetting setting = static_cast<IntSetting>(itr);

      _intSettings.insert({{Rendering::Settings::to_string(setting), static_cast<int32_t>(IntConstant::DEFAULT)}});
    }
  } else {
    for (std::size_t i = 0; i < static_cast<std::size_t>(IntSetting::ALL); ++i) {
      IntSetting setting = static_cast<IntSetting>(i);

      _intSettings.insert({{Rendering::Settings::to_string(setting), static_cast<int32_t>(IntConstant::DEFAULT)}});
    }
  }

  if (strSettingsToUse.size()) {
    for (const auto& itr : strSettingsToUse) {
      StrSetting setting = static_cast<StrSetting>(itr);

      _strSettings.insert({{Rendering::Settings::to_string(setting), ""}});
    }
  } else {
    for (std::size_t i = 0; i < static_cast<std::size_t>(StrSetting::ALL); ++i) {
      StrSetting setting = static_cast<StrSetting>(i);

      _strSettings.insert({{Rendering::Settings::to_string(setting), ""}});
    }
  }
}

BaseSettings::BaseSettings(const BaseSettings& baseSettings) {
  _intSettings = baseSettings._intSettings;
  _strSettings = baseSettings._strSettings;
}

BaseSettings::~BaseSettings() {
}

BaseSettings& BaseSettings::operator=(const BaseSettings& rhs) {
  _intSettings = rhs._intSettings;
  _strSettings = rhs._strSettings;

  return *this;
}

bool BaseSettings::operator==(const BaseSettings& rhs) const {
  return (_intSettings == rhs._intSettings && _strSettings == rhs._strSettings);
}

void BaseSettings::setIntSetting(const std::string& setting, int32_t val) {
  setSetting(_intSettings, setting, val);

  _postSetIntSetting(setting, val);
}

void BaseSettings::setIntSetting(const std::string& setting, IntConstant val) {
  return setIntSetting(setting, static_cast<int32_t>(val));
}

void BaseSettings::setIntSetting(IntSetting setting, int32_t val) {
  // TODO(croot): make a static version of the IntSetting enums as strings to avoid
  // recreating them all the time.
  std::string strSetting = Rendering::Settings::to_string(setting);

  return setIntSetting(strSetting, val);
}

void BaseSettings::setIntSetting(IntSetting setting, IntConstant val) {
  return setIntSetting(setting, static_cast<int32_t>(val));
}

int32_t BaseSettings::getIntSetting(const std::string& setting) const {
  auto itr = _intSettings.find(setting);
  RUNTIME_EX_ASSERT(itr != _intSettings.end(), setting + " is not a valid window int setting.");

  return itr->second;
}

int32_t BaseSettings::getIntSetting(IntSetting setting) const {
  std::string strSetting = Rendering::Settings::to_string(setting);

  return getIntSetting(strSetting);
}

IntConstant BaseSettings::getIntSettingAsConstant(const std::string& setting) const {
  return static_cast<IntConstant>(getIntSetting(setting));
}

IntConstant BaseSettings::getIntSettingAsConstant(IntSetting setting) const {
  std::string strSetting = Rendering::Settings::to_string(setting);

  return getIntSettingAsConstant(strSetting);
}

void BaseSettings::setStrSetting(const std::string& setting, const std::string& val) {
  setSetting(_strSettings, setting, val);
  _postSetStrSetting(setting, val);
}

void BaseSettings::setStrSetting(StrSetting setting, const std::string& val) {
  std::string strSetting = Rendering::Settings::to_string(setting);
  return setStrSetting(strSetting, val);
}

std::string BaseSettings::getStrSetting(const std::string& setting) const {
  auto itr = _strSettings.find(setting);
  RUNTIME_EX_ASSERT(itr != _strSettings.end(), "\"" + setting + "\" is not a valid window string setting.");

  return itr->second;
}

std::string BaseSettings::getStrSetting(StrSetting setting) const {
  return getStrSetting(Rendering::Settings::to_string(setting));
}

void BaseSettings::copyFrom(const BaseSettings& settings) {
  for (const auto& theirItr : settings._intSettings) {
    auto myItr = _intSettings.find(theirItr.first);
    if (myItr != _intSettings.end()) {
      setIntSetting(theirItr.first, theirItr.second);
    }
  }

  for (const auto& theirItr : settings._strSettings) {
    auto myItr = _strSettings.find(theirItr.first);
    if (myItr != _strSettings.end()) {
      setStrSetting(theirItr.first, theirItr.second);
    }
  }
}

bool isSettingActive(const BaseSettings& settings, IntSetting isActiveSetting, bool defaultVal) {
  bool rtn = defaultVal;
  IntConstant isActive = settings.getIntSettingAsConstant(isActiveSetting);
  switch (isActive) {
    case IntConstant::OFF:
      rtn = false;
      break;
    case IntConstant::ON:
      rtn = true;
      break;

    default:
      // allow to pass thru
      LOG(WARNING) << isActive << " is an invalid value for " << isActiveSetting << " setting. Using default ("
                   << defaultVal << ").";

    case IntConstant::UNDEFINED:
    case IntConstant::DEFAULT:
    case IntConstant::AUTO:
      rtn = defaultVal;
      break;
  }

  return rtn;
}

}  // namespace Settings

}  // namespace Rendering

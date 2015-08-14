//==================================================================================================
///  @file Main_Unix.cpp
///
///  Implementation of DSIDriverFactory and DSIExtCollatorFactory for Unix platforms.
///
///  Copyright (C) 2008-2014 Simba Technologies Incorporated.
//==================================================================================================

#include "DSIDriverFactory.h"
#include "QSDriver.h"
#include "SimbaSettingReader.h"

#ifdef SERVERTARGET
#include "SimbaServer.h"
#endif

#include <unistd.h>

//==================================================================================================
/// @brief Creates an instance of IDriver for a driver.
///
/// The resulting object is made available through DSIDriverSingleton::GetDSIDriver().
///
/// @param out_instanceID   Unique identifier for the IDriver instance.
///
/// @return IDriver instance. (OWN)
//==================================================================================================
Simba::DSI::IDriver* Simba::DSI::DSIDriverFactory(simba_handle& out_instanceID) {
  out_instanceID = getpid();

#ifdef SERVERTARGET
  SimbaSettingReader::SetConfigurationBranding(SERVER_LINUX_BRANDING);
#else
  SimbaSettingReader::SetConfigurationBranding(DRIVER_LINUX_BRANDING);
#endif

  return new Simba::Quickstart::QSDriver();
}

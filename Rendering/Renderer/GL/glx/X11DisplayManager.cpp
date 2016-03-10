#include "X11DisplayManager.h"
#include <string>
#include <regex>
#include <cstdlib>
#include <cstdio>
#include "../../../RenderError.h"
#include <unordered_map>
#include <iostream>

namespace Rendering {
namespace GL {
namespace GLX {

struct X11GpuInfo {
  X11GpuInfo() : names(0), typeStr("") {}
  std::vector<std::string> names;
  std::string typeStr;
  // int pciBusId;
  // int pciDeviceId;
  // int pciDomainId;
};

/**
 * Gets gpu info from the 'nvidia-settings' command. The 'nvidia-settings -q gpus' command
 * gets the number of gpus on a current x display and will result in output that looks like this,
 * if command succeeds:
 *
 *  2 GPUs on greendragon:1
 *
 *     [0] greendragon:1[gpu:0] (GeForce GTX TITAN X)
 *
 *       Has the following names:
 *         GPU-0
 *         GPU-2e92c74d-6866-d25c-2ac0-89d13dde342a
 *
 *     [1] greendragon:1[gpu:1] (GeForce GTX TITAN X)
 *
 *       Has the following names:
 *         GPU-1
 *         GPU-b041871d-cfb2-392b-0ae0-334c75cd0644
 *
 */
struct X11AllGpuInfo {
  X11AllGpuInfo(const std::string displayStr = "") : displayStr(displayStr), numGpus(0), allGpuInfo() {
    std::string cmd = "nvidia-settings -q gpus";
    if (displayStr.length()) {
      cmd += " --display=" + displayStr;
    }

    // redirect stderr to stdout
    cmd += " 2>&1";

    std::shared_ptr<FILE> pipe(popen(cmd.c_str(), "r"), pclose);

    RUNTIME_EX_ASSERT(
        pipe != nullptr,
        "Error trying to run 'nvida-settings' command needed to gather gpu information for an X11 display.");

    char buffer[512];
    std::string result;
    int currGpuId = -1;
    std::string currGpuIdStr = "";
    std::string line;
    bool gettingNames = false;

    std::regex numGpusRegex("^\\s*(\\d+)\\s+GPUs{0,1}\\s+on\\s+([\\w,-.]+)[:,::](\\d+)");
    std::regex gpuInfoRegex("^\\s*\\[(\\d+)\\]\\s+([\\w,-]+)[:,::](\\d+)\\s*(\\[\\w+:\\d+\\])\\s+\\((.+)\\)\\s*$");
    std::regex hasTheFollowingNamesRegex("^\\s*Has the following names:\\s*$");
    std::regex gpuNameRegex("^\\s*(\\S+)\\s*$");
    std::smatch matches;
    while (!feof(pipe.get())) {
      if (fgets(buffer, 512, pipe.get()) != nullptr) {
        line = buffer;
        if (std::regex_search(line, matches, numGpusRegex)) {
          numGpus = std::stoi(matches[1]);
          // std::string hostname = matches[2];
          // int displayNum = std::stoi(matches[3]);
          allGpuInfo.resize(numGpus);
        } else if (numGpus) {
          if (std::regex_search(line, matches, gpuInfoRegex)) {
            currGpuId = std::stoi(matches[1]);
            // std::string hostname = matches[2];
            // int displayNum = std::stoi(matches[3]);
            currGpuIdStr = matches[4];
            allGpuInfo[currGpuId].typeStr = matches[5];

            // TODO(croot): get PCI bus info? If so, use
            // nvidia-settings -q currGpuIdStr/PCIBus
            // nvidia-settings -q currGpuIdStr/PCIDevice
            // nvidia-settings -q currGpuIdStr/PCIDomain
          } else if (currGpuIdStr.length() && !gettingNames) {
            if (std::regex_search(line, hasTheFollowingNamesRegex)) {
              gettingNames = true;
            }
          } else if (currGpuIdStr.length() && gettingNames) {
            if (std::regex_search(line, matches, gpuNameRegex)) {
              allGpuInfo[currGpuId].names.push_back(matches[1]);
            } else {
              currGpuId = -1;
              currGpuIdStr = "";
            }
          }
        }
        result += line;
      }
    }

    std::regex errRegex("\\bERROR|Failed\\b");
    RUNTIME_EX_ASSERT(!std::regex_search(result, errRegex),
                      "Error trying to run 'nvidia-settings' command which is necessary to gather gpu information for "
                      "an X11 display. Error:\n\n" +
                          result);

    // std::cout << "CROOT - nvidia-settings: " << result.length() << " " << result << std::endl;
  }

  std::string displayStr;
  size_t numGpus;
  std::vector<X11GpuInfo> allGpuInfo;
  // std::vector<std::string> pciBusId;
};

static const X11AllGpuInfo& getX11GpuInfo(const std::string& displayName = "") {
  static std::unordered_map<std::string, X11AllGpuInfo> gpuInfo;

  // TODO(croot): make this thread safe?
  std::unordered_map<std::string, X11AllGpuInfo>::iterator itr;
  if ((itr = gpuInfo.find(displayName)) == gpuInfo.end()) {
    std::pair<std::unordered_map<std::string, X11AllGpuInfo>::iterator, bool> insertPair =
        gpuInfo.insert({displayName, X11AllGpuInfo(displayName)});
    itr = insertPair.first;
  }

  return itr->second;
}

static const std::regex displayStrRegex("\\s*(\\w*)([:,::])(\\d+)\\.*(\\d*)");
std::pair<std::string, int> getKeyAndDefaultScreenFromDisplayStr(const std::string& displayStr,
                                                                 int defaultScreenOverride = -1) {
  std::smatch matches;

  if (!std::regex_match(displayStr, matches, displayStrRegex) || matches.size() != 5) {
    THROW_RUNTIME_EX("Cannot open display \"" + displayStr + "\". It is an invalid display string format.");
  }

  std::string hostname = matches[1];

  // connection can be ':' or '::'. XLib docs state that ':' indicates
  // a TCP connection and a '::' indicates a DECnet connection.
  // But if there is no hostname, we'll use ':' as the server
  // will pick the most appropriate connection in that case.
  // See: https://tronche.com/gui/x/xlib/display/opening.html
  // for more
  std::string connectionStr = matches[2];
  if (!hostname.length()) {
    connectionStr = ":";
  }

  std::string display = matches[3];
  std::string screen = matches[4];
  int defaultScreen = defaultScreenOverride;
  if (defaultScreen < 0 && screen.length()) {
    defaultScreen = std::stoi(screen);
  }

  return std::make_pair(hostname + connectionStr + display, defaultScreen);
}

void validateScreenForDisplay(Display* dpy, int screen) {
  int screenCnt = XScreenCount(dpy);
  RUNTIME_EX_ASSERT(screen < screenCnt,
                    "Invalid screen " + std::to_string(screen) + " for X display " +
                        std::to_string(XConnectionNumber(dpy)) + ". There are only " + std::to_string(screenCnt) +
                        " screens on the display");
}

X11DisplayShPtr openDisplayWithDefaultScreen(const std::string& displayStr,
                                             int& defaultScreen,
                                             std::weak_ptr<Display>& displayPtr) {
  CHECK(displayPtr.expired());

  Display* dpy = nullptr;
  if (defaultScreen < 0) {
    dpy = XOpenDisplay(displayStr.c_str());
  } else {
    dpy = XOpenDisplay((displayStr + "." + std::to_string(defaultScreen)).c_str());
  }

  if (!dpy) {
    return nullptr;
  }

  if (defaultScreen < 0) {
    defaultScreen = XDefaultScreen(dpy);
  } else {
    validateScreenForDisplay(dpy, defaultScreen);
  }

  X11DisplayShPtr rtn(dpy, XCloseDisplay);
  displayPtr = rtn;

  return rtn;
}

X11DisplayManager::X11DisplayManager() {
}

X11DisplayManager::~X11DisplayManager() {
}

size_t X11DisplayManager::getNumGpus(const std::string& displayName) const {
  return getX11GpuInfo(displayName).numGpus;
}

DisplayScreenPair X11DisplayManager::_connectToDisplay(const std::string& displayName, int defaultScreen) {
  std::string dpyStr(displayName);
  if (!displayName.length()) {
    // use the DISPLAY environment variable, if defined =
    const char* envDpy = std::getenv("DISPLAY");
    RUNTIME_EX_ASSERT(envDpy != nullptr,
                      "Attempting to open the default XServer display designated by the \"DISPLAY\" environment "
                      "variable, but \"DISPLAY\" is undefined.");

    dpyStr = envDpy;
  }

  auto keyAndScreen = getKeyAndDefaultScreenFromDisplayStr(dpyStr, defaultScreen);
  std::string key = keyAndScreen.first;
  int screen = keyAndScreen.second;

  X11DisplayShPtr rtnDisplayPtr;

  OpenDisplayMap::iterator itr;
  if ((itr = _openedDisplays.find(key)) == _openedDisplays.end()) {
    std::weak_ptr<Display> displayPtr;
    rtnDisplayPtr = openDisplayWithDefaultScreen(key, screen, displayPtr);

    RUNTIME_EX_ASSERT(rtnDisplayPtr != nullptr, "Cannot open display \"" + dpyStr + "\"");
    _openedDisplays.insert({key, std::make_pair(std::move(displayPtr), screen)});
  } else {
    rtnDisplayPtr = itr->second.first.lock();
    if (!rtnDisplayPtr) {
      rtnDisplayPtr = openDisplayWithDefaultScreen(key, screen, itr->second.first);
      itr->second.second = screen;

      rtnDisplayPtr = itr->second.first.lock();
    } else {
      if (screen < 0) {
        screen = itr->second.second;
      } else {
        validateScreenForDisplay(rtnDisplayPtr.get(), screen);
      }
    }
  }

  return std::make_pair(rtnDisplayPtr, screen);
}

DisplayScreenPair X11DisplayManager::connectToDisplay(const std::string& displayName) {
  return _connectToDisplay(displayName, -1);
}

DisplayScreenPair X11DisplayManager::connectToDisplay(const std::string& displayName, int screenId) {
  return _connectToDisplay(displayName, screenId);
}

DisplayScreenPair X11DisplayManager::connectToDisplay(int screenId) {
  return _connectToDisplay("", screenId);
}

}  // namespace GLX
}  // namespace GL
}  // namespace Rendering

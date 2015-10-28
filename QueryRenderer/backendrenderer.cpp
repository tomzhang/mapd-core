#include "QueryRenderManager.h"
#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <assert.h>
#include <iostream>
#include <sstream>
#include "rapidjson/document.h"
#include "rapidjson/error/en.h"

#include "gd.h"
#include <fstream>

using namespace MapD_Renderer;

int fbwWidthRatio, fbwHeightRatio;
int windowWidth, windowHeight;

QueryRenderManager renderManager(500000, true);
// QueryRenderManager renderManager;

UserWidgetPair userWidgetId1 = std::make_pair(1, 1);
UserWidgetPair userWidgetId2 = std::make_pair(1, 2);
UserWidgetPair userWidgetId3 = std::make_pair(2, 1);

void mouse_btn_click_callback(GLFWwindow* window, int btn, int action, int mods) {
  double xpos, ypos;
  if (btn == GLFW_MOUSE_BUTTON_LEFT && action == GLFW_PRESS) {
    glfwGetCursorPos(window, &xpos, &ypos);

    // invert y
    ypos = double(windowHeight) - ypos;

    // now compensate for a different-sized framebuffer (i.e. retina displays)
    xpos *= fbwWidthRatio;
    ypos *= fbwHeightRatio;
  }
}

void mouse_move_callback(GLFWwindow* window, double xpos, double ypos) {
  if (xpos >= 0 && xpos <= windowWidth && ypos >= 0 && ypos <= windowHeight) {
    // std::cerr << "CROOT - in mouse move callback: " << xpos << ", " << ypos << std::endl;

    // invert y
    ypos = double(windowHeight) - ypos;

    // now compensate for a different-sized framebuffer (i.e. retina displays)
    xpos *= fbwWidthRatio;
    ypos *= fbwHeightRatio;

    unsigned int id = renderManager.getIdAt(xpos, ypos);
    std::cout << "CROOT - mouse move: (" << xpos << ", " << ypos << ") " << id << std::endl;
  }
}

int main(int argc, char* argv[]) {
  // GLFWwindow* window;

  assert(argc == 2);

  std::string configJSON;

  std::ifstream inFile(argv[1]);
  inFile.seekg(0, std::ios::end);
  configJSON.reserve(inFile.tellg());
  inFile.seekg(0, std::ios::beg);
  configJSON.assign((std::istreambuf_iterator<char>(inFile)), std::istreambuf_iterator<char>());

  renderManager.addUserWidget(userWidgetId1, true, false);
  renderManager.setActiveUserWidget(userWidgetId1);

  rapidjson::Document jsonDoc;
  jsonDoc.Parse(configJSON.c_str());
  if (jsonDoc.HasParseError()) {
    throw std::runtime_error("Error parsing the json config: offset: " + std::to_string(jsonDoc.GetErrorOffset()) +
                             ", error: " + rapidjson::GetParseError_En(jsonDoc.GetParseError()));
  }

  renderManager.configureRender(jsonDoc);

  // CudaHandle cudaHandle = renderManager.getCudaHandle();

  if (renderManager.inDebugMode()) {
    double lastTime = glfwGetTime();
    unsigned int nbFrames = 0;

    // Loop until the user closes the window
    GLFWwindow* window = renderManager.getWindow();

    // glfwSetMouseButtonCallback(window, mouse_btn_click_callback);
    glfwSetCursorPosCallback(window, mouse_move_callback);

    int framebufferWidth, framebufferHeight;

    // first need to get the true framebuffer size of the window
    glfwGetWindowSize(window, &windowWidth, &windowHeight);
    glfwGetFramebufferSize(window, &framebufferWidth, &framebufferHeight);

    fbwWidthRatio = float(framebufferWidth) / float(windowWidth);
    fbwHeightRatio = float(framebufferHeight) / float(windowHeight);

    while (!glfwWindowShouldClose(window)) {
      // Poll for and process events
      glfwPollEvents();

      renderManager.render();

      // Measure speed
      double currentTime = glfwGetTime();
      nbFrames++;
      if (currentTime - lastTime >= 1.0) {  // If last prinf() was more than 1 sec ago
        // printf and reset timer
        printf("%f ms/frame\n", 1000.0 / double(nbFrames));
        nbFrames = 0;
        lastTime += 1.0;
      }

      // Swap front and back buffers
      glfwSwapBuffers(window);
    }
  } else {
    PngData pngData = renderManager.renderToPng();
    pngData.writeToFile("out.png");

    std::cout << "DONE with render" << std::endl;
  }

  return 0;
}

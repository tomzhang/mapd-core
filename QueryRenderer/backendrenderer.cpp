#include "QueryRenderer.h"
#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <assert.h>

#include "gd.h"
#include <fstream>

using namespace MapD_Renderer;

int main(int argc, char *argv[]) {
    GLFWwindow* window;

    assert(glfwInit());
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 1);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
    glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);

    window = glfwCreateWindow(1, 1, "", NULL, NULL);
    if (window == NULL) {
        glfwTerminate();
        assert(false);
    }

    glfwMakeContextCurrent(window);
    assert(!glGetError());

    glewExperimental = GL_TRUE; // needed for core profile

    assert(glewInit() == GLEW_OK);
    glGetError();  // clear error code - this always throws error but seems to not matter


    QueryRenderer renderer;

    PngData pngData = renderer.getColorNoisePNG(500, 500);
    std::ofstream pngFile("out.png", std::ios::binary);
    pngFile.write(pngData.pngDataPtr, pngData.pngSize);
    pngFile.close();


    UserWidgetIdPair userWidgetId1 = std::make_pair(1, 1);
    UserWidgetIdPair userWidgetId2 = std::make_pair(1, 2);
    UserWidgetIdPair userWidgetId3 = std::make_pair(2, 1);

    renderer.addFramebuffer(userWidgetId1, true, false);
    renderer.setActiveFramebufferById(userWidgetId1);

    renderer.addFramebuffer(userWidgetId2, true, false);
    renderer.addFramebuffer(userWidgetId3, true, false);
    renderer.setActiveFramebufferById(userWidgetId3);


    std::string configJSON =
"{"
"    \"width\" : 1000,"
"    \"height\" : 1500,"
"    \"scales\": ["
"        {"
"            \"name\" : \"x\","
"            \"type\" : \"linear\","
"            \"domain\" : [1,2],"
"            \"range\" : [1.0, 2.0]"
"        }"
"    ],"
"    \"marks\" : ["
"        {"
"            \"type\" : \"points\","
"            \"properties\" : {"
"                \"x\" : 0.0,"
"                \"y\" : 0.0,"
"                \"fillColor\" : \"rgba(255,0,0,1)\","
"                \"size\" : 50.0"
"            }"
"        }"
"    ]"
"}";

    DataTable data;
    renderer.render(data, 0, configJSON);


    return 0;
}

find_package(Boost COMPONENTS filesystem system regex REQUIRED)
find_package(OpenGL REQUIRED)
find_package(GLEW REQUIRED)
find_package(GD REQUIRED)
find_package(GLFW3 REQUIRED)

include_directories(${Boost_INCLUDE_DIR})
include_directories(${GL_INCLUDE_DIR})
include_directories(${GLEW_INCLUDE_DIR})
include_directories(${GD_INCLUDE_DIR})
include_directories(${GLFW3_INCLUDE_DIR})
include_directories(ThirdParty/rapidjson)

add_executable(backendrenderer backendrenderer.cpp QueryRenderManager.cpp QueryFramebuffer.cpp QueryRenderer.cpp QueryRendererObjects.cpp Shader.cpp BufferLayout.cpp TypeGL.cpp ShaderUtils.cpp DataTable.cpp Color.cpp)
add_library(QueryRenderer QueryRenderManager.cpp QueryFramebuffer.cpp QueryRenderer.cpp QueryRendererObjects.cpp Shader.cpp BufferLayout.cpp TypeGL.cpp ShaderUtils.cpp DataTable.cpp Color.cpp)

target_link_libraries(backendrenderer ${Boost_LIBRARIES} ${OPENGL_LIBRARIES} ${GLEW_LIBRARIES} ${GD_LIBRARIES} ${GLFW3_LIBRARIES})
target_link_libraries(QueryRenderer ${Boost_LIBRARIES} ${OPENGL_LIBRARIES} ${GLEW_LIBRARIES} ${GD_LIBRARIES} ${GLFW3_LIBRARIES})

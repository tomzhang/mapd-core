find_package(Boost COMPONENTS filesystem system regex REQUIRED)
find_package(OpenGL REQUIRED)
find_package(GLEW REQUIRED)
find_package(GD REQUIRED)
find_package(GLFW3 REQUIRED)

# set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -L/Users/croot/src/poly2tri/lib")

include_directories(${Boost_INCLUDE_DIR})
include_directories(${GL_INCLUDE_DIR})
include_directories(${GLEW_INCLUDE_DIR})
include_directories(${GD_INCLUDE_DIR})
include_directories(${GLFW3_INCLUDE_DIR})
include_directories(ThirdParty/rapidjson)

# include_directories("/Users/croot/src/poly2tri/poly2tri")

add_executable(backendrenderer backendrenderer.cpp QueryRenderer.cpp QueryFramebuffer.cpp QueryRendererConfig.cpp ScaleConfig.cpp GeomConfig.cpp Shader.cpp RenderProperty.cpp BufferLayout.cpp)
add_library(QueryRenderer QueryRenderer.cpp QueryFramebuffer.cpp QueryRendererConfig.cpp ScaleConfig.cpp GeomConfig.cpp Shader.cpp RenderProperty.cpp BufferLayout.cpp)
target_link_libraries(backendrenderer ${Boost_LIBRARIES} ${OPENGL_LIBRARIES} ${GLEW_LIBRARIES} ${GD_LIBRARIES} ${GLFW3_LIBRARIES})
target_link_libraries(QueryRenderer ${Boost_LIBRARIES} ${OPENGL_LIBRARIES} ${GLEW_LIBRARIES} ${GD_LIBRARIES} ${GLFW3_LIBRARIES})

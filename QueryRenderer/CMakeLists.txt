find_package(Boost COMPONENTS filesystem system regex REQUIRED)
find_package(OpenGL REQUIRED)
find_package(GLEW REQUIRED)
find_package(GD REQUIRED)
find_package(GLFW3 REQUIRED)
find_package(CUDA REQUIRED)
find_package(Glog REQUIRED)

include_directories(${Boost_INCLUDE_DIR})
include_directories(${GL_INCLUDE_DIR})
include_directories(${GLEW_INCLUDE_DIR})
include_directories(${GD_INCLUDE_DIR})
include_directories(${GLFW3_INCLUDE_DIR})
include_directories(${CUDA_INCLUDE_DIR})
include_directories(${Glog_INCLUDE_DIR})
include_directories(ThirdParty/rapidjson)


add_custom_command(
    DEPENDS backendrendererSetup.cu
    OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/backendrendererSetup.o
    COMMAND nvcc
    ARGS -DEXECUTOR_RT -O3 -arch sm_30 -c ${CMAKE_CURRENT_SOURCE_DIR}/backendrendererSetup.cu)



add_executable(backendrenderer backendrenderer.cpp QueryRenderManager.cpp QueryFramebuffer.cpp QueryRenderer.cpp QueryRendererObjects.cpp Shader.cpp BufferLayout.cpp TypeGL.cpp ShaderUtils.cpp DataTable.cpp Color.cpp MapDGL.cpp ${CMAKE_CURRENT_BINARY_DIR}/backendrendererSetup.o)
# add_executable(backendrenderer backendrenderer.cpp QueryRenderManager.cpp QueryFramebuffer.cpp QueryRenderer.cpp QueryRendererObjects.cpp Shader.cpp BufferLayout.cpp TypeGL.cpp ShaderUtils.cpp DataTable.cpp Color.cpp MapDGL.cpp)
add_library(QueryRenderer QueryRenderManager.cpp QueryFramebuffer.cpp QueryRenderer.cpp QueryRendererObjects.cpp Shader.cpp BufferLayout.cpp TypeGL.cpp ShaderUtils.cpp DataTable.cpp Color.cpp MapDGL.cpp)

target_link_libraries(backendrenderer ${Boost_LIBRARIES} ${OPENGL_LIBRARIES} ${GLEW_LIBRARIES} ${GD_LIBRARIES} ${GLFW3_LIBRARIES} ${CUDA_LIBRARIES} ${Glog_LIBRARIES})
target_link_libraries(QueryRenderer ${Boost_LIBRARIES} ${OPENGL_LIBRARIES} ${GLEW_LIBRARIES} ${GD_LIBRARIES} ${GLFW3_LIBRARIES} ${CUDA_LIBRARIES} ${Glog_LIBRARIES})

# target_link_libraries(backendrenderer ${Boost_LIBRARIES} ${OPENGL_LIBRARIES} ${GLEW_LIBRARIES} ${GD_LIBRARIES} ${GLFW3_LIBRARIES} ${Glog_LIBRARIES})
# target_link_libraries(QueryRenderer ${Boost_LIBRARIES} ${OPENGL_LIBRARIES} ${GLEW_LIBRARIES} ${GD_LIBRARIES} ${GLFW3_LIBRARIES} ${Glog_LIBRARIES})

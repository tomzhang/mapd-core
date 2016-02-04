find_package(Boost COMPONENTS filesystem system regex REQUIRED)
find_package(OpenGL REQUIRED)
find_package(PNG REQUIRED)
find_package(GLFW3 REQUIRED)
find_package(Glog REQUIRED)
find_package(CUDA REQUIRED)

include_directories(${Boost_INCLUDE_DIR})
include_directories(${GL_INCLUDE_DIR})
include_directories(${GLEW_INCLUDE_DIR})
include_directories(${PNG_INCLUDE_DIR})
include_directories(${GLFW3_INCLUDE_DIR})
include_directories(${CUDA_INCLUDE_DIR})
include_directories(${Glog_INCLUDE_DIR})
include_directories(ThirdParty/rapidjson)


add_library(QueryRenderer QueryRenderManager.cpp QueryFramebuffer.cpp QueryRenderer.cpp QueryRendererObjects.cpp QueryDataTable.cpp RapidJSONUtils.cpp QueryResultVertexBuffer.cpp QueryDataLayout.cpp)

target_link_libraries(QueryRenderer ${Boost_LIBRARIES} ${OPENGL_LIBRARIES} ${PNG_LIBRARIES} ${GLFW3_LIBRARIES} ${CUDA_LIBRARIES} ${Glog_LIBRARIES})
target_compile_definitions(QueryRenderer PRIVATE ${RENDERING_DEFS})

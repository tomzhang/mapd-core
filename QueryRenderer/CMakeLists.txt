find_package(Boost 1.57 COMPONENTS filesystem system regex REQUIRED)
find_package(OpenGL REQUIRED)
find_package(PNG REQUIRED)
find_package(Glog REQUIRED)
find_package(CUDA REQUIRED)

set(QUERY_RENDERER_SOURCES QueryRenderManager.cpp QueryFramebuffer.cpp QueryRenderer.cpp QueryRendererObjects.cpp QueryDataTable.cpp RapidJSONUtils.cpp QueryResultVertexBuffer.cpp QueryDataLayout.cpp QueryRenderCompositor.cpp PngData.cpp)

set(QUERY_RENDERER_LIBS ${Boost_LIBRARIES} ${OPENGL_LIBRARIES} ${PNG_LIBRARIES} ${CUDA_LIBRARIES} ${Glog_LIBRARIES})

set(QUERY_RENDERER_INCLUDE_DIRS ${Boost_INCLUDE_DIR} ${GL_INCLUDE_DIR} ${PNG_INCLUDE_DIR} ${CUDA_INCLUDE_DIR} ${Glog_INCLUDE_DIR} ThirdParty/rapidjson)

if (_MAPDGL_X11)
  list(APPEND QUERY_RENDERER_INCLUDE_DIRS "${X11_X11_INCLUDE_PATH}")
  list(APPEND QUERY_RENDERER_LIBS "${X11_X11_LIB}")
  list(APPEND QUERY_RENDERER_SOURCES glx/GlxQueryRenderCompositorImpl.cpp)
elseif (_MAPDGL_EGL)
  list(APPEND QUERY_RENDERER_INCLUDE_DIRS "${EGL_INCLUDE_DIR}")
  list(APPEND QUERY_RENDERER_LIBS "${EGL_LIBRARY}")
  list(APPEND QUERY_RENDERER_SOURCES egl/EglQueryRenderCompositorImpl.cpp)
endif()


add_library(QueryRenderer ${QUERY_RENDERER_SOURCES})
target_include_directories(QueryRenderer PRIVATE ${QUERY_RENDERER_INCLUDE_DIRS} ${RENDERING_INCLUDE_DIRS})
target_link_libraries(QueryRenderer ${QUERY_RENDERER_LIBS})
target_compile_definitions(QueryRenderer PRIVATE ${RENDERING_DEFS})

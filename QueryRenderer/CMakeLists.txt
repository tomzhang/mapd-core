find_package(Boost 1.57 COMPONENTS filesystem system regex REQUIRED)
find_package(OpenGL REQUIRED)
find_package(PNG REQUIRED)
find_package(Glog REQUIRED)
find_package(CUDA REQUIRED)

set(QUERY_RENDERER_LIBS ${Boost_LIBRARIES} ${OPENGL_LIBRARIES} ${PNG_LIBRARIES} ${CUDA_LIBRARIES} ${Glog_LIBRARIES} CudaMgr poly2tri)

set(QUERY_RENDERER_INCLUDE_DIRS ${Boost_INCLUDE_DIR} ${GL_INCLUDE_DIR} ${PNG_INCLUDE_DIR} ${CUDA_INCLUDE_DIR} ${Glog_INCLUDE_DIR} ThirdParty/rapidjson)

set(QUERY_RENDERER_SOURCES PngData.cpp QueryDataLayout.cpp QueryRenderManager.cpp QueryRenderer.cpp QueryRendererContext.cpp Types.cpp PerGpuData.cpp)

list(APPEND QUERY_RENDERER_SOURCES Data/BaseQueryDataTable.cpp Data/QueryDataTable.cpp Data/QueryPolyDataTable.cpp Data/Utils.cpp Data/Types.cpp)

list(APPEND QUERY_RENDERER_SOURCES Interop/QueryBuffer.cpp)

list(APPEND QUERY_RENDERER_SOURCES Marks/BaseMark.cpp Marks/PointMark.cpp Marks/PolyMark.cpp Marks/RenderProperty.cpp Marks/Utils.cpp Marks/Enums.cpp)

list(APPEND QUERY_RENDERER_SOURCES Rendering/QueryFramebuffer.cpp Rendering/QueryIdMapPboPool.cpp Rendering/QueryIdMapPixelBuffer.cpp Rendering/QueryRenderCompositor.cpp)

if (_MAPDGL_X11)
  list(APPEND QUERY_RENDERER_INCLUDE_DIRS "${X11_X11_INCLUDE_PATH}")
  list(APPEND QUERY_RENDERER_LIBS "${X11_X11_LIB}")
  list(APPEND QUERY_RENDERER_SOURCES Rendering/glx/GlxQueryRenderCompositorImpl.cpp)
elseif (_MAPDGL_EGL)
  list(APPEND QUERY_RENDERER_INCLUDE_DIRS "${EGL_INCLUDE_DIR}")
  list(APPEND QUERY_RENDERER_LIBS "${EGL_LIBRARY}")
  list(APPEND QUERY_RENDERER_SOURCES Rendering/egl/EglQueryRenderCompositorImpl.cpp)
endif()

list(APPEND QUERY_RENDERER_SOURCES Scales/Scale.cpp Scales/ScaleDomainRangeData.cpp Scales/ScaleRef.cpp Scales/Types.cpp Scales/Utils.cpp)

list(APPEND QUERY_RENDERER_SOURCES Utils/RapidJSONUtils.cpp Utils/Utils.cpp Utils/ShaderUtils.cpp)

add_library(QueryRenderer ${QUERY_RENDERER_SOURCES})
target_include_directories(QueryRenderer PRIVATE ${QUERY_RENDERER_INCLUDE_DIRS} ${RENDERING_INCLUDE_DIRS})
target_link_libraries(QueryRenderer ${QUERY_RENDERER_LIBS})
target_compile_definitions(QueryRenderer PRIVATE ${RENDERING_DEFS})

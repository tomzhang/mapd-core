# option(ENABLE_MULTIRENDERCONTEXT "Enable multrendercontext" ON)
# option(ENABLE_MULTITHREADING "Enable multithreading" ON)

set (RENDERER_TYPE "OpenGL" CACHE STRING "Renderer type")
set_property(CACHE RENDERER_TYPE PROPERTY STRINGS "OpenGL")


set(RENDERING_SOURCES WindowManager.cpp Window.cpp Renderer.cpp Settings/BaseSettings.cpp Settings/IntSettings.cpp Settings/RendererSettings.cpp Settings/StrSettings.cpp Settings/WindowSettings.cpp)
list(APPEND RENDERING_SOURCES Objects/ColorRGBA.cpp)
set(RENDERING_DEFS "" PARENT_SCOPE)
set(RENDERING_INCLUDE_DIRS "")
set(RENDERING_LIBS "")

find_package(Boost COMPONENTS filesystem system regex REQUIRED)
list(APPEND RENDERING_INCLUDE_DIRS "${Boost_INCLUDE_DIR}")
list(APPEND RENDERING_LIBS "${Boost_LIBRARIES}")

find_package(GD REQUIRED)
list(APPEND RENDERING_INCLUDE_DIRS "${GD_INCLUDE_DIR}")
list(APPEND RENDERING_LIBS "${GD_LIBRARIES}")

  find_package(Glog REQUIRED)
list(APPEND RENDERING_INCLUDE_DIRS "${Glog_INCLUDE_DIR}")
list(APPEND RENDERING_LIBS "${Glog_LIBRARIES}")

if (${RENDERER_TYPE} STREQUAL "OpenGL")
  find_package(OpenGL REQUIRED)

  list(APPEND RENDERING_SOURCES Renderer/GL/MapDGL.cpp Renderer/GL/GLWindow.cpp Renderer/GL/GLResourceManager.cpp Renderer/GL/GLUtility.cpp Renderer/GL/GLRenderer.cpp Renderer/GL/TypeGL.cpp)
  list(APPEND RENDERING_SOURCES Renderer/GL/State/GLBindState.cpp)
  list(APPEND RENDERING_SOURCES Renderer/GL/Resources/Enums.cpp Renderer/GL/Resources/GLBaseBuffer.cpp Renderer/GL/Resources/GLBufferLayout.cpp Renderer/GL/Resources/GLFramebuffer.cpp Renderer/GL/Resources/GLPixelBuffer2d.cpp Renderer/GL/Resources/GLRenderbuffer.cpp Renderer/GL/Resources/GLShader.cpp Renderer/GL/Resources/GLTexture2d.cpp Renderer/GL/Resources/GLTexture2dArray.cpp Renderer/GL/Resources/GLVertexArray.cpp Renderer/GL/Resources/GLVertexBuffer.cpp)

  list(APPEND RENDERING_INCLUDE_DIRS "${GL_INCLUDE_DIR}")
  list(APPEND RENDERING_LIBS "${OPENGL_LIBRARIES}")
  list(APPEND RENDERING_DEFS "-DMAPDGL")

  find_package(GLEW_MX REQUIRED)
  list(APPEND RENDERING_INCLUDE_DIRS "${GLEW_MX_INCLUDE_DIR}")
  list(APPEND RENDERING_LIBS "${GLEW_MX_LIBRARIES}")
  list(APPEND RENDERING_DEFS "${GLEW_MX_DEFINITIONS}")

  # if (ENABLE_MULTIRENDERCONTEXT)
  #   find_package(GLEW_MX REQUIRED)
  #   list(APPEND RENDERING_INCLUDE_DIRS "${GLEW_MX_INCLUDE_DIR}")
  #   list(APPEND RENDERING_LIBS "${GLEW_MX_LIBRARIES}")
  #   list(APPEND RENDERING_DEFS "${GLEW_MX_DEFINITIONS}")
  # else ()
  #   find_package(GLEW REQUIRED)
  #   list(APPEND RENDERING_INCLUDE_DIRS "${GLEW_INCLUDE_DIR}")
  #   list(APPEND RENDERING_LIBS "${GLEW_LIBRARIES}")
  # endif()

  if (UNIX)
    set (RENDERER_CONTEXT_TYPE "GLX" CACHE STRING "Renderer context type")
    set_property(CACHE RENDERER_CONTEXT_TYPE PROPERTY STRINGS "GLX" "EGL")

    message(STATUS "Render context type: " ${RENDERER_CONTEXT_TYPE})

    if (${RENDERER_CONTEXT_TYPE} STREQUAL "GLX")
      set(_MAPDGL_X11 1)
      message(STATUS "Using X11 for window creation")
      message(STATUS "Using GLX for opengl context creation")
    elseif (${RENDERER_CONTEXT_TYPE} STREQUAL "EGL")
      set (_MAPDGL_EGL 1)
      message(STATUS "Using EGL for window creation")
      message(STATUS "Using EGL for opengl context creation")
    endif()

  endif()

  if (_MAPDGL_X11)
    find_package(X11 REQUIRED)

    list(APPEND RENDERING_INCLUDE_DIRS "${X11_X11_INCLUDE_PATH}")
    list(APPEND RENDERING_LIBS "${X11_X11_LIB}")
    list(APPEND RENDERING_SOURCES Renderer/GL/glx/X11DisplayManager.cpp Renderer/GL/glx/X11WindowManager.cpp Renderer/GL/glx/GlxGLWindow.cpp Renderer/GL/glx/GlxGLRenderer.cpp Renderer/GL/glx/GlxUtils.cpp)
    list(APPEND RENDERING_DEFS "-DMAPDGL_GLX")
  elseif (_MAPDGL_EGL)
    find_package(EGL REQUIRED)
    list(APPEND RENDERING_INCLUDE_DIRS "${EGL_INCLUDE_DIR}")
    list(APPEND RENDERING_LIBS "${EGL_LIBRARY}")
    list(APPEND RENDERING_SOURCES Renderer/GL/egl/MapDEGL.cpp Renderer/GL/egl/EglUtils.cpp Renderer/GL/egl/EglWindowManager.cpp Renderer/GL/egl/EglDisplayManager.cpp Renderer/GL/egl/EglGLWindow.cpp Renderer/GL/egl/EglGLRenderer.cpp)
    list(APPEND RENDERING_DEFS "-DMAPDGL_EGL")
  endif()


endif()

add_library(Rendering ${RENDERING_SOURCES})
target_include_directories(Rendering PRIVATE ${RENDERING_INCLUDE_DIRS})
target_link_libraries(Rendering ${RENDERING_LIBS})
target_compile_definitions(Rendering PRIVATE ${RENDERING_DEFS})

# expose the definitions to the parent
set(RENDERING_INCLUDE_DIRS ${RENDERING_INCLUDE_DIRS} PARENT_SCOPE)
set(RENDERING_DEFS ${RENDERING_DEFS} PARENT_SCOPE)
set(_MAPDGL_EGL ${_MAPDGL_EGL} PARENT_SCOPE)
set(_MAPDGL_X11 ${_MAPDGL_X11} PARENT_SCOPE)


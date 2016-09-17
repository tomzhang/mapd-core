#ifndef RENDERING_GL_TYPES_H_
#define RENDERING_GL_TYPES_H_

#include <memory>
#include "../../Types.h"

namespace Rendering {

namespace GL {

class GLRenderer;
typedef std::unique_ptr<GLRenderer> GLRendererUqPtr;
typedef std::shared_ptr<GLRenderer> GLRendererShPtr;
typedef std::weak_ptr<GLRenderer> GLRendererWkPtr;

class GLResourceManager;
typedef std::weak_ptr<GLResourceManager> GLResourceManagerWkPtr;
typedef std::shared_ptr<GLResourceManager> GLResourceManagerShPtr;

class GLWindow;
typedef std::shared_ptr<GLWindow> GLWindowShPtr;
typedef std::unique_ptr<GLWindow> GLWindowUqPtr;

struct BaseTypeGL;
typedef std::shared_ptr<BaseTypeGL> TypeGLShPtr;
typedef std::unique_ptr<BaseTypeGL> TypeGLUqPtr;

typedef uint32_t ResourceId;
typedef std::pair<RendererId, ResourceId> UniqueResourceId;

std::string to_string(const UniqueResourceId& id);

}  // namespace GL

}  // namespace Rendering

#endif  // RENDERING_GL_TYPES_H_

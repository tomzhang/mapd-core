#include "GLShaderBlockLayout.h"

namespace Rendering {
namespace GL {
namespace Resources {

int GLShaderBlockLayout::uniformBufferOffsetAlignment = -1;
int GLShaderBlockLayout::maxUniformBlockSize = -1;

// all shader block layouts must be interleaved, but there are rules for
// the stride/offset for each layout type, hence the reasoning for
// a specific shader block layout.
GLShaderBlockLayout::GLShaderBlockLayout(ShaderBlockLayoutType layoutType)
    : GLBaseBufferLayout(GLBufferLayoutType::INTERLEAVED), _layoutType(layoutType), _addingAttrs(false) {
  _itemByteSize = 0;
}

GLShaderBlockLayout::GLShaderBlockLayout(ShaderBlockLayoutType layoutType,
                                         const GLShaderShPtr& shaderPtr,
                                         size_t blockByteSize)
    : GLBaseBufferLayout(GLBufferLayoutType::INTERLEAVED),
      _shaderPtr(shaderPtr),
      _layoutType(layoutType),
      _addingAttrs(false) {
  _initNumAlignmentBytes();
  _initMaxUniformBlockSize();

  RUNTIME_EX_ASSERT(shaderPtr != nullptr && blockByteSize > 0,
                    "Cannot create a GLShaderBlockLayout without a shader or without a block size.");

  RUNTIME_EX_ASSERT(static_cast<int>(blockByteSize) <= maxUniformBlockSize,
                    "Block size of " + std::to_string(blockByteSize) + " bytes exceeds the maximum size of " +
                        std::to_string(maxUniformBlockSize) + " bytes.");

  if (layoutType == ShaderBlockLayoutType::PACKED || layoutType == ShaderBlockLayoutType::SHARED) {
    _itemByteSize = blockByteSize;
  }
}

GLShaderBlockLayout::~GLShaderBlockLayout() {
}

bool GLShaderBlockLayout::operator==(const GLShaderBlockLayout& layout) const {
  bool checkAttrs = (getLayoutType() == layout.getLayoutType() && getNumBytesInBlock() == layout.getNumBytesInBlock() &&
                     numAttributes() == layout.numAttributes());

  if (checkAttrs) {
    for (int i = 0; i < numAttributes(); ++i) {
      if (operator[](i) != layout[i]) {
        return false;
      }
    }
    return true;
  }
  return false;
}

bool GLShaderBlockLayout::operator!=(const GLShaderBlockLayout& layout) const {
  return !operator==(layout);
}

void GLShaderBlockLayout::beginAddingAttrs() {
  // TODO(croot): should we warn the user that attrs will be cleared (if there are any)
  if (_addingAttrs) {
    return;
  }

  _attrMap.clear();
  _addingAttrs = true;
}

void GLShaderBlockLayout::endAddingAttrs() {
  if (_layoutType == ShaderBlockLayoutType::STD140 || _layoutType == ShaderBlockLayoutType::STD430) {
    static const size_t byteAlignment = 4 * sizeof(float);
    size_t offset;
    if ((offset = _itemByteSize % byteAlignment) != 0) {
      // TODO(croot) - this shouldn't happen here. The re-alignment
      // should be done per attr and needs to match the offset that
      // would result from a glGetActiveUniformsiv(programId, 1, &attrIdx, GL_UNIFORM_OFFSET, &attrOffset)
      // call. Would adding dummy attrs in such a case make sense? If so, here's some code
      // that would do that.
      //
      // auto bytesLeft = byteAlignment - offset;
      // RUNTIME_EX_ASSERT(bytesLeft % 4 == 0,
      //                   "Cannot re-align GLShaderBlockLayout. There are currently " + std::to_string(_itemByteSize) +
      //                       " bytes in the layout and " + std::to_string(bytesLeft) +
      //                       " bytes are needed to re-align, but the bytes left need to be a multiple of 4.");
      // static const std::string dummyPrefix = "_blocklayout_dummy";
      // for (size_t i = 0; i < bytesLeft / 4; ++i) {
      //   this->addAttribute<float>(dummyPrefix + std::to_string(i));
      // }

      _itemByteSize += byteAlignment - offset;
    }
  }

  _addingAttrs = false;
}

std::string GLShaderBlockLayout::buildShaderBlockCode(const std::string& blockName,
                                                      const std::string& instanceName,
                                                      StorageQualifier storageQualifier) {
  CHECK(!_addingAttrs)
      << "Please call the endAddingAttrs() method before calling methods requiring the all attributes to be added.";

  std::string rtn;

  // TODO(croot) what about other layout/binding/location rules?
  switch (_layoutType) {
    case ShaderBlockLayoutType::PACKED:
      rtn += "layout(packed) ";
      break;
    case ShaderBlockLayoutType::SHARED:
      rtn += "layout(shared) ";
      break;
    case ShaderBlockLayoutType::STD140:
      rtn += "layout(std140) ";
      break;
    case ShaderBlockLayoutType::STD430:
      rtn += "layout(std430) ";
      break;
  }

  std::string qualifier = to_string(storageQualifier);
  std::transform(qualifier.begin(), qualifier.end(), qualifier.begin(), ::tolower);
  rtn += qualifier + " ";

  rtn += blockName + " {\n";

  for (auto& attrInfoPtr : _attrMap) {
    rtn += "  " + attrInfoPtr->typeInfo->glslType() + " " + attrInfoPtr->name + ";\n";
  }

  rtn += "}";
  if (instanceName.length()) {
    rtn += " " + instanceName + ";\n";
  } else {
    rtn += ";\n";
  }

  return rtn;
}

void GLShaderBlockLayout::bindToShader(GLShader* activeShader,
                                       const size_t usedBytes,
                                       const size_t offsetBytes,
                                       const std::string& attr,
                                       const std::string& shaderAttr) const {
  THROW_RUNTIME_EX(
      "A GLShaderBlockLayout cannot be bound to a shader. It is instead defined by a shader via a shader storage "
      "block.")
}

void GLShaderBlockLayout::_initNumAlignmentBytes() {
  if (uniformBufferOffsetAlignment < 0) {
    MAPD_CHECK_GL_ERROR(glGetIntegerv(GL_UNIFORM_BUFFER_OFFSET_ALIGNMENT, &uniformBufferOffsetAlignment));
    CHECK(uniformBufferOffsetAlignment > 0);
  }
}

void GLShaderBlockLayout::_initMaxUniformBlockSize() {
  if (maxUniformBlockSize < 0) {
    MAPD_CHECK_GL_ERROR(glGetIntegerv(GL_MAX_UNIFORM_BLOCK_SIZE, &maxUniformBlockSize));
  }
}

size_t GLShaderBlockLayout::getNumAlignmentBytes() {
  // TODO(croot): if this is called outside of this class, should I throw
  // a runtime error if the alignment bytes is un-initialized instead? This
  // would force that a GLShaderBlockLayout be defined first, which means
  // a context would be defined.
  _initNumAlignmentBytes();
  return uniformBufferOffsetAlignment;
}

}  // namespace Resources
}  // namespace GL
}  // namespace Rendering

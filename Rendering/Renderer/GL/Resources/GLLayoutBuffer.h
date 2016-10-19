#ifndef RENDERING_GL_RESOURCES_GLLAYOUTBUFFER_H_
#define RENDERING_GL_RESOURCES_GLLAYOUTBUFFER_H_

#include "Types.h"
#include "GLBufferLayout.h"
#include "GLBaseBuffer.h"

#include <boost/multi_index_container.hpp>
#include <boost/multi_index/ordered_index.hpp>
#include <boost/multi_index/hashed_index.hpp>
#include <boost/multi_index/mem_fun.hpp>
#include <boost/lambda/lambda.hpp>

namespace Rendering {
namespace GL {
namespace Resources {

class GLLayoutManagerBuffer : public GLBaseBuffer {
 public:
  size_t getNumUsedBytes(const GLBufferLayoutShPtr& layoutPtr = nullptr) const;
  size_t numItems(const GLBufferLayoutShPtr& layoutPtr = nullptr) const;

  bool hasAttribute(const std::string& attrName, const GLBufferLayoutShPtr& layoutPtr = nullptr) const;
  TypeGLShPtr getAttributeTypeGL(const std::string& attrName, const GLBufferLayoutShPtr& layoutPtr = nullptr) const;
  GLBufferAttrType getAttributeType(const std::string& attrName, const GLBufferLayoutShPtr& layoutPtr = nullptr) const;

  int getNumBufferLayouts() const { return _layoutMap.size(); }
  bool hasBufferLayout(const GLBufferLayoutShPtr& layoutPtr) const;
  bool hasBufferLayoutAtOffset(const size_t offsetBytes) const;
  void addBufferLayout(const GLBufferLayoutShPtr& layoutPtr, const size_t numBytes, const size_t offsetBytes = 0);
  void replaceBufferLayoutAtOffset(const GLBufferLayoutShPtr& newLayoutPtr,
                                   const size_t numBytes,
                                   const size_t offsetBytes);
  void deleteBufferLayoutAtOffset(const size_t offsetBytes);
  void deleteBufferLayout(const GLBufferLayoutShPtr& layoutPtr);
  void deleteAllBufferLayouts();

  const GLBufferLayoutShPtr getBufferLayoutAtIndex(const size_t idx) const;
  const GLBufferLayoutShPtr getBufferLayoutAtOffset(const size_t offsetBytes) const;
  std::pair<size_t, size_t> getBufferLayoutData(const GLBufferLayoutShPtr& layoutPtr) const;
  std::pair<size_t, size_t> getBufferLayoutDataAtIndex(const size_t idx) const;

 protected:
  explicit GLLayoutManagerBuffer(const RendererWkPtr& rendererPtr,
                                 GLResourceType rsrcType,
                                 GLBufferType type,
                                 GLenum target = GL_ARRAY_BUFFER,
                                 BufferAccessType accessType = BufferAccessType::READ_AND_WRITE,
                                 BufferAccessFreq accessFreq = BufferAccessFreq::STATIC)
      : GLBaseBuffer(rendererPtr, rsrcType, type, target, accessType, accessFreq) {}

  virtual void _makeEmpty() override { deleteAllBufferLayouts(); }

  struct BoundLayoutData {
    GLBufferLayoutShPtr layoutPtr;
    size_t usedBytes;
    size_t offsetBytes;

    BoundLayoutData() : layoutPtr(nullptr), usedBytes(0), offsetBytes(0) {}
    BoundLayoutData(const GLBufferLayoutShPtr& layout, const size_t usedBytes, const size_t offsetBytes)
        : layoutPtr(layout), usedBytes(usedBytes), offsetBytes(offsetBytes) {}
  };

  const GLBaseBufferLayout* _getBufferLayoutToUse(const GLBufferLayoutShPtr& layoutPtr,
                                                  const std::string& errPrefix = "") const;

  const BoundLayoutData& _getBufferLayoutDataToUse(const GLBufferLayoutShPtr& layoutPtr,
                                                   const std::string& errPrefix = "") const;

  const BoundLayoutData& _getBufferLayoutDataAtIndexToUse(const size_t idx) const;

  virtual void _validateBufferLayout(size_t numBytes,
                                     size_t offsetBytes,
                                     const GLBufferLayoutShPtr& layoutPtr,
                                     bool replaceExistingLayout = false,
                                     const std::string& errPrefix = "");
  void _updateBufferLayouts(const GLBufferLayoutShPtr& layoutPtr,
                            const size_t numBytes,
                            const size_t offsetBytes,
                            bool replaceExistingLayout = false);
  void _deleteBufferLayoutsExcept(const GLBufferLayoutShPtr& layout);
  void _deleteBufferLayout(const GLBufferLayoutShPtr& layout);

  //   // void _addVertexArray(GLVertexArrayShPtr& vao);
  //   // void _deleteVertexArray(GLVertexArray* vao);
  //   // void _deleteAllVertexArrays(const GLBaseBufferLayout* layout = nullptr);
  //   // void _updateVertexArrays(const GLBufferLayoutShPtr& layoutPtr);

 private:
  virtual void _layoutAddedCB(const BoundLayoutData& layoutData) {}
  virtual void _layoutUpdatedCB(const BoundLayoutData& preUpdateLayoutData,
                                const BoundLayoutData& postUpdateLayoutData) {}
  virtual void _layoutReplacedCB(const BoundLayoutData& replacedLayoutData,
                                 const BoundLayoutData& replacementLayoutData) {}
  virtual void _layoutDeletedCB(const BoundLayoutData& layoutData) {}
  virtual void _allLayoutsDeletedCB() {}

  struct BufferLayoutOffsetTag {};
  struct BufferLayoutTag {};
  typedef boost::multi_index_container<
      BoundLayoutData,
      boost::multi_index::indexed_by<
          boost::multi_index::ordered_unique<
              boost::multi_index::tag<BufferLayoutOffsetTag>,
              boost::multi_index::member<BoundLayoutData, size_t, &BoundLayoutData::offsetBytes>>,
          boost::multi_index::hashed_unique<
              boost::multi_index::tag<BufferLayoutTag>,
              boost::multi_index::member<BoundLayoutData, GLBufferLayoutShPtr, &BoundLayoutData::layoutPtr>>>>
      BufferLayoutMap;

  typedef BufferLayoutMap::index<BufferLayoutOffsetTag>::type BufferLayoutMapByOffset;
  typedef BufferLayoutMap::index<BufferLayoutTag>::type BufferLayoutMapByPtr;

  BufferLayoutMap _layoutMap;
};

template <typename GLLayoutType = GLBaseBufferLayout>
class GLLayoutBuffer : public GLLayoutManagerBuffer {
 public:
  virtual ~GLLayoutBuffer() {}

  const std::shared_ptr<GLLayoutType> getBufferLayoutAtIndex(const size_t idx) const {
    return _getLayoutSharedPointer(GLLayoutManagerBuffer::getBufferLayoutAtIndex(idx));
  }

  virtual void bufferData(void* data, const size_t numBytes, const std::shared_ptr<GLLayoutType>& layoutPtr = nullptr) {
    // TODO(croot): need to come up with a better error
    // resolution scheme here -- i.e. should we restore
    // the state of the layouts and memory on an error?
    // Or mark the vbo as undefined? Need to come up
    // with an appropriate scheme

    // Now do the actual buffering
    GLBaseBuffer::bufferData(data, numBytes);

    _deleteBufferLayoutsExcept(layoutPtr != nullptr ? layoutPtr : getBufferLayoutAtOffset(0));

    if (layoutPtr) {
      // add the layout, or update it if it is alread
      // bound to this vbo
      _updateBufferLayouts(layoutPtr, numBytes, 0);
    }
  }

  virtual void bufferSubData(void* data,
                             const size_t numBytes,
                             const size_t offsetBytes,
                             const std::shared_ptr<GLLayoutType>& layoutPtr = nullptr) {
    // TODO(croot): need to come up with a better error
    // resolution scheme here -- i.e. should we restore
    // the state of the layouts and memory on an error?
    // Or mark the vbo as undefined? Need to come up
    // with an appropriate scheme
    if (layoutPtr) {
      _validateBufferLayout(numBytes, offsetBytes, layoutPtr);
    }

    // now do the buffering
    GLBaseBuffer::bufferSubData(data, numBytes, offsetBytes);

    _updateBufferLayouts(layoutPtr, numBytes, offsetBytes);
  }

 protected:
  explicit GLLayoutBuffer(const RendererWkPtr& rendererPtr,
                          GLResourceType rsrcType,
                          GLBufferType type,
                          GLenum target = GL_ARRAY_BUFFER,
                          BufferAccessType accessType = BufferAccessType::READ_AND_WRITE,
                          BufferAccessFreq accessFreq = BufferAccessFreq::STATIC)
      : GLLayoutManagerBuffer(rendererPtr, rsrcType, type, target, accessType, accessFreq) {}

  const GLLayoutType* _getBufferLayoutToUse(const std::shared_ptr<GLLayoutType>& layoutPtr,
                                            const std::string& errPrefix = "") const {
    return _getLayoutPointer(GLLayoutManagerBuffer::_getBufferLayoutToUse(layoutPtr, errPrefix));
  }

 private:
  const std::shared_ptr<GLLayoutType> _getLayoutSharedPointer(const GLBufferLayoutShPtr& layout) const {
    auto rtn = std::dynamic_pointer_cast<GLLayoutType>(layout);
    return const_cast<const std::shared_ptr<GLLayoutType>&>(rtn);
  }

  std::shared_ptr<GLLayoutType> _getLayoutSharedPointer(GLBufferLayoutShPtr& layout) {
    return std::dynamic_pointer_cast<GLLayoutType>(layout);
  }

  const GLLayoutType* _getLayoutPointer(const GLBaseBufferLayout* layout) const {
    return dynamic_cast<const GLLayoutType*>(layout);
  }

  GLLayoutType* _getLayoutPointer(GLBaseBufferLayout* layout) { return dynamic_cast<GLLayoutType*>(layout); }
};

}  // namespace Resources
}  // namespace GL
}  // namespace Rendering

#endif  // RENDERING_GL_RESOURCES_GLLAYOUTBUFFER_H_

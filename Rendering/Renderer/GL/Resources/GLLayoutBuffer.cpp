#include "GLLayoutBuffer.h"
#include "../GLResourceManager.h"

namespace Rendering {
namespace GL {
namespace Resources {

size_t GLLayoutManagerBuffer::getNumUsedBytes(const GLBufferLayoutShPtr& layoutPtr) const {
  if (!layoutPtr) {
    size_t usedBytes = 0;
    // TODO(croot): cache this and use a dirty flag?
    for (auto& item : _layoutMap) {
      usedBytes += item.usedBytes;
    }
    return usedBytes;
  }

  return _getBufferLayoutDataToUse(layoutPtr, "Cannot get number of used bytes. ").usedBytes;
}

size_t GLLayoutManagerBuffer::numItems(const GLBufferLayoutShPtr& layoutPtr) const {
  auto& layoutData = _getBufferLayoutDataToUse(layoutPtr, "Cannoc calculate number of items. ");
  return layoutData.usedBytes / layoutData.layoutPtr->getNumBytesPerItem();
}

bool GLLayoutManagerBuffer::hasAttribute(const std::string& attrName, const GLBufferLayoutShPtr& layoutPtr) const {
  if (!_layoutMap.size()) {
    return false;
  }

  return _getBufferLayoutToUse(layoutPtr, "Cannot check the existence of attribute " + attrName + ". ")
      ->hasAttribute(attrName);
}

TypeGLShPtr GLLayoutManagerBuffer::getAttributeTypeGL(const std::string& attrName,
                                                      const GLBufferLayoutShPtr& layoutPtr) const {
  return _getBufferLayoutToUse(layoutPtr, "Cannot get GL attribute type for " + attrName + ". ")
      ->getAttributeTypeGL(attrName);
}

GLBufferAttrType GLLayoutManagerBuffer::getAttributeType(const std::string& attrName,
                                                         const GLBufferLayoutShPtr& layoutPtr) const {
  return _getBufferLayoutToUse(layoutPtr, "Cannot get attribute type for " + attrName + ". ")
      ->getAttributeType(attrName);
}

bool GLLayoutManagerBuffer::hasBufferLayout(const GLBufferLayoutShPtr& layoutPtr) const {
  if (!layoutPtr) {
    return false;
  }

  const auto& ptrLookup = _layoutMap.get<BufferLayoutTag>();
  return ptrLookup.find(layoutPtr) != ptrLookup.end();
}

bool GLLayoutManagerBuffer::hasBufferLayoutAtOffset(const size_t offsetBytes) const {
  return _layoutMap.find(offsetBytes) != _layoutMap.end();
}

void GLLayoutManagerBuffer::addBufferLayout(const GLBufferLayoutShPtr& layoutPtr,
                                            const size_t numBytes,
                                            const size_t offsetBytes) {
  RUNTIME_EX_ASSERT(layoutPtr, "Cannot add a null layout to the buffer.");

  _validateBufferLayout(numBytes, offsetBytes, layoutPtr, false);

  _updateBufferLayouts(layoutPtr, numBytes, offsetBytes, false);
}

void GLLayoutManagerBuffer::replaceBufferLayoutAtOffset(const GLBufferLayoutShPtr& newLayoutPtr,
                                                        const size_t numBytes,
                                                        const size_t offsetBytes) {
  RUNTIME_EX_ASSERT(newLayoutPtr, "Cannot replace an existing layout with a null");

  _validateBufferLayout(numBytes, offsetBytes, newLayoutPtr, true);

  _updateBufferLayouts(newLayoutPtr, numBytes, offsetBytes, true);
}

void GLLayoutManagerBuffer::deleteBufferLayoutAtOffset(const size_t offsetBytes) {
  auto itr = _layoutMap.find(offsetBytes);
  if (itr != _layoutMap.end()) {
    auto deletedData = *itr;
    _layoutMap.erase(itr);
    _layoutDeletedCB(deletedData);
  }
}

void GLLayoutManagerBuffer::deleteBufferLayout(const GLBufferLayoutShPtr& layoutPtr) {
  BufferLayoutMapByPtr& ptrLookup = _layoutMap.get<BufferLayoutTag>();
  auto itr = ptrLookup.find(layoutPtr);
  if (itr != ptrLookup.end()) {
    auto deletedData = *itr;
    ptrLookup.erase(itr);
    _layoutDeletedCB(deletedData);
  }
}

void GLLayoutManagerBuffer::deleteAllBufferLayouts() {
  _layoutMap.clear();
  _allLayoutsDeletedCB();
}

const GLBufferLayoutShPtr GLLayoutManagerBuffer::getBufferLayoutAtIndex(const size_t i) const {
  auto data = _getBufferLayoutDataAtIndexToUse(i);
  return data.layoutPtr;
}

const GLBufferLayoutShPtr GLLayoutManagerBuffer::getBufferLayoutAtOffset(const size_t offsetBytes) const {
  auto itr = _layoutMap.find(offsetBytes);
  return (itr != _layoutMap.end() ? itr->layoutPtr : nullptr);
}

std::pair<size_t, size_t> GLLayoutManagerBuffer::getBufferLayoutData(const GLBufferLayoutShPtr& layoutPtr) const {
  auto data = _getBufferLayoutDataToUse(layoutPtr, "Cannot get buffer layout data. ");
  return std::make_pair(data.usedBytes, data.offsetBytes);
}

std::pair<size_t, size_t> GLLayoutManagerBuffer::getBufferLayoutDataAtIndex(const size_t idx) const {
  auto data = _getBufferLayoutDataAtIndexToUse(idx);
  return std::make_pair(data.usedBytes, data.offsetBytes);
}

const GLBaseBufferLayout* GLLayoutManagerBuffer::_getBufferLayoutToUse(const GLBufferLayoutShPtr& layoutPtr,
                                                                       const std::string& errPrefix) const {
  RUNTIME_EX_ASSERT(
      layoutPtr != nullptr || _layoutMap.size() == 1,
      errPrefix + "Buffer layout validation failed. There isn't a layout supplied as an argument and the vbo has " +
          std::to_string(_layoutMap.size()) + " attached layouts. Cannot determine which layout to use.");

  const auto& ptrLookup = _layoutMap.get<BufferLayoutTag>();
  RUNTIME_EX_ASSERT(!layoutPtr || ptrLookup.find(layoutPtr) != ptrLookup.end(),
                    errPrefix + "The layout supplied as an argument is not a currently attached layout to the vbo.");

  return (layoutPtr ? layoutPtr.get() : (*_layoutMap.begin()).layoutPtr.get());
}

const GLLayoutManagerBuffer::BoundLayoutData& GLLayoutManagerBuffer::_getBufferLayoutDataToUse(
    const GLBufferLayoutShPtr& layoutPtr,
    const std::string& errPrefix) const {
  RUNTIME_EX_ASSERT(
      layoutPtr != nullptr || _layoutMap.size() == 1,
      errPrefix + "Buffer layout validation failed. There isn't a layout supplied as an argument and the buffer has " +
          std::to_string(_layoutMap.size()) + " attached layouts. Cannot determine which layout to use.");

  const auto& ptrLookup = _layoutMap.get<BufferLayoutTag>();

  auto itr = (layoutPtr ? ptrLookup.find(layoutPtr) : ptrLookup.begin());

  RUNTIME_EX_ASSERT(itr != ptrLookup.end(),
                    errPrefix + "The layout supplied as an argument is not a currently attached layout to the buffer.");

  return *itr;
}

const GLLayoutManagerBuffer::BoundLayoutData& GLLayoutManagerBuffer::_getBufferLayoutDataAtIndexToUse(
    const size_t idx) const {
  // TODO(croot): perhaps find a way to add a random access index to the
  // boost::multi_index where the indices align with the ordered index
  //
  // Or use iterators and std::set methods such as begin() and end().
  RUNTIME_EX_ASSERT(idx < _layoutMap.size(),
                    "Index " + std::to_string(idx) + " out of range. There are only " +
                        std::to_string(_layoutMap.size()) + " layouts available.");

  auto itr = _layoutMap.begin();
  size_t cnt = 0;
  while (++cnt <= idx) {
    itr++;
  }

  CHECK(itr != _layoutMap.end());

  return *itr;
}

void GLLayoutManagerBuffer::_validateBufferLayout(size_t numBytes,
                                                  size_t offsetBytes,
                                                  const GLBufferLayoutShPtr& layoutPtr,
                                                  bool replaceExistingLayout,
                                                  const std::string& errPrefix) {
  RUNTIME_EX_ASSERT(offsetBytes + numBytes <= getNumBytes(),
                    "Cannot update buffer with " + std::to_string(numBytes) + " bytes starting at offsetBytes: " +
                        std::to_string(offsetBytes) + " as it would overrun the length (" +
                        std::to_string(getNumBytes()) + " bytes) of the buffer.");

  if (layoutPtr && _layoutMap.size()) {
    GLBufferLayoutShPtr replLayout;
    if (replaceExistingLayout) {
      auto offsetItr = _layoutMap.find(offsetBytes);
      if (offsetItr != _layoutMap.end()) {
        replLayout = offsetItr->layoutPtr;
      }
    }
    auto boundItrs = _layoutMap.range(offsetBytes <= boost::lambda::_1, boost::lambda::_1 < offsetBytes + numBytes);

    auto startItr = boundItrs.first;

    RUNTIME_EX_ASSERT(
        startItr == boundItrs.second ||
            ((startItr->layoutPtr == layoutPtr || (replaceExistingLayout && startItr->layoutPtr == replLayout)) &&
             ++startItr == boundItrs.second),
        "Cannot successfully add layout with byte offset: " + std::to_string(offsetBytes) + " and num bytes: " +
            std::to_string(numBytes) + " as it overlaps with another layout at byte offset: " +
            std::to_string(boundItrs.first->layoutPtr == layoutPtr ? startItr->offsetBytes
                                                                   : boundItrs.first->offsetBytes));

    if (boundItrs.first == _layoutMap.begin()) {
      return;
    }

    boundItrs.first--;
    RUNTIME_EX_ASSERT(boundItrs.first == _layoutMap.end() || boundItrs.first->layoutPtr == layoutPtr ||
                          (replaceExistingLayout && boundItrs.first->layoutPtr == replLayout) ||
                          boundItrs.first->offsetBytes + boundItrs.first->usedBytes <= offsetBytes,
                      "Cannot successfully add layout with byte offset: " + std::to_string(offsetBytes) +
                          " as it overlaps with another layout at byte offset: " +
                          std::to_string(boundItrs.first->offsetBytes) + " and num bytes: " +
                          std::to_string(boundItrs.first->usedBytes));
  }
}

void GLLayoutManagerBuffer::_updateBufferLayouts(const GLBufferLayoutShPtr& layoutPtr,
                                                 const size_t numBytes,
                                                 const size_t offsetBytes,
                                                 bool replaceExistingLayout) {
  BufferLayoutMapByPtr& ptrLookup = _layoutMap.get<BufferLayoutTag>();

  // TODO(croot): might need to come up with a better error
  // resolution scheme here -- i.e. should we restore
  // the state of the layouts and memory on an error?
  // Or mark the vbo as undefined? Need to come up
  // with an appropriate scheme

  auto itr = ptrLookup.find(layoutPtr);
  BufferLayoutMapByOffset::iterator offsetItr;
  bool success = false;
  BoundLayoutData oldData;
  if (itr != ptrLookup.end()) {
    // first project into a offsetBytes iterator
    offsetItr = _layoutMap.project<BufferLayoutOffsetTag>(itr);
    oldData = *offsetItr;
    success = _layoutMap.modify(offsetItr, [numBytes, offsetBytes](BoundLayoutData& layoutData) {
      layoutData.usedBytes = numBytes;
      layoutData.offsetBytes = offsetBytes;
    });

    RUNTIME_EX_ASSERT(
        success,
        "Cannot successfully add layout to the buffer object. Another layout is already bound to the byte offset: " +
            std::to_string(offsetBytes) + ".");

    offsetItr = _layoutMap.find(offsetBytes);
    _layoutUpdatedCB(oldData, *offsetItr);
  } else {
    GLBufferLayoutShPtr replLayout;
    if (replaceExistingLayout) {
      auto replaceItr = _layoutMap.find(offsetBytes);
      if (replaceItr != _layoutMap.end()) {
        replLayout = replaceItr->layoutPtr;
        oldData = *replaceItr;
        success = _layoutMap.replace(replaceItr, BoundLayoutData(layoutPtr, numBytes, offsetBytes));
        offsetItr = replaceItr;
      }
    }

    if (!replLayout) {
      auto insertPair = _layoutMap.emplace(layoutPtr, numBytes, offsetBytes);
      offsetItr = insertPair.first;
      success = insertPair.second;
    }

    RUNTIME_EX_ASSERT(success,
                      "Cannot successfully add layout to vbo. Another layout is already bound to the byte offset: " +
                          std::to_string(offsetBytes) + ".");

    if (replLayout) {
      _layoutReplacedCB(oldData, *offsetItr);
    } else {
      _layoutAddedCB(*offsetItr);
    }
  }

  // now make sure the offsetBytes + useBytes doesn't overlap with a neighboring
  // layout

  // TODO(croot): this might be overkill as the _validateBufferLayout checks for
  // overlap
  auto currItr = offsetItr;
  if (++offsetItr != _layoutMap.end()) {
    if (offsetBytes + numBytes > offsetItr->offsetBytes) {
      _layoutMap.erase(currItr);
      THROW_RUNTIME_EX("Cannot successfully add layout with byte offset: " + std::to_string(offsetBytes) +
                       " and num bytes: " + std::to_string(numBytes) +
                       " as it overlaps with another layout at byte offset: " + std::to_string(offsetItr->offsetBytes));
    }
  }

  offsetItr = currItr;
  if (offsetItr != _layoutMap.begin()) {
    offsetItr--;
    if (offsetItr->offsetBytes + offsetItr->usedBytes > offsetBytes) {
      _layoutMap.erase(currItr);
      THROW_RUNTIME_EX("Cannot successfully add layout with byte offset: " + std::to_string(offsetBytes) +
                       " as it overlaps with another layout at byte offset: " + std::to_string(offsetItr->offsetBytes) +
                       " and num bytes: " + std::to_string(offsetItr->usedBytes));
    }
  }
}

void GLLayoutManagerBuffer::_deleteBufferLayoutsExcept(const GLBufferLayoutShPtr& layout) {
  if (!layout || !hasBufferLayout(layout)) {
    deleteAllBufferLayouts();
  } else if (_layoutMap.size() > 1) {
    // TODO(croot): this is hacky -- may need to revisit
    // Attempting to clear all other layouts except 1.
    // Easiest/fastest way is to clear out all the
    // other layouts and re-add the one in question,
    // but that could have repercusions with the callbacks
    BoundLayoutData ptrToReinsert;
    for (auto& layoutData : _layoutMap) {
      if (layoutData.layoutPtr != layout) {
        // layoutData->clearBuffer();
        _layoutDeletedCB(layoutData);
      } else {
        ptrToReinsert = layoutData;
      }
    }
    _layoutMap.clear();
    _layoutMap.insert(ptrToReinsert);
  }
}

}  // namespace Resources
}  // namespace GL
}  // namespace Rendering

#ifndef BUFFER_LAYOUT_H_
#define BUFFER_LAYOUT_H_

#include <unordered_map>
#include <array>
#include <vector>
#include <string>
#include <assert.h>
#include <utility> // std::pair, std::make_pair
#include <memory>  // std::unique_ptr
#include <GL/glew.h>

namespace MapD_Renderer {

enum class BufferAttrType {
    UINT = 0,

    INT,
    VEC2I,
    VEC3I,
    VEC4I,

    FLOAT,
    VEC2F,
    VEC3F,
    VEC4F,

    DOUBLE,
    VEC2D,
    VEC3D,
    VEC4D
};


BufferAttrType getBufferAttrType(unsigned int a, int numComponents=1);
BufferAttrType getBufferAttrType(int a, int numComponents=1);
BufferAttrType getBufferAttrType(float a, int numComponents=1);
BufferAttrType getBufferAttrType(double a, int numComponents=1);

struct BaseTypeInfo {
    virtual ~BaseTypeInfo() {}

    virtual int numComponents() = 0;
    virtual int numBytes() = 0;
    virtual int glType() = 0;
};

template <typename T, int componentCnt, int typeGL>
struct AttrTypeInfo : BaseTypeInfo {
    AttrTypeInfo() {}

    int numComponents() {
        return componentCnt;
    }

    int numBytes() {
        return sizeof(T)*numComponents();
    }

    int glType() {
        return typeGL;
    }
};

struct BufferAttrInfo {
    BufferAttrType type;
    BaseTypeInfo *typeInfo;
    int stride;
    int offset;

    BufferAttrInfo(BufferAttrType type, BaseTypeInfo *typeInfo, int stride, int offset) : type(type), typeInfo(typeInfo), stride(stride), offset(offset) {}
};



typedef std::unique_ptr<BaseTypeInfo> BaseTypeInfoPtr;
typedef std::unique_ptr<BufferAttrInfo> BufferAttrInfoPtr;
typedef std::unordered_map<std::string, BufferAttrInfoPtr> AttrMap;

static std::array<BaseTypeInfoPtr, 13> attrTypeInfo = {
    {BaseTypeInfoPtr(new AttrTypeInfo<unsigned int, 1, GL_UNSIGNED_INT>()), // UINT

     BaseTypeInfoPtr(new AttrTypeInfo<int, 1, GL_INT>()),          // INT
     BaseTypeInfoPtr(new AttrTypeInfo<int, 2, GL_INT>()),          // VEC2I
     BaseTypeInfoPtr(new AttrTypeInfo<int, 3, GL_INT>()),          // VEC3I
     BaseTypeInfoPtr(new AttrTypeInfo<int, 4, GL_INT>()),          // VEC4I

     BaseTypeInfoPtr(new AttrTypeInfo<float, 1, GL_FLOAT>()),      // FLOAT
     BaseTypeInfoPtr(new AttrTypeInfo<float, 2, GL_FLOAT>()),      // VEC2F
     BaseTypeInfoPtr(new AttrTypeInfo<float, 3, GL_FLOAT>()),      // VEC3F
     BaseTypeInfoPtr(new AttrTypeInfo<float, 4, GL_FLOAT>()),      // VEC4F

     BaseTypeInfoPtr(new AttrTypeInfo<double, 1, GL_DOUBLE>()),    // DOUBLE
     BaseTypeInfoPtr(new AttrTypeInfo<double, 2, GL_DOUBLE>()),    // VEC2D
     BaseTypeInfoPtr(new AttrTypeInfo<double, 3, GL_DOUBLE>()),    // VEC3D
     BaseTypeInfoPtr(new AttrTypeInfo<double, 4, GL_DOUBLE>())}    // VEC4D

};

// static std::vector<BaseTypeInfoPtr> attrTypeInfo = {
//     std::move(BaseTypeInfoPtr(new AttrTypeInfo<unsigned int, 1, GL_UNSIGNED_INT>())), // UINT

//     std::move(BaseTypeInfoPtr(new AttrTypeInfo<int, 1, GL_INT>())),          // INT
//     std::move(BaseTypeInfoPtr(new AttrTypeInfo<int, 2, GL_INT>())),          // VEC2I
//     std::move(BaseTypeInfoPtr(new AttrTypeInfo<int, 3, GL_INT>())),          // VEC3I
//     std::move(BaseTypeInfoPtr(new AttrTypeInfo<int, 4, GL_INT>())),          // VEC4I

//     std::move(BaseTypeInfoPtr(new AttrTypeInfo<int, 1, GL_FLOAT>())),        // FLOAT
//     std::move(BaseTypeInfoPtr(new AttrTypeInfo<float, 2, GL_FLOAT>())),      // VEC2F
//     std::move(BaseTypeInfoPtr(new AttrTypeInfo<float, 3, GL_FLOAT>())),      // VEC3F
//     std::move(BaseTypeInfoPtr(new AttrTypeInfo<float, 4, GL_FLOAT>())),      // VEC4F

//     std::move(BaseTypeInfoPtr(new AttrTypeInfo<int, 1, GL_DOUBLE>())),       // DOUBLE
//     std::move(BaseTypeInfoPtr(new AttrTypeInfo<double, 2, GL_DOUBLE>())),    // VEC2D
//     std::move(BaseTypeInfoPtr(new AttrTypeInfo<double, 3, GL_DOUBLE>())),    // VEC3D
//     std::move(BaseTypeInfoPtr(new AttrTypeInfo<double, 4, GL_DOUBLE>())),    // VEC4D
// };





enum LayoutType {
    INTERLEAVED = 0,
    SEQUENTIAL,
    CUSTOM
};



class BaseBufferLayout {
    public:
        BaseBufferLayout(LayoutType layoutType) : _layoutType(layoutType) {}
        virtual ~BaseBufferLayout() {}

        bool hasAttr(const std::string& attrName) {
            return (_attrMap.find(attrName) != _attrMap.end());
        }

        virtual void bindToRenderer() = 0;

    protected:
        LayoutType _layoutType;
        AttrMap _attrMap;
};

class CustomBufferLayout : public BaseBufferLayout {
    public:
        CustomBufferLayout() : BaseBufferLayout(CUSTOM) {}

        void addAttribute(const std::string& attrName, BufferAttrType type, int stride, int offset) {
            // TODO: throw exception instead
            assert(!hasAttr(attrName) && attrName.length());

            _attrMap[attrName] = BufferAttrInfoPtr(new BufferAttrInfo(type, attrTypeInfo[static_cast<int>(type)].get(), stride, offset));
        }

        void bindToRenderer() {}
};


class InterleavedBufferLayout : public BaseBufferLayout {
    public:
        InterleavedBufferLayout() : BaseBufferLayout(INTERLEAVED), _vertexByteSize(0) {}

        void addAttribute(const std::string& attrName, BufferAttrType type) {
            // TODO: throw exception instead
            assert(!hasAttr(attrName) && attrName.length());

            // TODO, set the stride of all currently existing attrs, or leave
            // that for when the layout is bound to the renderer/shader/VAO

            int enumVal = static_cast<int>(type);
            _attrMap[attrName] = BufferAttrInfoPtr(new BufferAttrInfo(type, attrTypeInfo[enumVal].get(), -1, _vertexByteSize));
            _vertexByteSize += attrTypeInfo[enumVal]->numBytes();
        }

        void bindToRenderer() {}

    private:
        int _vertexByteSize;
};


class SequentialBufferLayout : public BaseBufferLayout {
    public:
        SequentialBufferLayout() : BaseBufferLayout(SEQUENTIAL) {}

        void addAttribute(const std::string& attrName, BufferAttrType type) {
            // TODO: throw exception instead
            assert(!hasAttr(attrName) && attrName.length());

            int enumVal = static_cast<int>(type);
            _attrMap[attrName] = BufferAttrInfoPtr(new BufferAttrInfo(type, attrTypeInfo[enumVal].get(), attrTypeInfo[enumVal]->numBytes(), -1));
        }

        template <typename T, int numComponents=1>
        void addAttribute(const std::string& attrName) {
            // TODO: throw exception
            assert(!hasAttr(attrName) && attrName.length());

            BufferAttrType type = getBufferAttrType(T(0), numComponents);
            int enumVal = static_cast<int>(type);
            _attrMap[attrName] = BufferAttrInfoPtr(new BufferAttrInfo(type, attrTypeInfo[enumVal].get(), attrTypeInfo[enumVal]->numBytes(), -1));
        }

        void bindToRenderer() {}
};




} // namespace MapD_Renderer

#endif  // BUFFER_LAYOUT_H_

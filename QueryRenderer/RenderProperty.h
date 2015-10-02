#ifndef RENDER_PROPERTY_H_
#define RENDER_PROPERTY_H_

#include "ScaleConfig.h"
#include "VertexBuffer.h"
#include "BufferLayout.h"
#include <string>
#include <memory>
#include <iostream>
#include "rapidjson/document.h"

namespace MapD_Renderer {

template <typename T>
T getNumValFromJSONObj(const rapidjson::Value& obj) {
    assert(obj.IsNumber());

    T rtn(0);

    // TODO: do min/max checks?
    // How would we do this? implicit conversions apparently
    // take place in >=< operations and therefore
    // min/max checks here wouldn't work. For now, implicitly
    // coverting between possible types, but it could result
    // in undefined/unexpected behavior.
    // One way is to have a template specialization for
    // each basic type. Might be the only way to fix.
    // T max = std::numeric_limits<T>::max();
    // T min = std::numeric_limits<T>::lowest();

    if (obj.IsDouble()) {
        double val = obj.GetDouble();
        rtn = static_cast<T>(val);
    } else if (obj.IsInt()) {
        int val = obj.GetInt();
        rtn = static_cast<T>(val);
    } else if (obj.IsUint()) {
        unsigned int val = obj.GetUint();
        rtn = static_cast<T>(val);
    } // else if (obj.IsInt64()) {
    // } else if (obj.IsUInt64()) {
    // }

    return rtn;
}

template <typename T>
class RenderProperty {
    public:
        typedef std::shared_ptr<VertexBuffer> VboPtr;

        std::string name;
        std::shared_ptr<BaseScaleConfig> scale;
        T mult;
        T offset;

        // RenderProperty() : name(), scale(), mult(), offset() {}
        RenderProperty(const std::string& name) : name(name), scale(), mult(), offset() {}
        ~RenderProperty() {
            std::cerr << "IN RenderProperty DESTRUCTOR " << name << std::endl;
        }

        void initializeFromJSONObj(const rapidjson::Value& obj) {
            if (obj.IsObject()) {

            } else {
                _initValueFromJSONObj(obj);
            }
        }

    private:
        VboPtr _vboPtr;

        void _initValueFromJSONObj(const rapidjson::Value& obj) {
            T val = getNumValFromJSONObj<T>(obj);

            std::vector<T> data = { val };

            SequentialBufferLayout *vboLayout = new SequentialBufferLayout();
            vboLayout->addAttribute<T>(name);

            // vboLayout->addAttribute(name, BufferAttrType::FLOAT);
            VertexBuffer::BufferLayoutPtr vboLayoutPtr;
            vboLayoutPtr.reset(dynamic_cast<BaseBufferLayout *>(vboLayout));

            VertexBuffer *vbo = new VertexBuffer(data, vboLayoutPtr);
            _vboPtr.reset(vbo);
        }
};



// template <>
// void RenderProperty<float>::_initValueFromJSONObj(const rapidjson::Value& obj) {
//     assert(obj.IsDouble());
//     double val = obj.GetDouble();
//     assert(val <= std::numeric_limits<float>::max() && val >= std::numeric_limits<float>::min());

//     std::vector<float> data = { float(val) };

//     SequentialBufferLayout *vboLayout = new SequentialBufferLayout();
//     vboLayout->addAttribute(name, BufferAttrType::FLOAT);
//     // VertexBuffer::BufferLayoutPtr vboLayoutPtr = std::make_shared<BaseBufferLayout>(vboLayout);
//     VertexBuffer::BufferLayoutPtr vboLayoutPtr;
//     vboLayoutPtr.reset(dynamic_cast<BaseBufferLayout *>(vboLayout));

//     VertexBuffer *vbo = new VertexBuffer(data, vboLayoutPtr);
//     // VertexBuffer *vbo = new VertexBuffer(data, std::make_shared<BaseBufferLayout>(vboLayout));

//     _vboPtr.reset(vbo);
// }

// template <>
// void RenderProperty<int>::_initValueFromJSONObj(const rapidjson::Value& obj) {
//     assert(obj.IsInt());
//     int val = obj.GetInt();
// }

// template <typename T>
// class RenderProperty {
//     public:
//         std::string name;
//         std::shared_ptr<BaseScaleConfig> scale;
//         T mult;
//         T offset;

//         RenderProperty();
//     private:
// };



} // MapD_Renderer namespace

#endif // RENDER_PROPERTY_H_

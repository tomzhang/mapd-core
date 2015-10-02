// #include <vector>
// #include <limits>       // std::numeric_limits

#include "RenderProperty.h"

using namespace MapD_Renderer;

// template <typename T>
// RenderProperty<T>::RenderProperty() : name(), scale(), mult(), offset() {}

// template <typename T>
// RenderProperty<T>::RenderProperty(const std::string& name) : name(), scale(), mult(), offset() {}

// template <typename T>
// void RenderProperty<T>::initializeFromJSONObj(const rapidjson::Value& obj) {
//     if (obj.IsObject()) {

//     } else {
//         // _initValueFromJSONObj(obj);
//     }
// }


// template <>
// void RenderProperty<float>::_initValueFromJSONObj(const rapidjson::Value& obj) {
//     assert(obj.IsDouble());
//     double val = obj.GetDouble();

//     assert(val <= std::numeric_limits<float>::max() && val >= std::numeric_limits<float>::lowest());

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

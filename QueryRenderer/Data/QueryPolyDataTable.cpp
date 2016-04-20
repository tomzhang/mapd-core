#include "QueryPolyDataTable.h"

#include <Rendering/Objects/ColorRGBA.h>
#include <Rendering/Math/AABox.h>
#include <Rendering/Renderer/GL/GLRenderer.h>
#include <Rendering/Renderer/GL/Resources/GLBufferLayout.h>
#include <Rendering/Renderer/GL/Resources/GLShaderBlockLayout.h>

#include <poly2tri/poly2tri.h>

#include <boost/filesystem.hpp>
#include <boost/algorithm/string.hpp>

namespace QueryRenderer {

using ::Rendering::Objects::ColorRGBA;
using ::Rendering::GL::GLRenderer;
using ::Rendering::GL::Resources::GLBufferAttrType;
using ::Rendering::GL::Resources::GLBufferLayoutShPtr;
using ::Rendering::GL::Resources::GLInterleavedBufferLayout;
using ::Rendering::GL::Resources::GLSequentialBufferLayout;
using ::Rendering::GL::Resources::ShaderBlockLayoutType;
using ::Rendering::GL::Resources::GLShaderBlockLayout;
using ::Rendering::GL::Resources::GLShaderBlockLayoutShPtr;
using ::Rendering::GL::Resources::GLIndexBufferShPtr;
using ::Rendering::GL::Resources::GLUniformBufferShPtr;
using ::Rendering::GL::Resources::GLIndirectDrawVertexBufferShPtr;
using ::Rendering::GL::Resources::GLIndirectDrawIndexBufferShPtr;
using ::Rendering::GL::Resources::IndirectDrawIndexData;
using ::Rendering::GL::Resources::IndirectDrawVertexData;

GLIndexBufferShPtr BaseQueryPolyDataTable::getGLIndexBuffer(const GpuId& gpuId) const {
  auto itr = _perGpuData.find(gpuId);
  RUNTIME_EX_ASSERT(itr != _perGpuData.end(),
                    std::string(*this) + ": Cannot find gpu id: " + std::to_string(gpuId) + " in per-gpu resources.");

  return (itr->second.ibo ? itr->second.ibo->getGLIndexBufferPtr() : nullptr);
}

GLUniformBufferShPtr BaseQueryPolyDataTable::getGLUniformBuffer(const GpuId& gpuId) const {
  auto itr = _perGpuData.find(gpuId);
  RUNTIME_EX_ASSERT(itr != _perGpuData.end(),
                    std::string(*this) + ": Cannot find gpu id: " + std::to_string(gpuId) + " in per-gpu resources.");

  return (itr->second.ubo ? itr->second.ubo->getGLUniformBufferPtr() : nullptr);
}

GLIndirectDrawVertexBufferShPtr BaseQueryPolyDataTable::getGLIndirectDrawVertexBuffer(const GpuId& gpuId) const {
  auto itr = _perGpuData.find(gpuId);
  RUNTIME_EX_ASSERT(itr != _perGpuData.end(),
                    std::string(*this) + ": Cannot find gpu id: " + std::to_string(gpuId) + " in per-gpu resources.");

  return (itr->second.indvbo ? itr->second.indvbo->getGLIndirectVboPtr() : nullptr);
}

GLIndirectDrawIndexBufferShPtr BaseQueryPolyDataTable::getGLIndirectDrawIndexBuffer(const GpuId& gpuId) const {
  auto itr = _perGpuData.find(gpuId);
  RUNTIME_EX_ASSERT(itr != _perGpuData.end(),
                    std::string(*this) + ": Cannot find gpu id: " + std::to_string(gpuId) + " in per-gpu resources.");

  return (itr->second.indibo ? itr->second.indibo->getGLIndirectIboPtr() : nullptr);
}

std::vector<GpuId> BaseQueryPolyDataTable::getUsedGpuIds() const {
  std::vector<GpuId> rtn;

  for (auto& itr : _perGpuData) {
    rtn.push_back(itr.first);
  }

  return rtn;
}

void BaseQueryPolyDataTable::_initGpuResources(const QueryRendererContext* ctx) {
  // const QueryRendererContext::PerGpuDataMap& qrcPerGpuData = ctx->getGpuDataMap();

  // for (auto& itr : qrcPerGpuData) {
  //   if (_perGpuData.find(itr.first) == _perGpuData.end()) {
  //     PerGpuData gpuData(itr.second);

  //     if (!initializing) {
  //       switch (getType()) {
  //         // case QueryDataTableType::EMBEDDED:
  //         // case QueryDataTableType::URL:
  //         //   break;
  //         // case QueryDataTableType::SQLQUERY:
  //         //   gpuData.vbo = itr.second.getRootPerGpuData()->queryResultBufferPtr;
  //         //   break;
  //         default:
  //           THROW_RUNTIME_EX(std::string(*this) + ": Unsupported data table type for mult-gpu configuration: " +
  //                            std::to_string(static_cast<int>(getType())));
  //       }
  //     }

  //     _perGpuData.emplace(itr.first, std::move(gpuData));
  //   }
  // }

  // for (auto gpuId : unusedGpus) {
  //   _perGpuData.erase(gpuId);
  // }
}

void BaseQueryPolyDataTable::_initBuffers() {
  // CHECK(vboMap.size() == _perGpuData.size());

  // for (const auto& itr : vboMap) {
  //   auto myItr = _perGpuData.find(itr.first);
  //   CHECK(myItr != _perGpuData.end());
  //   myItr->second.vbo = itr.second;
  // }
}

template <typename T>
struct PolyData2d {
  // T xmin, xmax, ymin, ymax;
  ::Rendering::Math::AABox<T, 2> bounds;

  std::vector<T> x_coords;
  std::vector<T> y_coords;
  std::vector<unsigned int> triangulation_indices;
  std::vector<::Rendering::GL::Resources::IndirectDrawVertexData> lineDrawInfo;
  std::vector<::Rendering::GL::Resources::IndirectDrawIndexData> polyDrawInfo;

  PolyData2d(unsigned int startVert = 0, unsigned int startIdx = 0)
      : _ended(true), _startVert(startVert), _startIdx(startIdx), _startTriIdx(0) {}
  ~PolyData2d() {}

  size_t numVerts() const { return x_coords.size(); }
  size_t numPolys() const { return polyDrawInfo.size(); }
  size_t numLineLoops() const { return lineDrawInfo.size(); }
  size_t numTris() const {
    CHECK(triangulation_indices.size() % 3 == 0);
    return triangulation_indices.size() / 3;
  }
  size_t numIndices() const { return triangulation_indices.size(); }

  unsigned int startVert() const { return _startVert; }
  unsigned int startIdx() const { return _startIdx; }

  void beginPoly() {
    assert(_ended);
    _ended = false;
    _startTriIdx = numVerts() - lineDrawInfo.back().count;

    if (!polyDrawInfo.size()) {
      // polyDrawInfo.emplace_back(0, _startIdx + triangulation_indices.size(), lineDrawInfo.back().firstIndex);
      polyDrawInfo.emplace_back(0, _startIdx, _startVert);
    }
  }

  void endPoly() {
    assert(!_ended);
    _ended = true;
  }

  void beginLine() {
    assert(_ended);
    _ended = false;

    lineDrawInfo.emplace_back(0, _startVert + numVerts());
  }

  void addLinePoint(const std::shared_ptr<p2t::Point>& vertPtr) {
    _addPoint(static_cast<T>(vertPtr->x), static_cast<T>(vertPtr->y));
    lineDrawInfo.back().count++;
  }

  bool endLine() {
    bool rtn = false;
    auto& lineDrawItem = lineDrawInfo.back();
    size_t idx0 = lineDrawItem.firstIndex - _startVert;
    size_t idx1 = idx0 + (lineDrawItem.count - 1);
    if (x_coords[idx0] == x_coords[idx1] && y_coords[idx0] == y_coords[idx1]) {
      x_coords.pop_back();
      y_coords.pop_back();
      lineDrawItem.count--;
      rtn = true;
    }

    // add an empty coord as a separator
    // coords.push_back(-10000000.0);
    // coords.push_back(-10000000.0);

    _ended = true;
    return rtn;
  }

  void addTriangle(unsigned int idx0, unsigned int idx1, unsigned int idx2) {
    // triangulation_indices.push_back(idx0);
    // triangulation_indices.push_back(idx1);
    // triangulation_indices.push_back(idx2);

    triangulation_indices.push_back(_startTriIdx + idx0);
    triangulation_indices.push_back(_startTriIdx + idx1);
    triangulation_indices.push_back(_startTriIdx + idx2);

    polyDrawInfo.back().count += 3;
  }

 private:
  bool _ended;
  unsigned int _startVert;
  unsigned int _startIdx;
  unsigned int _startTriIdx;

  void _addPoint(T x, T y) {
    bounds.encapsulate({x, y});

    x_coords.push_back(x);
    y_coords.push_back(y);
  }
};

static rapidjson::Value::ConstMemberIterator validateCoordinate(const rapidjson::Value& obj,
                                                                const std::string& coordName) {
  rapidjson::Value::ConstMemberIterator mitr = obj.FindMember(coordName.c_str());

  RUNTIME_EX_ASSERT(
      mitr != obj.MemberEnd(),
      RapidJSONUtils::getJsonParseErrorStr(obj, "Polygonal data is missing the \"" + coordName + "\" coordinate."));

  RUNTIME_EX_ASSERT(mitr->value.IsArray(),
                    RapidJSONUtils::getJsonParseErrorStr(
                        obj, "Polygonal data for coordinate \"" + coordName + "\" must be an array."));

  RUNTIME_EX_ASSERT(
      mitr->value.Size() > 0,
      RapidJSONUtils::getJsonParseErrorStr(obj, "Data for polygonal coordinate \"" + coordName + "\" is empty."));

  const rapidjson::Value* array = &mitr->value;
  const rapidjson::Value& item = mitr->value[0];

  bool isArray;
  RUNTIME_EX_ASSERT((isArray = item.IsArray()) || item.IsNumber(),
                    RapidJSONUtils::getJsonParseErrorStr(
                        obj,
                        "Unsupported type for coordinate \"" + coordName +
                            "\". Coordinates must be arrays of numbers, or arrays of arrays of numbers."));

  if (isArray) {
    array = &item;
  }

  RUNTIME_EX_ASSERT(array->Size() >= 3,
                    RapidJSONUtils::getJsonParseErrorStr(
                        obj, "Coordinate \"" + coordName + "\" needs to have at least 3 values to create a polygon."));

  RUNTIME_EX_ASSERT((*array)[0].IsNumber(),
                    RapidJSONUtils::getJsonParseErrorStr(obj, "Coordinate \"" + coordName + "\" must be a number."));

  return mitr;
}

template <typename T>
static void buildPolygonFromJSONObj(const rapidjson::Value& xarray,
                                    const rapidjson::Value& yarray,
                                    PolyData2d<T>& polyData) {
  std::vector<std::shared_ptr<p2t::Point>> vertexShPtrs;
  std::vector<p2t::Point*> vertexPtrs;
  std::vector<int> tris;
  std::unordered_map<p2t::Point*, int> pointIndices;

  T x, y;
  double x_d, y_d;

  polyData.beginLine();

  // NOTE: xarray & yarray have already been validated (both are arrays and are of the same size)
  for (rapidjson::SizeType i = 0; i < xarray.Size(); i++) {
    RUNTIME_EX_ASSERT(
        xarray[i].IsNumber(),
        RapidJSONUtils::getJsonParseErrorStr(xarray,
                                             "Found a non-number at index " + std::to_string(i) +
                                                 " in the \"x\" coord of the polygon. All coords must be numbers."));

    RUNTIME_EX_ASSERT(
        yarray[i].IsNumber(),
        RapidJSONUtils::getJsonParseErrorStr(yarray,
                                             "Found a non-number at index " + std::to_string(i) +
                                                 " in the \"y\" coord of the polygon. All coords must be numbers."));

    x = RapidJSONUtils::getNumValFromJSONObj<T>(xarray[i]);
    y = RapidJSONUtils::getNumValFromJSONObj<T>(yarray[i]);

    // TODO(croot): provide some transform functions?
    // x_d = lon2x_d(x);
    // y_d = lat2y_d(y);

    x_d = static_cast<double>(x);
    y_d = static_cast<double>(y);

    if (!vertexPtrs.size() || vertexPtrs.back()->x != x_d || vertexPtrs.back()->y != y_d) {
      vertexShPtrs.emplace_back(new p2t::Point(x_d, y_d));
      polyData.addLinePoint(vertexShPtrs.back());
      vertexPtrs.push_back(vertexShPtrs.back().get());

      pointIndices.insert({vertexShPtrs.back().get(), vertexPtrs.size() - 1});
    }
  }

  if (polyData.endLine()) {
    p2t::Point* lastVert = vertexPtrs.back();
    vertexPtrs.pop_back();
    pointIndices.erase(lastVert);
    vertexShPtrs.pop_back();
  }

  p2t::CDT triangulator(vertexPtrs);

  triangulator.Triangulate();

  int idx0, idx1, idx2;

  std::unordered_map<p2t::Point*, int>::iterator itr;

  polyData.beginPoly();
  for (p2t::Triangle* tri : triangulator.GetTriangles()) {
    itr = pointIndices.find(tri->GetPoint(0));
    CHECK(itr != pointIndices.end()) << "Could not properly triangulate polygon. Could be self-intersecting.";
    idx0 = itr->second;

    itr = pointIndices.find(tri->GetPoint(1));
    CHECK(itr != pointIndices.end()) << "Could not properly triangulate polygon. Could be self-intersecting.";
    idx1 = itr->second;

    itr = pointIndices.find(tri->GetPoint(2));
    CHECK(itr != pointIndices.end()) << "Could not properly triangulate polygon. Could be self-intersecting.";
    idx2 = itr->second;

    polyData.addTriangle(idx0, idx1, idx2);
  }
  polyData.endPoly();
}

template <>
void TDataColumn<PolyData2d<double>>::push_back(const std::string& val) {
}

template <>
void TDataColumn<PolyData2d<double>>::_initFromRowMajorJSONObj(const rapidjson::Value& dataArrayObj) {
  RUNTIME_EX_ASSERT(dataArrayObj.IsArray(),
                    RapidJSONUtils::getJsonParseErrorStr(dataArrayObj, "Row-major data object is not an array."));

  rapidjson::Value::ConstValueIterator vitr;
  rapidjson::Value::ConstMemberIterator mitrx, mitry;

  for (vitr = dataArrayObj.Begin(); vitr != dataArrayObj.End(); ++vitr) {
    RUNTIME_EX_ASSERT(
        vitr->IsObject(),
        RapidJSONUtils::getJsonParseErrorStr(dataArrayObj,
                                             "Item " + std::to_string(vitr - dataArrayObj.Begin()) +
                                                 "in data array must be an object for row-major-defined data."));

    mitrx = validateCoordinate(*vitr, PolyDataTable::xcoordName);
    mitry = validateCoordinate(*vitr, PolyDataTable::ycoordName);

    // We know after validation that the objects at mitrx & mitry are arrays > 0
    RUNTIME_EX_ASSERT(
        mitrx->value.Size() == mitry->value.Size(),
        RapidJSONUtils::getJsonParseErrorStr(
            *vitr,
            "Item " + std::to_string(vitr - dataArrayObj.Begin()) +
                " in data array has mismatched sizes in the \"x\" & \"y\" coords. They must be the same length."));

    if (_columnDataPtr->size()) {
      _columnDataPtr->emplace_back(_columnDataPtr->back().startVert() + _columnDataPtr->back().numVerts(),
                                   _columnDataPtr->back().startIdx() + _columnDataPtr->back().numIndices());
    } else {
      _columnDataPtr->emplace_back();
    }
    if (mitrx->value[0].IsArray()) {
      for (rapidjson::SizeType i = 0; i < mitrx->value.Size(); i++) {
        RUNTIME_EX_ASSERT(
            mitrx->value[i].IsArray() && mitry->value[i].IsArray() && mitrx->value[i].Size() == mitry->value[i].Size(),
            RapidJSONUtils::getJsonParseErrorStr(*vitr,
                                                 "Item " + std::to_string(i) +
                                                     " in data array has mismatched types/sizes in the \"x\" & \"y\" "
                                                     "coords. They must be the same type & same length."));

        buildPolygonFromJSONObj<double>(mitrx->value[i], mitry->value[i], _columnDataPtr->back());
      }
    } else {
      buildPolygonFromJSONObj<double>(mitrx->value, mitry->value, _columnDataPtr->back());
    }
  }
}

template <>
QueryDataType TDataColumn<PolyData2d<double>>::getColumnType() {
  return QueryDataType::POLYGON_DOUBLE;
}

template <typename T>
static int getNumVertsInPolyColumn(TDataColumn<PolyData2d<T>>* polyDataCol) {
  int numVerts = 0;

  std::vector<PolyData2d<T>>* polyDataVec = polyDataCol->getColumnData().get();

  for (auto& polyData : (*polyDataVec)) {
    numVerts += polyData.numVerts();
  }

  return numVerts;
}

template <typename T>
static int getNumPolysInPolyColumn(TDataColumn<PolyData2d<T>>* polyDataCol) {
  int numPolys = 0;

  std::vector<PolyData2d<T>>* polyDataVec = polyDataCol->getColumnData().get();

  for (auto& polyData : (*polyDataVec)) {
    numPolys += polyData.numPolys();
  }

  return numPolys;
}

template <typename T>
static int getNumTrisInPolyColumn(TDataColumn<PolyData2d<T>>* polyDataCol) {
  int numTris = 0;

  std::vector<PolyData2d<T>>* polyDataVec = polyDataCol->getColumnData().get();

  for (auto& polyData : (*polyDataVec)) {
    numTris += polyData.numTris();
  }

  return numTris;
}

template <typename T>
static int getNumLineLoopsInPolyColumn(TDataColumn<PolyData2d<T>>* polyDataCol) {
  int numLineLoops = 0;

  std::vector<PolyData2d<T>>* polyDataVec = polyDataCol->getColumnData().get();

  for (auto& polyData : (*polyDataVec)) {
    numLineLoops += polyData.numLineLoops();
  }

  return numLineLoops;
}

template <typename T>
static int setSequentialPolygonalData(std::unique_ptr<char[]>& byteDataPtr,
                                      TDataColumn<PolyData2d<T>>* polyDataCol,
                                      int numTotalVerts) {
  int dataSz = sizeof(T);

  // get x & y data
  int numBytes = numTotalVerts * dataSz * 2;
  int numBytesPerItem = dataSz * 2;

  // now cpy the column data into one big buffer, sequentially, and
  // buffer it all to the gpu via the VBO.
  // char byteData[numBytes];
  byteDataPtr.reset(new char[numBytes]);
  char* byteData = byteDataPtr.get();

  // float byteData[numBytes/sizeof(float)];
  // memset(byteData, 0x0, numBytes);
  // memset(&byteData[0], 0x0, numBytes);
  memset(byteData, 0x0, numBytes);

  int startIdx = 0;
  int bufSz;

  std::vector<PolyData2d<T>>* polyDataVec = polyDataCol->getColumnData().get();

  // get x coords first
  for (auto& polyData : (*polyDataVec)) {
    bufSz = polyData.x_coords.size() * dataSz;
    memcpy(&byteData[startIdx], &polyData.x_coords[0], bufSz);
    startIdx += bufSz;
  }

  // now get y coords
  for (auto& polyData : (*polyDataVec)) {
    bufSz = polyData.y_coords.size() * dataSz;
    memcpy(&byteData[startIdx], &polyData.y_coords[0], bufSz);
    startIdx += bufSz;
  }

  return numBytesPerItem;
}

template <typename T>
static int setInterleavedPolygonalData(std::unique_ptr<char[]>& byteDataPtr,
                                       TDataColumn<PolyData2d<T>>* polyDataCol,
                                       int numTotalVerts) {
  int dataSz = sizeof(T);

  // get x & y data
  int numBytes = numTotalVerts * dataSz * 2;
  int numBytesPerItem = dataSz * 2;

  // now cpy the column data into one big buffer, sequentially, and
  // buffer it all to the gpu via the VBO.
  // char byteData[numBytes];
  byteDataPtr.reset(new char[numBytes]);
  char* byteData = byteDataPtr.get();

  // float byteData[numBytes/sizeof(float)];
  // memset(byteData, 0x0, numBytes);
  // memset(&byteData[0], 0x0, numBytes);
  memset(byteData, 0x0, numBytes);

  int startIdx = 0;

  std::vector<PolyData2d<T>>* polyDataVec = polyDataCol->getColumnData().get();

  // get x/y coords in interleaved fashion
  for (auto& polyData : (*polyDataVec)) {
    for (size_t i = 0; i < polyData.numVerts(); ++i) {
      memcpy(&byteData[startIdx], &polyData.x_coords[i], dataSz);
      startIdx += dataSz;
      memcpy(&byteData[startIdx], &polyData.y_coords[i], dataSz);
      startIdx += dataSz;
    }
  }

  return numBytesPerItem;
}

static const rapidjson::Value& getCoordStartItem(const rapidjson::Value::ConstMemberIterator& mitr) {
  // NOTE: validation of mitr will have taken place before getting in here
  // via the validateCoordinate() function
  const rapidjson::Value* array = &mitr->value;
  const rapidjson::Value& item = mitr->value[0];

  if (item.IsArray()) {
    array = &item;
  }

  return (*array)[0];
}

std::tuple<DataColumnUqPtr, int, int, int, int> createPolyDataColumnFromRowMajorObj(
    const std::string& columnName,
    const rapidjson::Value::ConstMemberIterator& x_coord,
    const rapidjson::Value::ConstMemberIterator& y_coord,
    const rapidjson::Value& dataArray) {
  const rapidjson::Value& xitem = getCoordStartItem(x_coord);
  const rapidjson::Value& yitem = getCoordStartItem(y_coord);
  // if (xitem.IsInt()) {
  //   RUNTIME_EX_ASSERT(
  //       yitem.IsInt(),
  //       RapidJSONUtils::getJsonParseErrorStr(dataArray, "x and y coordinates of the array are different types."));
  //   return DataColumnUqPtr(new TDataColumn<PolyData2d<int>>(columnName, dataArray, DataColumn::InitType::ROW_MAJOR));
  // } else if (xitem.IsUint()) {
  //   RUNTIME_EX_ASSERT(
  //       yitem.IsUint(),
  //       RapidJSONUtils::getJsonParseErrorStr(dataArray, "x and y coordinates of the array are different types."));
  //   return DataColumnUqPtr(
  //       new TDataColumn<PolyData2d<unsigned int>>(columnName, dataArray, DataColumn::InitType::ROW_MAJOR));
  // }
  DataColumnUqPtr polyDataColumnPtr;
  if (xitem.IsDouble()) {
    RUNTIME_EX_ASSERT(
        yitem.IsDouble(),
        RapidJSONUtils::getJsonParseErrorStr(dataArray, "x and y coordinates of the array are different types."));
    // TODO(croot): How do we properly handle floats?
    polyDataColumnPtr.reset(
        new TDataColumn<PolyData2d<double>>(columnName, dataArray, DataColumn::InitType::ROW_MAJOR));
    auto polyDataColumn = dynamic_cast<TDataColumn<PolyData2d<double>>*>(polyDataColumnPtr.get());
    CHECK(polyDataColumn != nullptr);

    int numVerts = getNumVertsInPolyColumn(polyDataColumn);
    int numPolys = getNumPolysInPolyColumn(polyDataColumn);
    int numTris = getNumTrisInPolyColumn(polyDataColumn);
    int numLineLoops = getNumLineLoopsInPolyColumn(polyDataColumn);

    return std::make_tuple(std::move(polyDataColumnPtr), numVerts, numPolys, numTris, numLineLoops);

    // double val = rowItem.GetDouble();
    // if (val <= std::numeric_limits<float>::max() && val >= std::numeric_limits<float>::lowest()) {
    //   return DataColumnUqPtr(new TDataColumn<float>(columnName, dataArray, DataColumn::InitType::ROW_MAJOR));
    // } else {
    //   return DataColumnUqPtr(new TDataColumn<double>(columnName, dataArray, DataColumn::InitType::ROW_MAJOR));
    // }
  } else {
    THROW_RUNTIME_EX(RapidJSONUtils::getJsonParseErrorStr(
        xitem, "Cannot create poly data. The JSON data type for the coordinates is not supported."));
  }
}

std::string PolyDataTable::defaultPolyDataColumnName = "__polydata__";
std::string PolyDataTable::xcoordName = "x";
std::string PolyDataTable::ycoordName = "y";

PolyDataTable::PolyDataTable(const QueryRendererContextShPtr& ctx,
                             const std::string& name,
                             const rapidjson::Value& obj,
                             const rapidjson::Pointer& objPath,
                             QueryDataTableType type,
                             bool buildIdColumn,
                             DataTable::VboType vboType)
    : BaseQueryPolyDataTable(ctx, name, obj, objPath, type),
      _vboType(vboType),
      _numRows(0),
      _numVerts(0),
      _numPolys(0),
      _numTris(0),
      _numLineLoops(0) {
  _buildPolyDataFromJSONObj(obj, objPath, buildIdColumn);
}

PolyDataTable::~PolyDataTable() {
}

bool PolyDataTable::hasAttribute(const std::string& attrName) {
  if (attrName == xcoordName || attrName == ycoordName) {
    return true;
  } else {
    ColumnMap_by_name& nameLookup = _columns.get<DataColumn::ColumnName>();
    return (nameLookup.find(attrName) != nameLookup.end());
  }
}

QueryBufferShPtr PolyDataTable::getAttributeDataBuffer(const GpuId& gpuId, const std::string& attrName) {
  auto itr = _perGpuData.find(gpuId);

  RUNTIME_EX_ASSERT(itr != _perGpuData.end(),
                    std::string(*this) + ": Cannot get attribute data buffer for gpu " + std::to_string(gpuId));

  _initBuffers(itr->second);

  if (itr->second.vbo->hasAttribute(attrName)) {
    return itr->second.vbo;
  } else if (itr->second.ubo->hasAttribute(attrName)) {
    return itr->second.ubo;
  } else {
    THROW_RUNTIME_EX(std::string(*this) + " getAttributeDataBuffer(): attribute \"" + attrName + "\" does not exist.");
  }

  return nullptr;
}

std::map<GpuId, QueryBufferShPtr> PolyDataTable::getAttributeDataBuffers(const std::string& attrName) {
  std::map<GpuId, QueryBufferShPtr> rtn;
  std::map<GpuId, QueryBufferShPtr>::iterator insertedItr;
  std::pair<GLBufferLayoutShPtr, std::pair<std::unique_ptr<char[]>, size_t>> vboData;

  for (auto& itr : _perGpuData) {
    _initBuffers(itr.second);

    if (itr.second.vbo->hasAttribute(attrName)) {
      insertedItr = rtn.insert({itr.first, itr.second.vbo}).first;
    } else if (itr.second.ubo->hasAttribute(attrName)) {
      insertedItr = rtn.insert({itr.first, itr.second.ubo}).first;
    } else {
      THROW_RUNTIME_EX(std::string(*this) + " getAttributeDataBuffer(): attribute \"" + attrName +
                       "\" does not exist in table.");
    }

    CHECK(rtn.begin()->second->getGLResourceType() == insertedItr->second->getGLResourceType());
  }

  return rtn;
}

QueryDataType PolyDataTable::getAttributeType(const std::string& attrName) {
  // all buffers should have the same set of attributes, so only need to check the first one.
  auto itr = _perGpuData.begin();
  CHECK(itr != _perGpuData.end());

  GLBufferAttrType attrType;
  if (itr->second.vbo->hasAttribute(attrName)) {
    attrType = itr->second.vbo->getAttributeType(attrName);
  } else if (itr->second.ubo->hasAttribute(attrName)) {
    attrType = itr->second.ubo->getAttributeType(attrName);
  } else {
    THROW_RUNTIME_EX(std::string(*this) + " getAttributeType(): attribute \"" + attrName +
                     "\" does not exist in table.");
  }

  switch (attrType) {
    case GLBufferAttrType::UINT:
      return QueryDataType::UINT;
    case GLBufferAttrType::INT:
      return QueryDataType::INT;
    case GLBufferAttrType::FLOAT:
      return QueryDataType::FLOAT;
    case GLBufferAttrType::DOUBLE:
      return QueryDataType::DOUBLE;
    case GLBufferAttrType::VEC4F:
      return QueryDataType::COLOR;
    default:
      THROW_RUNTIME_EX(std::string(*this) + " getColumnType(): Vertex buffer attribute type: " +
                       std::to_string(static_cast<int>(attrType)) + " is not a supported type.");
  }
}

// template <typename C1, typename C2>
// std::pair<C1, C2> getExtrema(const std::string& column);

// bool hasColumn(const std::string& columnName) {
//   ColumnMap_by_name& nameLookup = _columns.get<DataColumn::ColumnName>();
//   return (nameLookup.find(columnName) != nameLookup.end());
// }

// QueryDataType getColumnType(const std::string& columnName);
// DataColumnShPtr getColumn(const std::string& columnName);

// QueryVertexBufferShPtr getColumnDataVBO(const GpuId& gpuId, const std::string& columnName);
// std::map<GpuId, QueryVertexBufferShPtr> getColumnDataVBOs(const std::string& columnName) final;

PolyDataTable::operator std::string() const {
  return "PolyDataTable(name: " + _name + ", num columns: " + std::to_string(_columns.size()) + ") " + _printInfo();
}

void PolyDataTable::_updateFromJSONObj(const rapidjson::Value& obj, const rapidjson::Pointer& objPath) {
  CHECK(false) << "PolyDataTable::_updateFromJSONObj() has yet to be implemented.";
}

void PolyDataTable::_readDataFromFile(const std::string& filename) {
  boost::filesystem::path p(filename);  // avoid repeated path construction below

  RUNTIME_EX_ASSERT(boost::filesystem::exists(p),
                    RapidJSONUtils::getJsonParseErrorStr(
                        _ctx->getUserWidgetIds(), std::string(*this) + ": File " + filename + " does not exist."));

  RUNTIME_EX_ASSERT(boost::filesystem::is_regular_file(p),
                    RapidJSONUtils::getJsonParseErrorStr(
                        _ctx->getUserWidgetIds(),
                        std::string(*this) + ": File " + filename +
                            " is not a regular file. Cannot read contents to build a poly data table."));

  RUNTIME_EX_ASSERT(p.has_extension(),
                    RapidJSONUtils::getJsonParseErrorStr(
                        _ctx->getUserWidgetIds(),
                        std::string(*this) + ": File " + filename +
                            " does not have an extension. Cannot read contents to build a poly data table."));

  std::string ext = p.extension().string();
  boost::to_lower(ext);

  // if (ext == ".shp") {
  //   readVerticesFromShapeFile(filename);
  // } else {
  //   THROW_RUNTIME_EX(RapidJSONUtils::getJsonParseErrorStr(_ctx->getUserWidgetIds(),
  //                                                         std::string(*this) + ": File " + filename +
  //                                                             " with extension \"" + ext +
  //                                                             "\" is not a supported poly data file."));
  // }

  THROW_RUNTIME_EX(RapidJSONUtils::getJsonParseErrorStr(_ctx->getUserWidgetIds(),
                                                        std::string(*this) + ": File " + filename +
                                                            " with extension \"" + ext +
                                                            "\" is not a supported poly data file."));
}

void PolyDataTable::_buildPolyRowsFromJSONObj(const rapidjson::Value& obj) {
  // NOTE: obj has already been verified to be an array and have at least 1 item

  const rapidjson::Value& item = obj[0];

  RUNTIME_EX_ASSERT(
      item.IsObject(),
      RapidJSONUtils::getJsonParseErrorStr(
          _ctx->getUserWidgetIds(), item, "every row of JSON polygon data must be defined as an object."));

  rapidjson::Value::ConstMemberIterator mitr, mitrx, mitry;

  mitrx = validateCoordinate(item, xcoordName);
  mitry = validateCoordinate(item, ycoordName);

  RUNTIME_EX_ASSERT(item.FindMember("z") == item.MemberEnd(),
                    RapidJSONUtils::getJsonParseErrorStr(
                        _ctx->getUserWidgetIds(), item, "Z coordinates for polygons are currently unsupported."));

  auto dataTuple = createPolyDataColumnFromRowMajorObj(defaultPolyDataColumnName, mitrx, mitry, obj);
  _columns.push_back(std::move(std::get<0>(dataTuple)));
  _numVerts = std::get<1>(dataTuple);
  _numPolys = std::get<2>(dataTuple);
  _numTris = std::get<3>(dataTuple);
  _numLineLoops = std::get<4>(dataTuple);

  // init all non-coord columns
  for (mitr = item.MemberBegin(); mitr != item.MemberEnd(); ++mitr) {
    if (mitr == mitrx || mitr == mitry) {
      continue;
    }

    std::string colName = mitr->name.GetString();

    // TODO: Support strings? bools? Anything else?
    if (mitr->value.IsNumber()) {
      _columns.push_back(DataTable::createDataColumnFromRowMajorObj(mitr->name.GetString(), mitr->value, obj));
    } else if (mitr->value.IsString()) {
      std::string val = mitr->value.GetString();
      RUNTIME_EX_ASSERT(
          ColorRGBA::isColorString(val),
          RapidJSONUtils::getJsonParseErrorStr(_ctx->getUserWidgetIds(),
                                               item,
                                               "Unsupported string for polygonal data column \"" + colName +
                                                   "\". Only color strings are currently supported."));
      _columns.push_back(DataTable::createColorDataColumnFromRowMajorObj(colName, mitr->value, obj));
    } else {
      THROW_RUNTIME_EX(RapidJSONUtils::getJsonParseErrorStr(
          _ctx->getUserWidgetIds(),
          item,
          "Unsupported type for polygonal data column \"" + std::string(mitr->name.GetString()) + "\""));
    }
  }
}

void PolyDataTable::_buildPolyDataFromJSONObj(const rapidjson::Value& obj,
                                              const rapidjson::Pointer& objPath,
                                              bool buildIdColumn) {
  RUNTIME_EX_ASSERT(obj.IsObject(),
                    RapidJSONUtils::getJsonParseErrorStr(
                        _ctx->getUserWidgetIds(), obj, "data must be an object. Cannot build data table from JSON."));

  rapidjson::Value::ConstMemberIterator mitr1;

  RUNTIME_EX_ASSERT((mitr1 = obj.FindMember("format")) != obj.MemberEnd() && mitr1->value.IsString() &&
                        std::string(mitr1->value.GetString()) == "polys",
                    RapidJSONUtils::getJsonParseErrorStr(
                        _ctx->getUserWidgetIds(),
                        obj,
                        "polygon data must have a \"format\" string property that is set to \"polys\""));

  if ((mitr1 = obj.FindMember("values")) != obj.MemberEnd()) {
    RUNTIME_EX_ASSERT(mitr1->value.IsArray(),
                      RapidJSONUtils::getJsonParseErrorStr(
                          _ctx->getUserWidgetIds(), obj, "\"values\" property in the json must be an array."));

    // row format in an array

    // TODO(croot) - should we just log a warning if no data is supplied instead?
    RUNTIME_EX_ASSERT(!mitr1->value.Empty(),
                      RapidJSONUtils::getJsonParseErrorStr(
                          _ctx->getUserWidgetIds(), mitr1->value, "there is no polygon data defined."));

    _buildPolyRowsFromJSONObj(mitr1->value);

  } else if ((mitr1 = obj.FindMember("url")) != obj.MemberEnd()) {
    RUNTIME_EX_ASSERT(
        mitr1->value.IsString(),
        RapidJSONUtils::getJsonParseErrorStr(_ctx->getUserWidgetIds(), obj, "\"url\" property must be a string."));

    _readDataFromFile(mitr1->value.GetString());
  } else {
    THROW_RUNTIME_EX(RapidJSONUtils::getJsonParseErrorStr(
        _ctx->getUserWidgetIds(), obj, "JSON data object must contain either a \"values\" or \"url\" property."));
  }

  // TODO(croot) - throw a warning instead if no data?
  RUNTIME_EX_ASSERT(_columns.size(),
                    RapidJSONUtils::getJsonParseErrorStr(
                        _ctx->getUserWidgetIds(), obj, "there are no columns in the poly data table."));
  _numRows = (*_columns.begin())->size();

  if (buildIdColumn) {
    TDataColumn<unsigned int>* idColumn = new TDataColumn<unsigned int>(DataTable::defaultIdColumnName, _numRows);

    for (int i = 0; i < _numRows; ++i) {
      (*idColumn)[i] = i;
    }

    _columns.push_back(DataColumnUqPtr(idColumn));
  }
}

DataColumnShPtr PolyDataTable::_getPolyDataColumn() {
  ColumnMap_by_name& nameLookup = _columns.get<DataColumn::ColumnName>();
  auto polyitr = nameLookup.find(defaultPolyDataColumnName);
  CHECK(polyitr != nameLookup.end());

  return *polyitr;
}

void PolyDataTable::_initBuffers(BaseQueryPolyDataTable::PerGpuData& gpuData) {
  if (gpuData.vbo == nullptr) {
    gpuData.makeActiveOnCurrentThread();

    RootPerGpuDataShPtr qrmGpuData = gpuData.getRootPerGpuData();
    CHECK(qrmGpuData && qrmGpuData->rendererPtr);
    GLRenderer* renderer = dynamic_cast<GLRenderer*>(qrmGpuData->rendererPtr.get());
    CHECK(renderer != nullptr);

    auto vboData = _createVBOData();
    gpuData.vbo.reset(new QueryVertexBuffer(renderer, std::get<0>(vboData)));
    gpuData.vbo->bufferData(std::get<1>(vboData).get(), _numVerts, std::get<2>(vboData));

    auto uboData = _createUBOData();
    gpuData.ubo.reset(new QueryUniformBuffer(renderer, std::get<0>(uboData)));
    gpuData.ubo->bufferData(std::get<1>(uboData).get(), _numRows, std::get<2>(uboData));

    auto iboData = _createIBOData();
    gpuData.ibo.reset(new QueryIndexBuffer(renderer, iboData));

    auto lineData = _createLineDrawData();
    gpuData.indvbo.reset(new QueryIndirectVbo(renderer, lineData));

    auto vertexData = _createPolyDrawData();
    gpuData.indibo.reset(new QueryIndirectIbo(renderer, vertexData));
  }
}

std::tuple<GLBufferLayoutShPtr, std::unique_ptr<char[]>, size_t> PolyDataTable::_createVBOData() {
  GLBufferLayoutShPtr vboLayoutPtr;
  QueryVertexBufferShPtr vbo;

  auto polyDataColumn = _getPolyDataColumn();

  switch (_vboType) {
    case DataTable::VboType::SEQUENTIAL: {
      vboLayoutPtr.reset(new GLSequentialBufferLayout());
      auto vboLayout = dynamic_cast<GLSequentialBufferLayout*>(vboLayoutPtr.get());

      // build up the layout of the vertex buffer

      // TODO(croot): We need to add per-vertex attributes here if we
      // support them ever, like fill/stroke color defined per vertex,
      // or varying stroke widths

      int numBytesPerItem = 0;

      std::unique_ptr<char[]> byteDataPtr;

      switch (polyDataColumn->getColumnType()) {
        case QueryDataType::POLYGON_DOUBLE: {
          vboLayout->addAttribute(xcoordName, GLBufferAttrType::DOUBLE);
          vboLayout->addAttribute(ycoordName, GLBufferAttrType::DOUBLE);

          auto tPolyDataColumn = dynamic_cast<TDataColumn<PolyData2d<double>>*>(polyDataColumn.get());
          numBytesPerItem = setSequentialPolygonalData<double>(byteDataPtr, tPolyDataColumn, _numVerts);

        } break;
        default:
          THROW_RUNTIME_EX(std::string(*this) +
                           ": Column type for polygonal data is not supported. Cannot build vertex buffer.");
          break;
      }

      return std::make_tuple(vboLayoutPtr, std::move(byteDataPtr), numBytesPerItem);

    } break;

    case DataTable::VboType::INTERLEAVED: {
      vboLayoutPtr.reset(new GLInterleavedBufferLayout());
      auto vboLayout = dynamic_cast<GLInterleavedBufferLayout*>(vboLayoutPtr.get());

      // build up the layout of the vertex buffer

      // TODO(croot): We need to add per-vertex attributes here if we
      // support them ever, like fill/stroke color defined per vertex,
      // or varying stroke widths

      int numBytesPerItem = 0;

      std::unique_ptr<char[]> byteDataPtr;

      switch (polyDataColumn->getColumnType()) {
        case QueryDataType::POLYGON_DOUBLE: {
          vboLayout->addAttribute(xcoordName, GLBufferAttrType::DOUBLE);
          vboLayout->addAttribute(ycoordName, GLBufferAttrType::DOUBLE);

          auto tPolyDataColumn = dynamic_cast<TDataColumn<PolyData2d<double>>*>(polyDataColumn.get());
          numBytesPerItem = setInterleavedPolygonalData<double>(byteDataPtr, tPolyDataColumn, _numVerts);

        } break;
        default:
          THROW_RUNTIME_EX(std::string(*this) +
                           ": Column type for polygonal data is not supported. Cannot build vertex buffer.");
          break;
      }

      return std::make_tuple(vboLayoutPtr, std::move(byteDataPtr), numBytesPerItem);

    } break;
    case DataTable::VboType::INDIVIDUAL:
      // TODO: What kind of data structure should we do in this case? Should we do
      // one big unordered map, but there's only one item in the SEQUENTIAL &
      // INTERLEAVED case?
      THROW_RUNTIME_EX(std::string(*this) + ": Individual buffers for the poly data has yet to be implemented.")
      break;
  }

  return std::make_tuple(nullptr, nullptr, 0);
}

std::tuple<GLShaderBlockLayoutShPtr, std::unique_ptr<char[]>, size_t> PolyDataTable::_createUBOData() {
  GLShaderBlockLayoutShPtr blockLayoutPtr(new GLShaderBlockLayout(ShaderBlockLayoutType::STD140));

  ColumnMap::iterator itr;

  // don't include the polydata column
  std::vector<std::pair<TypelessColumnData, int>> columnData(_columns.size() - 1);

  int idx = 0;
  blockLayoutPtr->beginAddingAttrs();
  for (itr = _columns.begin(); itr != _columns.end(); ++itr) {
    if ((*itr)->columnName == defaultPolyDataColumnName) {
      continue;
    }

    switch ((*itr)->getColumnType()) {
      case QueryDataType::UINT:
        blockLayoutPtr->addAttribute<unsigned int>((*itr)->columnName);
        break;
      case QueryDataType::INT:
        blockLayoutPtr->addAttribute<int>((*itr)->columnName);
        break;
      case QueryDataType::FLOAT:
        blockLayoutPtr->addAttribute<float>((*itr)->columnName);
        break;
      case QueryDataType::DOUBLE:
        blockLayoutPtr->addAttribute<double>((*itr)->columnName);
        break;
      case QueryDataType::COLOR:
        blockLayoutPtr->addAttribute<float, 4>((*itr)->columnName);
        break;
      default:
        THROW_RUNTIME_EX(std::string(*this) + ": Column type for column \"" + (*itr)->columnName +
                         "\" in data table \"" + _name + "\" is not supported. Cannot build vertex buffer.");
        break;
    }

    columnData[idx] = std::make_pair(std::move((*itr)->getTypelessColumnData()),
                                     blockLayoutPtr->getAttributeByteOffset((*itr)->columnName));

    idx++;
  }
  blockLayoutPtr->endAddingAttrs();

  // now cpy the column data into one big buffer, interleaving the data, and
  // buffer it all to the gpu via the VBO.
  // char byteData[numBytes];
  size_t bytesInBlock = blockLayoutPtr->getNumBytesInBlock();
  size_t numBytes = bytesInBlock * _numRows;

  std::unique_ptr<char[]> byteDataPtr(new char[numBytes]);
  char* byteData = byteDataPtr.get();
  memset(byteData, 0x0, numBytes);

  int startIdx = 0;
  int offset = 0;
  int bytesPerItem;
  for (int i = 0; i < _numRows; ++i) {
    for (size_t j = 0; j < columnData.size(); ++j) {
      bytesPerItem = columnData[j].first.numBytesPerItem;
      offset = columnData[j].second;
      memcpy(&byteData[startIdx + offset],
             static_cast<char*>(columnData[j].first.data) + (i * bytesPerItem),
             bytesPerItem);
    }
    startIdx += bytesInBlock;
  }

  return std::make_tuple(blockLayoutPtr, std::move(byteDataPtr), bytesInBlock);
}

std::vector<unsigned int> PolyDataTable::_createIBOData() {
  std::vector<unsigned int> combined_idxs;
  auto polyDataColumn = _getPolyDataColumn();
  switch (polyDataColumn->getColumnType()) {
    case QueryDataType::POLYGON_DOUBLE: {
      auto tPolyDataColumn = dynamic_cast<TDataColumn<PolyData2d<double>>*>(polyDataColumn.get());

      CHECK(_numTris);
      combined_idxs.resize(_numTris * 3);

      std::vector<PolyData2d<double>>* polyDataVec = tPolyDataColumn->getColumnData().get();

      // get all the triangle indices combined
      size_t idx = 0;
      for (auto& polyData : (*polyDataVec)) {
        for (size_t i = 0; i < polyData.triangulation_indices.size(); ++i, ++idx) {
          combined_idxs[idx] = polyData.triangulation_indices[i];
        }
      }

    } break;
    default:
      THROW_RUNTIME_EX(std::string(*this) +
                       ": Column type for polygonal data is not supported. Cannot build vertex buffer.");
      break;
  }

  return combined_idxs;
}

std::vector<IndirectDrawVertexData> PolyDataTable::_createLineDrawData() {
  std::vector<IndirectDrawVertexData> vertData;
  auto polyDataColumn = _getPolyDataColumn();

  switch (polyDataColumn->getColumnType()) {
    case QueryDataType::POLYGON_DOUBLE: {
      auto tPolyDataColumn = dynamic_cast<TDataColumn<PolyData2d<double>>*>(polyDataColumn.get());

      vertData.resize(_numLineLoops);
      std::vector<PolyData2d<double>>* polyDataVec = tPolyDataColumn->getColumnData().get();

      // get all the triangle indices combined
      size_t idx = 0;
      for (auto& polyData : (*polyDataVec)) {
        for (size_t i = 0; i < polyData.lineDrawInfo.size(); ++i, ++idx) {
          vertData[idx] = polyData.lineDrawInfo[i];
        }
      }

    } break;
    default:
      THROW_RUNTIME_EX(std::string(*this) +
                       ": Column type for polygonal data is not supported. Cannot build vertex buffer.");
      break;
  }

  return vertData;
}

std::vector<IndirectDrawIndexData> PolyDataTable::_createPolyDrawData() {
  std::vector<IndirectDrawIndexData> indexData;
  auto polyDataColumn = _getPolyDataColumn();

  switch (polyDataColumn->getColumnType()) {
    case QueryDataType::POLYGON_DOUBLE: {
      auto tPolyDataColumn = dynamic_cast<TDataColumn<PolyData2d<double>>*>(polyDataColumn.get());

      indexData.resize(_numPolys);
      std::vector<PolyData2d<double>>* polyDataVec = tPolyDataColumn->getColumnData().get();

      size_t idx = 0;
      for (auto& polyData : (*polyDataVec)) {
        for (size_t i = 0; i < polyData.polyDrawInfo.size(); ++i, ++idx) {
          indexData[idx] = polyData.polyDrawInfo[i];
        }
      }

    } break;
    default:
      THROW_RUNTIME_EX(std::string(*this) +
                       ": Column type for polygonal data is not supported. Cannot build vertex buffer.");
      break;
  }

  return indexData;
}

}  // namespace QueryRenderer

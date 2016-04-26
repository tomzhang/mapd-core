#ifndef QUERYRENDERER_DATA_QUERYPOLYDATATABLE_H_
#define QUERYRENDERER_DATA_QUERYPOLYDATATABLE_H_

#include "QueryDataTable.h"
#include <Rendering/Renderer/GL/Resources/Types.h>

namespace QueryRenderer {

class BaseQueryPolyDataTable : public BaseQueryDataTable {
 public:
  static std::string xcoordName;
  static std::string ycoordName;

  BaseQueryPolyDataTable(const RootCacheShPtr& qrmGpuCache, QueryDataTableType type)
      : BaseQueryDataTable(type, QueryDataTableBaseType::POLY) {}
  virtual ~BaseQueryPolyDataTable() {}

  ::Rendering::GL::Resources::GLIndexBufferShPtr getGLIndexBuffer(const GpuId& gpuId) const;
  ::Rendering::GL::Resources::GLUniformBufferShPtr getGLUniformBuffer(const GpuId& gpuId) const;
  ::Rendering::GL::Resources::GLIndirectDrawVertexBufferShPtr getGLIndirectDrawVertexBuffer(const GpuId& gpuId) const;
  ::Rendering::GL::Resources::GLIndirectDrawIndexBufferShPtr getGLIndirectDrawIndexBuffer(const GpuId& gpuId) const;

  bool usesGpu(const GpuId& gpuId) const { return (_perGpuData.find(gpuId) != _perGpuData.end()); }
  std::vector<GpuId> getUsedGpuIds() const final;

  PolyTableDataInfo getPolyBufferData(const GpuId& gpuId) const;

  BaseQueryPolyDataTable& operator=(const BaseQueryPolyDataTable& rhs);

 protected:
  struct PerGpuData : BasePerGpuData {
    QueryVertexBufferShPtr vbo;
    QueryIndexBufferShPtr ibo;
    QueryUniformBufferShPtr ubo;
    QueryIndirectVboShPtr indvbo;
    QueryIndirectIboShPtr indibo;

    PerGpuData() : BasePerGpuData(), vbo(nullptr), ibo(nullptr), ubo(nullptr), indvbo(nullptr), indibo(nullptr) {}
    explicit PerGpuData(const BasePerGpuData& data,
                        const QueryVertexBufferShPtr& vbo = nullptr,
                        const QueryIndexBufferShPtr& ibo = nullptr,
                        const QueryUniformBufferShPtr& ubo = nullptr,
                        const QueryIndirectVboShPtr& indvbo = nullptr,
                        const QueryIndirectIboShPtr& indibo = nullptr)
        : BasePerGpuData(data), vbo(vbo), ibo(ibo), ubo(ubo), indvbo(indvbo), indibo(indibo) {}
    PerGpuData(const PerGpuData& data)
        : BasePerGpuData(data), vbo(data.vbo), ibo(data.ibo), ubo(data.ubo), indvbo(data.indvbo), indibo(data.indibo) {}
    PerGpuData(PerGpuData&& data)
        : BasePerGpuData(std::move(data)),
          vbo(std::move(data.vbo)),
          ibo(std::move(data.ibo)),
          ubo(std::move(data.ubo)),
          indvbo(std::move(data.indvbo)),
          indibo(std::move(data.indibo)) {}

    ~PerGpuData() {
      // need to make active to properly delete gpu resources
      // TODO(croot): reset to previously active renderer?
      makeActiveOnCurrentThread();
    }
  };
  typedef std::map<GpuId, PerGpuData> PerGpuDataMap;

  PerGpuDataMap _perGpuData;

  std::string _printInfo(bool useClassSuffix = false) const;

 private:
  friend class QueryRendererContext;
};

class SqlQueryPolyDataTable : public BaseQueryPolyDataTable, public BaseQueryDataTableSQL {
 public:
  SqlQueryPolyDataTable(const RootCacheShPtr& qrmGpuCache,
                        const std::string& tableName,
                        const std::string& sqlQueryStr = "")
      : BaseQueryPolyDataTable(qrmGpuCache, QueryDataTableType::SQLQUERY),
        BaseQueryDataTableSQL(tableName, sqlQueryStr),
        _qrmGpuCache(qrmGpuCache) {}
  virtual ~SqlQueryPolyDataTable() {}

  bool hasAttribute(const std::string& attrName) final;
  QueryBufferShPtr getAttributeDataBuffer(const GpuId& gpuId, const std::string& attrName) final;
  std::map<GpuId, QueryBufferShPtr> getAttributeDataBuffers(const std::string& attrName) final;
  QueryDataType getAttributeType(const std::string& attrName) final;

  int numRows(const GpuId& gpuId) final;

 protected:
  std::string _printInfo(bool useClassSuffix = false) const {
    std::string rtn = BaseQueryPolyDataTable::_printInfo() + ", " + BaseQueryDataTableSQL::_printInfo();

    if (useClassSuffix) {
      rtn = "SqlQueryPolyDataTable(" + rtn + ")";
    }

    return rtn;
  }

  virtual void _initGpuResources(const RootCacheShPtr& qrmGpuCache);

 private:
  std::weak_ptr<RootCache> _qrmGpuCache;

  void allocBuffers(const GpuId& gpuId,
                    const PolyTableByteData& initData,
                    const QueryDataLayoutShPtr& vertLayoutPtr = nullptr,
                    const QueryDataLayoutShPtr& uniformLayoutPtr = nullptr);
  void reset();
  PolyCudaHandles getCudaHandlesPreQuery(const GpuId& gpuId);
  void updatePostQuery(const GpuId& gpuId,
                       const QueryDataLayoutShPtr& vertLayoutPtr,
                       const QueryDataLayoutShPtr& uniformLayoutPtr);

  friend class QueryRenderManager;
};

class SqlQueryPolyDataTableJSON : public SqlQueryPolyDataTable, public BaseQueryDataTableJSON {
 public:
  SqlQueryPolyDataTableJSON(const QueryRendererContextShPtr& ctx,
                            const std::string& name,
                            const rapidjson::Value& obj,
                            const rapidjson::Pointer& objPath,
                            const std::string& tableName = "",
                            const std::string& sqlQueryStr = "");
  ~SqlQueryPolyDataTableJSON() {}

  operator std::string() const final;

 private:
  void _initGpuResources(const RootCacheShPtr& qrmGpuCache) final;
  void _initFromJSONObj(const rapidjson::Value& obj, const rapidjson::Pointer& objPath, bool forceUpdate = false);
  void _updateFromJSONObj(const rapidjson::Value& obj, const rapidjson::Pointer& objPath) final;
};

// TODO(croot): create a base class for basic data tables (i.e. DataTable/PolyDataTable)
class PolyDataTable : public BaseQueryPolyDataTable, public BaseQueryDataTableJSON {
 public:
  static std::string defaultPolyDataColumnName;

  PolyDataTable(const QueryRendererContextShPtr& ctx,
                const std::string& name,
                const rapidjson::Value& obj,
                const rapidjson::Pointer& objPath,
                QueryDataTableType type,
                bool buildIdColumn = false,
                DataTable::VboType vboType = DataTable::VboType::SEQUENTIAL);
  ~PolyDataTable();

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

  int numRows(const GpuId& gpuId) { return _numRows; }

  bool hasAttribute(const std::string& attrName) final;
  QueryBufferShPtr getAttributeDataBuffer(const GpuId& gpuId, const std::string& attrName) final;
  std::map<GpuId, QueryBufferShPtr> getAttributeDataBuffers(const std::string& attrName) final;
  QueryDataType getAttributeType(const std::string& attrName) final;

  operator std::string() const final;

 private:
  DataTable::VboType _vboType;
  int _numRows;
  int _numVerts;
  int _numPolys;
  int _numTris;
  int _numLineLoops;

  typedef boost::multi_index_container<
      DataColumnShPtr,
      boost::multi_index::indexed_by<boost::multi_index::random_access<>,

                                     // hashed on name
                                     boost::multi_index::hashed_unique<
                                         boost::multi_index::tag<DataColumn::ColumnName>,
                                         boost::multi_index::member<DataColumn, std::string, &DataColumn::columnName>>>>
      ColumnMap;

  typedef ColumnMap::index<DataColumn::ColumnName>::type ColumnMap_by_name;

  ColumnMap _columns;

  void _updateFromJSONObj(const rapidjson::Value& obj, const rapidjson::Pointer& objPath) final;

  void _buildPolyRowsFromJSONObj(const rapidjson::Value& obj);
  void _buildPolyDataFromJSONObj(const rapidjson::Value& obj, const rapidjson::Pointer& objPath, bool buildIdColumn);

  // void _populateColumnsFromJSONObj(const rapidjson::Value& obj);
  void _readDataFromFile(const std::string& filename);
  // void _readFromShapeFile(const std::string& filename);

  DataColumnShPtr _getPolyDataColumn();

  void _initBuffers(BaseQueryPolyDataTable::PerGpuData& perGpuData);
  std::tuple<::Rendering::GL::Resources::GLBufferLayoutShPtr, std::unique_ptr<char[]>, size_t> _createVBOData();
  std::tuple<::Rendering::GL::Resources::GLShaderBlockLayoutShPtr, std::unique_ptr<char[]>, size_t> _createUBOData();
  std::vector<unsigned int> _createIBOData();
  std::vector<::Rendering::GL::Resources::IndirectDrawVertexData> _createLineDrawData();
  std::vector<::Rendering::GL::Resources::IndirectDrawIndexData> _createPolyDrawData();

  void _initGpuResources(const RootCacheShPtr& qrmGpuCache) final;
};

}  // namespace QueryRenderer

#endif  // QUERYRENDERER_DATA_QUERYPOLYDATATABLE_H_

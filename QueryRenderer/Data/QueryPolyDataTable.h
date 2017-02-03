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

  ::Rendering::GL::Resources::GLVertexBufferWkPtr getGLVertexBuffer(const GpuId& gpuId) const;
  ::Rendering::GL::Resources::GLIndexBufferWkPtr getGLIndexBuffer(const GpuId& gpuId) const;
  ::Rendering::GL::Resources::GLUniformBufferWkPtr getGLUniformBuffer(const GpuId& gpuId) const;
  ::Rendering::GL::Resources::GLIndirectDrawVertexBufferWkPtr getGLIndirectDrawVertexBuffer(const GpuId& gpuId) const;
  ::Rendering::GL::Resources::GLIndirectDrawIndexBufferWkPtr getGLIndirectDrawIndexBuffer(const GpuId& gpuId) const;
  const PolyRowDataShPtr getRowDataPtr(const GpuId& gpuId) const;

  bool usesGpu(const GpuId& gpuId) const { return (_perGpuData.find(gpuId) != _perGpuData.end()); }
  std::set<GpuId> getUsedGpuIds() const final;

  PolyTableByteData getPolyBufferByteData(const GpuId& gpuId) const;
  PolyTableDataInfo getPolyBufferData(const GpuId& gpuId) const;

  BaseQueryPolyDataTable& operator=(const BaseQueryPolyDataTable& rhs);

 protected:
  struct PerGpuData : BasePerGpuData {
    QueryVertexBufferShPtr vbo;
    QueryIndexBufferShPtr ibo;
    QueryUniformBufferShPtr ubo;
    QueryIndirectVboShPtr indvbo;
    QueryIndirectIboShPtr indibo;
    PolyRowDataShPtr rowDataPtr;

    PerGpuData() : BasePerGpuData(), vbo(nullptr), ibo(nullptr), ubo(nullptr), indvbo(nullptr), indibo(nullptr) {}
    explicit PerGpuData(const RootPerGpuDataShPtr& rootData,
                        const decltype(vbo)& vbo = nullptr,
                        const decltype(ibo)& ibo = nullptr,
                        const decltype(ubo)& ubo = nullptr,
                        const decltype(indvbo)& indvbo = nullptr,
                        const decltype(indibo)& indibo = nullptr,
                        const decltype(rowDataPtr)& rowDataPtr = nullptr)
        : BasePerGpuData(rootData),
          vbo(vbo),
          ibo(ibo),
          ubo(ubo),
          indvbo(indvbo),
          indibo(indibo),
          rowDataPtr(rowDataPtr) {}
    explicit PerGpuData(const BasePerGpuData& data,
                        const decltype(vbo)& vbo = nullptr,
                        const decltype(ibo)& ibo = nullptr,
                        const decltype(ubo)& ubo = nullptr,
                        const decltype(indvbo)& indvbo = nullptr,
                        const decltype(indibo)& indibo = nullptr,
                        const decltype(rowDataPtr)& rowDataPtr = nullptr)
        : BasePerGpuData(data), vbo(vbo), ibo(ibo), ubo(ubo), indvbo(indvbo), indibo(indibo), rowDataPtr(rowDataPtr) {}
    PerGpuData(const PerGpuData& data)
        : BasePerGpuData(data),
          vbo(data.vbo),
          ibo(data.ibo),
          ubo(data.ubo),
          indvbo(data.indvbo),
          indibo(data.indibo),
          rowDataPtr(data.rowDataPtr) {}
    PerGpuData(PerGpuData&& data)
        : BasePerGpuData(std::move(data)),
          vbo(std::move(data.vbo)),
          ibo(std::move(data.ibo)),
          ubo(std::move(data.ubo)),
          indvbo(std::move(data.indvbo)),
          indibo(std::move(data.indibo)),
          rowDataPtr(std::move(data.rowDataPtr)) {}

    ~PerGpuData() {
      // need to make active to properly delete gpu resources
      // TODO(croot): reset to previously active renderer?
      makeActiveOnCurrentThread();
    }
  };
  typedef std::map<GpuId, PerGpuData> PerGpuDataMap;

  RootCacheShPtr rootGpuCache;
  PerGpuDataMap _perGpuData;

  std::string _printInfo(bool useClassSuffix = false) const;

 private:
  friend class QueryRendererContext;
  friend class SqlPolyQueryCacheMap;
};

class SqlQueryPolyDataTableCache : public BaseQueryPolyDataTable {
 public:
  SqlQueryPolyDataTableCache(const RootCacheShPtr& qrmGpuCache,
                             const std::string& polyTableName,
                             const std::string& sqlQueryStr)
      : BaseQueryPolyDataTable(qrmGpuCache, QueryDataTableType::SQLQUERY),
        _qrmGpuCache(qrmGpuCache),
        _polyTableName(polyTableName),
        _sqlQueryStr(sqlQueryStr) {}

  virtual ~SqlQueryPolyDataTableCache() {}

  bool hasAttribute(const std::string& attrName) final;
  bool hasAttributeFromLayout(const std::string& attrName,
                              const QueryDataLayoutShPtr& vertLayout = nullptr,
                              const QueryDataLayoutShPtr& uniformLayout = nullptr);
  QueryBufferWkPtr getAttributeDataBuffer(const GpuId& gpuId, const std::string& attrName) final;
  QueryBufferWkPtr getAttributeDataBufferFromLayout(const GpuId& gpuId,
                                                    const std::string& attrName,
                                                    const QueryDataLayoutShPtr& vertLayout = nullptr,
                                                    const QueryDataLayoutShPtr& uniformLayout = nullptr);
  std::map<GpuId, QueryBufferWkPtr> getAttributeDataBuffers(const std::string& attrName) final;
  std::map<GpuId, QueryBufferWkPtr> getAttributeDataBuffersFromLayout(
      const std::string& attrName,
      const QueryDataLayoutShPtr& vertLayout = nullptr,
      const QueryDataLayoutShPtr& uniformLayout = nullptr);
  QueryDataType getAttributeType(const std::string& attrName) final;
  QueryDataType getAttributeTypeFromLayout(const std::string& attrName,
                                           const QueryDataLayoutShPtr& vertLayout = nullptr,
                                           const QueryDataLayoutShPtr& uniformLayout = nullptr);
  int numRows(const GpuId& gpuId) final;

  const std::string& getPolyTableNameRef() const { return _polyTableName; }
  const std::string& getSqlQueryStrRef() const { return _sqlQueryStr; }

  std::string printInfo(bool useClassSuffix = false) const {
    std::string rtn = BaseQueryPolyDataTable::_printInfo();

    if (useClassSuffix) {
      rtn = "SqlQueryPolyDataTableCache(" + rtn + ")";
    }

    return rtn;
  }

 private:
  std::weak_ptr<RootCache> _qrmGpuCache;
  std::string _polyTableName;
  std::string _sqlQueryStr;

  void _initGpuResources(const RootCacheShPtr& qrmGpuCache) final;

  BaseQueryPolyDataTable::PerGpuData& getQueryBuffers(const GpuId& gpuId);
  BaseQueryPolyDataTable::PerGpuDataMap& getAllQueryBuffers() { return _perGpuData; }
  void reset();
  void clear();
  BaseQueryPolyDataTable::PerGpuData& setUsedGpu(const RootPerGpuDataShPtr& rootData);

#ifdef HAVE_CUDA
  PolyCudaHandles getCudaHandlesPreQuery(const GpuId& gpuId);
#endif

  void updatePostQuery(const GpuId& gpuId,
                       const QueryDataLayoutShPtr& vertLayoutPtr,
                       const QueryDataLayoutShPtr& uniformLayoutPtr,
                       const PolyRowDataShPtr& rowDataPtr);

  friend class SqlPolyQueryCacheMap;
};

class SqlQueryPolyDataTableJSON : public BaseQueryDataTable, public BaseQueryDataTableSQLJSON {
 public:
  SqlQueryPolyDataTableJSON(const QueryRendererContextShPtr& ctx,
                            const std::string& name,
                            const rapidjson::Value& obj,
                            const rapidjson::Pointer& objPath);
  ~SqlQueryPolyDataTableJSON();

  bool hasAttribute(const std::string& attrName) final;
  QueryBufferWkPtr getAttributeDataBuffer(const GpuId& gpuId, const std::string& attrName) final;
  std::map<GpuId, QueryBufferWkPtr> getAttributeDataBuffers(const std::string& attrName) final;
  QueryDataType getAttributeType(const std::string& attrName) final;
  int numRows(const GpuId& gpuId) final;
  std::set<GpuId> getUsedGpuIds() const final;

  QueryPolyDataTableShPtr getPolyCacheTable() const {
    return std::dynamic_pointer_cast<BaseQueryPolyDataTable>(_sqlCache);
  }

  operator std::string() const final;

 private:
  bool _justInitialized;
  std::string _sqlQueryStrOverride;
  std::string _polysKey;
  std::string _factsKey;
  std::string _aggExpr;
  std::string _filterExpr;
  std::string _factsTableName;
  RootCacheShPtr _rootCachePtr;
  SqlQueryPolyDataTableCacheShPtr _sqlCache;

  const std::string* _getSqlStrToUse() const;
  std::string _printInfo(bool useClassSuffix = false) const;

  bool _isInternalCacheUpToDate() final;
  void _updateFromJSONObj(const rapidjson::Value& obj,
                          const rapidjson::Pointer& objPath,
                          const bool force = false) final;
  void _runQueryAndInitResources(bool isInitializing, const rapidjson::Value* dataObj = nullptr);
  void _initGpuResources(const RootCacheShPtr& qrmPerGpuDataPtr) final;
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

  int numRows(const GpuId& gpuId) { return _numRows; }

  bool hasAttribute(const std::string& attrName) final;
  QueryBufferWkPtr getAttributeDataBuffer(const GpuId& gpuId, const std::string& attrName) final;
  std::map<GpuId, QueryBufferWkPtr> getAttributeDataBuffers(const std::string& attrName) final;
  QueryDataType getAttributeType(const std::string& attrName) final;
  DataColumnShPtr getColumn(const std::string& columnName);

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

  void _updateFromJSONObj(const rapidjson::Value& obj,
                          const rapidjson::Pointer& objPath,
                          const bool force = false) final;

  void _buildPolyRowsFromJSONObj(const rapidjson::Value& obj);
  void _buildPolyDataFromJSONObj(const rapidjson::Value& obj, const rapidjson::Pointer& objPath, bool buildIdColumn);

  void _readDataFromFile(const std::string& filename);

  DataColumnShPtr _getPolyDataColumn();

  void _initBuffers(BaseQueryPolyDataTable::PerGpuData& perGpuData);
  std::tuple<::Rendering::GL::Resources::GLBufferLayoutShPtr, std::unique_ptr<char[]>, size_t> _createVBOData();
  std::tuple<::Rendering::GL::Resources::GLShaderBlockLayoutShPtr, std::unique_ptr<char[]>, size_t> _createUBOData();
  std::vector<unsigned int> _createIBOData();
  std::vector<::Rendering::GL::Resources::IndirectDrawVertexData> _createLineDrawData();
  std::vector<::Rendering::GL::Resources::IndirectDrawIndexData> _createPolyDrawData();
  PolyRowDataShPtr _createRowData();

  void _initGpuResources(const RootCacheShPtr& qrmGpuCache) final;
};

}  // namespace QueryRenderer

#endif  // QUERYRENDERER_DATA_QUERYPOLYDATATABLE_H_

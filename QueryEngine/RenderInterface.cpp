#ifdef HAVE_RENDERING
#include "Execute.h"
#include "JsonAccessors.h"

#include "Shared/scope.h"

namespace {

::QueryRenderer::QueryDataLayout::AttrType sql_type_to_render_type(const std::string& attr, const SQLTypeInfo& ti) {
  if (ti.is_fp()) {
    return ::QueryRenderer::QueryDataLayout::AttrType::DOUBLE;
  }
  if (ti.is_integer() || ti.is_boolean() || ti.is_decimal()) {
    return ::QueryRenderer::QueryDataLayout::AttrType::INT64;
  }
  if (ti.is_string()) {
    CHECK_EQ(kENCODING_DICT, ti.get_compression());
    return ::QueryRenderer::QueryDataLayout::AttrType::INT64;
  }
  throw std::runtime_error("The attr \"" + attr + "\" has a sql type of " + ti.get_type_name() +
                           " which is not supported in render queries.");
  return ::QueryRenderer::QueryDataLayout::AttrType::INT64;
}

void set_render_widget(::QueryRenderer::QueryRenderManager* render_manager,
                       const std::string& session_id,
                       const int render_widget_id) {
  CHECK(render_manager);
  if (!render_manager->hasUserWidget(session_id, render_widget_id)) {
    render_manager->addUserWidget(session_id, render_widget_id, true);
  }
  render_manager->setActiveUserWidget(session_id, render_widget_id);
}

}  // namespace

std::string Executor::renderRows(const std::vector<std::shared_ptr<Analyzer::TargetEntry>>& targets,
                                 RenderInfo* render_info) {
  CHECK(render_info);
  render_info->in_situ_data = true;  // this function is only be called in in-situ rendering cases

  std::vector<std::string> attr_names{"key"};
  std::vector<::QueryRenderer::QueryDataLayout::AttrType> attr_types{::QueryRenderer::QueryDataLayout::AttrType::INT64};
  std::unordered_map<std::string, std::string> alias_to_name;
  std::unordered_map<std::string, uint64_t> decimal_to_scale;
  for (const auto te : targets) {
    const auto alias = te->get_resname();
    attr_names.push_back(alias);
    const auto target_expr = te->get_expr();
    const auto type_info = target_expr->get_type_info();
    attr_types.push_back(sql_type_to_render_type(alias, type_info));
    if (type_info.is_decimal()) {
      decimal_to_scale.insert(std::make_pair(alias, exp_to_scale(type_info.get_scale())));
    }
    const auto col_var = dynamic_cast<const Analyzer::ColumnVar*>(target_expr);
    if (col_var) {
      const int col_id = col_var->get_column_id();
      const auto cd = get_column_descriptor(col_id, col_var->get_table_id(), *catalog_);
      CHECK(cd);
      const auto it_ok = alias_to_name.insert(std::make_pair(alias, cd->columnName));
      CHECK(it_ok.second);
    }
  }

  // TODO(alex): make it more general, only works for projection queries
  std::shared_ptr<::QueryRenderer::QueryDataLayout> query_data_layout(
      new ::QueryRenderer::QueryDataLayout(attr_names,
                                           attr_types,
                                           alias_to_name,
                                           decimal_to_scale,
                                           ::QueryRenderer::QueryDataLayout::LayoutType::INTERLEAVED,
                                           EMPTY_KEY_64,
                                           0));

  if (!render_info->render_vega.length() || render_info->render_vega == "NONE") {
    // NOTE: this is a new render call, but we need the
    // query_data_layout for later use, so storing it here
    // TODO(croot): refactor as part of render query validation
    // effort
    render_info->vbo_result_query_data_layout = query_data_layout;
    return "";
  }

  CHECK(render_info->render_allocator_map_ptr);

  // the following function unmaps the gl buffers used by cuda
  render_info->render_allocator_map_ptr->prepForRendering(query_data_layout);

  set_render_widget(render_manager_, render_info->session_id, render_info->render_widget_id);

  std::shared_ptr<rapidjson::Document> json_doc(new rapidjson::Document());
  json_doc->Parse(render_info->render_vega.c_str());
  CHECK(!json_doc->HasParseError());

  render_manager_->configureRender(json_doc, this);

  const auto png_data = render_manager_->renderToPng(3);

  CHECK(png_data.pngDataPtr);
  CHECK(png_data.pngSize);

  return std::string(png_data.pngDataPtr.get(), png_data.pngSize);
}

int64_t Executor::getRowidForPixel(const int64_t x,
                                   const int64_t y,
                                   const std::string& session_id,
                                   const int render_widget_id,
                                   const int pixelRadius) {
  // DEPRECATED
  set_render_widget(render_manager_, session_id, render_widget_id);

  auto pixelData = render_manager_->getIdAt(x, y, pixelRadius);

  // NOTE: the table id of the above call should be -1,
  // indicating the use of the old APIs
  CHECK(std::get<0>(pixelData) == -1);

  return std::get<1>(pixelData);
}

namespace {

size_t get_rowid_idx(const std::vector<TargetMetaInfo>& row_shape) {
  for (size_t i = 0; i < row_shape.size(); ++i) {
    const auto& col_info = row_shape[i];
    if (col_info.get_resname() == "rowid") {
      const auto& col_ti = col_info.get_type_info();
      CHECK_EQ(kBIGINT, col_ti.get_type());
      return i;
    }
  }
  CHECK(false);
  return 0;
}

struct ChunkWithMetaInfo {
  const std::shared_ptr<Chunk_NS::Chunk> chunk;
  const ChunkMetadata meta;
};

ChunkWithMetaInfo get_poly_shapes_chunk(const Catalog_Namespace::Catalog& cat,
                                        const TableDescriptor* td,
                                        const std::string& col_name) {
  const auto cd = cat.getMetadataForColumn(td->tableId, col_name);
  CHECK(cd);  // TODO(alex): throw exception instead
  const auto table_info = td->fragmenter->getFragmentsForQuery();
  CHECK_EQ(size_t(1), table_info.fragments.size());  // TODO(alex): throw exception instead
  const auto& fragment = table_info.fragments.front();
  ChunkKey chunk_key{cat.get_currentDB().dbId, td->tableId, cd->columnId, fragment.fragmentId};
  auto chunk_meta_it = fragment.getChunkMetadataMap().find(cd->columnId);
  CHECK(chunk_meta_it != fragment.getChunkMetadataMap().end());
  return {Chunk_NS::Chunk::getChunk(cd,
                                    &cat.get_dataMgr(),
                                    chunk_key,
                                    Data_Namespace::CPU_LEVEL,
                                    0,
                                    chunk_meta_it->second.numBytes,
                                    chunk_meta_it->second.numElements),
          chunk_meta_it->second};
}

}  // namespace

ResultRows Executor::renderPolygons(const std::string& queryStr,
                                    const ResultRows& rows,
                                    const std::vector<TargetMetaInfo>& row_shape,
                                    const Catalog_Namespace::SessionInfo& session,
                                    const int render_widget_id,
                                    const rapidjson::Value& data_desc,
                                    const std::string* render_config_json,
                                    const bool is_projection_query,
                                    const std::string& poly_table_name,
                                    RenderInfo* render_query_data) {
#ifdef HAVE_CUDA
  // capture the lock acquistion time
  auto clock_begin = timer_start();

  std::lock_guard<std::mutex> lock(execute_mutex_);
  ScopeGuard restore_metainfo_cache = [this] { clearMetaInfoCache(); };
  int64_t queue_time_ms = timer_stop(clock_begin);
  clock_begin = timer_start();
  const auto session_id = session.get_session_id();

  std::string polyTableName = poly_table_name;
  if (data_desc.HasMember("dbTableName")) {
    polyTableName = json_str(field(data_desc, "dbTableName"));
  }
  CHECK(polyTableName.size());

  // TODO(croot): we need to reset the cache if the poly table has been modified in any way
  // NOTE: we have to be careful here about appropriately locking for multi-threads.
  // This function can be called thru two separate code paths, one is
  // QueryRenderManager::runRenderRequest(), which uses an appropriate render lock,
  // and the other is thru the old MapDHandler::render(), which also has a render lock
  // which keeps this code safe, but if either of those locks are to be removed, it
  // could leave this code vulnerable without the above lock
  bool changed = false;
  bool update = !render_manager_->hasPolyTableGpuCache(polyTableName, queryStr);
  if (update || changed) {
    const int gpuId = render_manager_->getPolyTableCacheGpuIdx(polyTableName);
    session.get_catalog().get_dataMgr().cudaMgr_->setContext(gpuId);

    std::string shape_col_group = "mapd";
    if (data_desc.HasMember("shapeColGroup")) {
      shape_col_group = json_str(field(data_desc, "shapeColGroup"));
    }

    // initialize the poly rendering data
    const auto& cat = session.get_catalog();
    const auto td = cat.getMetadataForTable(polyTableName);
    CHECK(td);  // TODO(alex): throw exception instead

    const auto row_count = rows.rowCount();

    QueryRenderer::PolyRowDataShPtr rowData(new QueryRenderer::PolyRowData(row_count));

    LineDrawData lineDrawData;
    lineDrawData.offsets.reserve(row_count);

    const auto lineloop_chunk_with_meta = get_poly_shapes_chunk(cat, td, shape_col_group + "_geo_linedrawinfo");
    CHECK(lineloop_chunk_with_meta.chunk);
    auto lineloop_chunk_iter = lineloop_chunk_with_meta.chunk->begin_iterator(lineloop_chunk_with_meta.meta);
    const auto lineloop_count = lineloop_chunk_with_meta.meta.numElements;

    CHECK(row_count <= lineloop_count);

    std::vector<::Rendering::GL::Resources::IndirectDrawIndexData> polyDrawData;
    polyDrawData.reserve(row_count);

    const auto polydraw_chunk_with_meta = get_poly_shapes_chunk(cat, td, shape_col_group + "_geo_polydrawinfo");
    CHECK(polydraw_chunk_with_meta.chunk);
    auto polydata_chunk_iter = polydraw_chunk_with_meta.chunk->begin_iterator(polydraw_chunk_with_meta.meta);
    const auto polydata_count = polydraw_chunk_with_meta.meta.numElements;

    CHECK(polydata_count == lineloop_count);

    const auto rowid_idx = get_rowid_idx(row_shape);

    auto data_query_result = getPolyRenderDataTemplate(row_shape, row_count, gpuId, rowid_idx, is_projection_query);

    size_t rowidx = 0;
    while (true) {
      const auto crt_row = rows.getNextRow(false, false);
      if (crt_row.empty()) {
        break;
      }
      const auto tv = crt_row[rowid_idx];
      const auto scalar_tv = boost::get<ScalarTargetValue>(&tv);
      const auto rowid_ptr = boost::get<int64_t>(scalar_tv);
      CHECK(rowid_ptr);

      ArrayDatum ad;
      bool is_end;
      ChunkIter_get_nth(&lineloop_chunk_iter, *rowid_ptr, &ad, &is_end);
      CHECK(!is_end);
      CHECK(ad.pointer);
      auto num_elems = ad.length / sizeof(uint32_t);  // TODO(alex): support other integer sizes as well
      CHECK_EQ(size_t(0), num_elems % 4);
      auto ui32_buff = reinterpret_cast<const uint32_t*>(ad.pointer);
      (*rowData)[rowidx].rowIdx = *rowid_ptr;
      (*rowData)[rowidx].numLineLoops = 0;
      for (size_t i = 0; i < num_elems; i += 4) {
        (*rowData)[rowidx].numLineLoops++;
        lineDrawData.data.push_back(
            ::Rendering::GL::Resources::IndirectDrawVertexData(ui32_buff[i], ui32_buff[i + 2], 1));
      }
      lineDrawData.offsets.push_back(lineDrawData.data.size());

      ChunkIter_get_nth(&polydata_chunk_iter, *rowid_ptr, &ad, &is_end);
      CHECK(!is_end);
      CHECK(ad.pointer);
      num_elems = ad.length / sizeof(uint32_t);  // TODO(alex): support other integer sizes as well
      CHECK_EQ(size_t(5), num_elems);
      ui32_buff = reinterpret_cast<const uint32_t*>(ad.pointer);
      (*rowData)[rowidx].numPolys = 1;
      polyDrawData.push_back(
          ::Rendering::GL::Resources::IndirectDrawIndexData(ui32_buff[0], ui32_buff[2], ui32_buff[3], 1));

      setPolyRenderDataEntry(data_query_result,
                             crt_row,
                             row_shape,
                             rowidx++,
                             rowid_idx,
                             data_query_result.align_bytes,
                             is_projection_query);
    }

    // build a struct specifying the number of bytes needed for each buffer used
    // for poly rendering:
    // 1) vertex buffer
    // 2) index buffer (triangles)
    // 3) line draw struct (for strokes/outlines)
    // 4) poly draw struct (for filled polys)
    // 5) extra rendering data / rowids
    ::QueryRenderer::PolyTableByteData polyByteData(
        {-1,
         -1,
         static_cast<int>(lineDrawData.data.size() * sizeof(::Rendering::GL::Resources::IndirectDrawVertexData)),
         static_cast<int>(polyDrawData.size() * sizeof(::Rendering::GL::Resources::IndirectDrawIndexData)),
         static_cast<int>(data_query_result.num_data_bytes)});

    ::QueryRenderer::PolyCudaHandles polyData;

    if (!render_manager_->hasPolyTableGpuCache(polyTableName, gpuId)) {
      // TODO(croot): if the vertices in the poly table have changed, then we
      // need to force an update and reset the vbo, ibo here.

      // setup the layout for the cached vertex buffer.
      // Since we're caching, this layout should never change, hence only necessary at
      // cache creation

      // setup the verts - 2 squares and 2 triangles
      // NOTE: the first 3 verts of each polygon are repeated at the end of
      // its vertex list in order to get all the adjacent data
      // need for the custom line-drawing shader to draw a closed line.
      const auto verts = getShapeVertices(session, td, shape_col_group, rowData);

      // setup the tri tesselation by index (must be unsigned int)
      const auto indices = getShapeIndices(session, td, shape_col_group, rowData);

      polyByteData.numVertBytes = verts.size() * sizeof(double);
      polyByteData.numIndexBytes = indices.size() * sizeof(unsigned int);

      std::shared_ptr<::QueryRenderer::QueryDataLayout> vertLayout(new ::QueryRenderer::QueryDataLayout(
          {"x", "y"},
          {::QueryRenderer::QueryDataLayout::AttrType::DOUBLE, ::QueryRenderer::QueryDataLayout::AttrType::DOUBLE},
          {{}},
          {{}}));

      // now create the cache
      render_manager_->createPolyTableCache(polyTableName, queryStr, gpuId, polyByteData, vertLayout, rowData);

      if (render_query_data) {
        render_query_data->in_situ_data = false;
        render_query_data->vbo_result_query_data_layout = vertLayout;
      }

      // get cuda handles for each of the 5 buffers for poly rendering
      polyData = render_manager_->getPolyTableCudaHandles(polyTableName, queryStr, gpuId);

      // using simple cuda driver calls to push data to buffers
      cuMemcpyHtoD(reinterpret_cast<CUdeviceptr>(polyData.verts.handle), &verts[0], polyByteData.numVertBytes);
      cuMemcpyHtoD(reinterpret_cast<CUdeviceptr>(polyData.polyIndices.handle), &indices[0], polyByteData.numIndexBytes);
    } else {
      if (changed) {
        auto cachedByteData = render_manager_->getPolyTableCacheByteInfo(polyTableName, queryStr, gpuId);
        if (cachedByteData.numLineLoopBytes != polyByteData.numLineLoopBytes ||
            cachedByteData.numPolyBytes != polyByteData.numPolyBytes ||
            cachedByteData.numDataBytes != polyByteData.numDataBytes) {
          // TODO(croot): improve this API
          polyByteData.numVertBytes = cachedByteData.numVertBytes;
          polyByteData.numIndexBytes = cachedByteData.numIndexBytes;
        }
      }

      if (update || changed) {
        render_manager_->updatePolyTableCache(polyTableName, queryStr, gpuId, polyByteData, nullptr, rowData);
      }

      // get cuda handles for each of the 5 buffers for poly rendering
      polyData = render_manager_->getPolyTableCudaHandles(polyTableName, queryStr, gpuId);
    }

    if (!lineDrawData.data.empty()) {
      cuMemcpyHtoD(reinterpret_cast<CUdeviceptr>(polyData.lineDrawStruct.handle),
                   &lineDrawData.data[0],
                   polyByteData.numLineLoopBytes);
    }
    cuMemcpyHtoD(
        reinterpret_cast<CUdeviceptr>(polyData.polyDrawStruct.handle), &polyDrawData[0], polyByteData.numPolyBytes);
    cuMemcpyHtoD(reinterpret_cast<CUdeviceptr>(polyData.perRowData.handle),
                 data_query_result.data.get(),
                 polyByteData.numDataBytes);

    // set the buffers as renderable
    render_manager_->setPolyQueryReadyForRender(
        polyTableName, queryStr, gpuId, data_query_result.poly_render_data_layout);

    if (render_query_data) {
      render_query_data->in_situ_data = false;
      render_query_data->ubo_result_query_data_layout = data_query_result.poly_render_data_layout;
    }
  }

  if (!render_config_json || !render_config_json->length() || *render_config_json == "NONE") {
    int64_t render_time_ms = timer_stop(clock_begin);
    return ResultRows(std::string(), queue_time_ms, render_time_ms, row_set_mem_owner_);
  }

  // now configure the render like normal
  std::shared_ptr<rapidjson::Document> json_doc(new rapidjson::Document());
  json_doc->Parse(render_config_json->c_str());
  CHECK(!json_doc->HasParseError());
  CHECK(json_doc->IsObject());

  // set the session, like normal
  set_render_widget(render_manager_, session_id, render_widget_id);

  render_manager_->configureRender(json_doc, this);

  // now perform the render
  const auto png_data = render_manager_->renderToPng(3);

  int64_t render_time_ms = timer_stop(clock_begin);

  return ResultRows(std::string(png_data.pngDataPtr.get(), png_data.pngSize), queue_time_ms, render_time_ms);
#else
  LOG(ERROR) << "Cannot run testRenderSimplePolys() without cuda enabled";
  return ResultRows(std::string(), 0, 0);
#endif  // HAVE_CUDA
}

std::vector<double> Executor::getShapeVertices(const Catalog_Namespace::SessionInfo& session,
                                               const TableDescriptor* td,
                                               const std::string& shape_col_group,
                                               QueryRenderer::PolyRowDataShPtr& rowDataPtr) {
  const auto& cat = session.get_catalog();
  const auto chunk_with_meta = get_poly_shapes_chunk(cat, td, shape_col_group + "_geo_coords");
  CHECK(chunk_with_meta.chunk);
  auto chunk_iter = chunk_with_meta.chunk->begin_iterator(chunk_with_meta.meta);
  const auto row_count = chunk_with_meta.meta.numElements;

  std::vector<double> geo_coords;
  size_t idx = 0;
  for (size_t row_pos = 0; row_pos < row_count; ++row_pos) {
    ArrayDatum ad;
    bool is_end;
    ChunkIter_get_nth(&chunk_iter, row_pos, &ad, &is_end);
    CHECK(!is_end);
    CHECK(ad.pointer);
    const auto num_elems = ad.length / sizeof(double);  // TODO(alex): support float as well
    if (idx < rowDataPtr->size() && row_pos == (*rowDataPtr)[idx].rowIdx) {
      (*rowDataPtr)[idx].numVerts = num_elems / 2;  // Verts are packed, [x0,y0,x1,y1,...]
      (*rowDataPtr)[idx].startVertIdx = geo_coords.size() / 2;
      idx++;
    }
    const auto double_buff = reinterpret_cast<const double*>(ad.pointer);
    for (size_t i = 0; i < num_elems; ++i) {
      geo_coords.push_back(double_buff[i]);
    }
  }
  return geo_coords;
}

std::vector<unsigned> Executor::getShapeIndices(const Catalog_Namespace::SessionInfo& session,
                                                const TableDescriptor* td,
                                                const std::string& shape_col_group,
                                                QueryRenderer::PolyRowDataShPtr& rowDataPtr) {
  const auto& cat = session.get_catalog();
  const auto chunk_with_meta = get_poly_shapes_chunk(cat, td, shape_col_group + "_geo_indices");
  CHECK(chunk_with_meta.chunk);
  auto chunk_iter = chunk_with_meta.chunk->begin_iterator(chunk_with_meta.meta);
  const auto row_count = chunk_with_meta.meta.numElements;

  std::vector<unsigned> geo_indices;
  size_t idx = 0;
  for (size_t row_pos = 0; row_pos < row_count; ++row_pos) {
    ArrayDatum ad;
    bool is_end;
    ChunkIter_get_nth(&chunk_iter, row_pos, &ad, &is_end);
    CHECK(!is_end);
    CHECK(ad.pointer);
    const auto num_elems = ad.length / sizeof(uint32_t);  // TODO(alex): support other integer sizes as well
    if (idx < rowDataPtr->size() && row_pos == (*rowDataPtr)[idx].rowIdx) {
      (*rowDataPtr)[idx].numIndices = num_elems;
      (*rowDataPtr)[idx].startIndIdx = geo_indices.size();
      idx++;
    }
    const auto ui32_buff = reinterpret_cast<const uint32_t*>(ad.pointer);
    for (size_t i = 0; i < num_elems; ++i) {
      geo_indices.push_back(ui32_buff[i]);
    }
  }
  return geo_indices;
}

namespace {

size_t get_data_row_size(const std::vector<TargetMetaInfo>& row_shape) {
  size_t sz = 0;
  for (const auto& target_meta : row_shape) {
    const auto& target_ti = target_meta.get_type_info();
    sz += target_ti.get_logical_size();
  }
  return sz;
}

}  // namespace

// TODO(alex): We can cache this template based on the row shape.
Executor::PolyRenderDataQueryResult Executor::getPolyRenderDataTemplate(const std::vector<TargetMetaInfo>& row_shape,
                                                                        const size_t entry_count,
                                                                        const size_t gpuId,
                                                                        const size_t rowid_idx,
                                                                        const bool is_projection_query) {
  // the rendering/rowid data is put in a special "uniform" buffer. This buffer
  // has specific byte alignment rules. As long as we're dealing with
  // basic types here (and not arrays or structs), then the only rule we need
  // to worry about is the padding/byte alignment. The number of bytes per row must
  // be a multiple of the alignBytes below.
  size_t align_bytes = render_manager_->getPolyDataBufferAlignmentBytes(gpuId);

  const auto row_size = get_data_row_size(row_shape);
  CHECK_LE(row_size, align_bytes);

  size_t num_data_bytes = entry_count * align_bytes;

  // setting the rendering/rowid data. Tightly packed at the front of each row
  // with extra padding at end to align with alignBytes.
  auto raw_data = new char[num_data_bytes];
  std::memset(raw_data, 0, num_data_bytes);

  std::vector<std::string> attr_names;
  std::vector<::QueryRenderer::QueryDataLayout::AttrType> attr_types;
  std::unordered_map<std::string, uint64_t> decimal_to_scale;
  for (size_t i = 0; i < row_shape.size(); ++i) {
    const auto& target_meta_info = row_shape[i];
    auto alias = target_meta_info.get_resname();
    attr_names.push_back(alias);
    if (!is_projection_query && i == rowid_idx) {
      // Adding the row index as our rowid in these cases, not the rowid of
      // the actual poly

      // TODO(croot): can we ever have more than max(size_t) rows?
      attr_types.push_back(::QueryRenderer::QueryDataLayout::AttrType::UINT64);
    } else {
      const auto type_info = target_meta_info.get_type_info();
      attr_types.push_back(sql_type_to_render_type(alias, type_info));
      if (type_info.is_decimal()) {
        decimal_to_scale.insert(std::make_pair(alias, exp_to_scale(type_info.get_scale())));
      }
    }
  }

  auto query_data_layout = new ::QueryRenderer::QueryDataLayout(attr_names, attr_types, {{}}, decimal_to_scale);

  return {std::shared_ptr<::QueryRenderer::QueryDataLayout>(query_data_layout),
          std::unique_ptr<char[]>(raw_data),
          num_data_bytes,
          align_bytes};
}

void Executor::setPolyRenderDataEntry(Executor::PolyRenderDataQueryResult& render_data,
                                      const std::vector<TargetValue>& row,
                                      const std::vector<TargetMetaInfo>& row_shape,
                                      const size_t rowidx,
                                      const size_t rowid_idx,
                                      const size_t align_bytes,
                                      const bool is_projection_query) {
  CHECK_EQ(row.size(), row_shape.size());
  auto startoffset = rowidx * align_bytes;
  auto offset = startoffset;
  for (size_t col_idx = 0; col_idx < row_shape.size(); ++col_idx) {
    const auto tv = row[col_idx];
    const auto scalar_tv = boost::get<ScalarTargetValue>(&tv);
    const auto i64_ptr = boost::get<int64_t>(scalar_tv);

    // TODO(croot): get attr offset per row from the layout
    // and use here
    if (i64_ptr) {
      if (!is_projection_query && col_idx == rowid_idx) {
        std::memcpy(render_data.data.get() + offset, &rowidx, sizeof(decltype(rowidx)));
        offset += sizeof(decltype(rowidx));
      } else {
        std::memcpy(render_data.data.get() + offset, i64_ptr, sizeof(int64_t));
        offset += sizeof(int64_t);
      }
      continue;
    }
    const auto float_ptr = boost::get<float>(scalar_tv);
    if (float_ptr) {
      const double dval = *float_ptr;  // TODO(alex): remove this conversion
      std::memcpy(render_data.data.get() + offset, &dval, sizeof(double));
      offset += sizeof(double);
      continue;
    }
    const auto double_ptr = boost::get<double>(scalar_tv);
    CHECK(double_ptr);
    std::memcpy(render_data.data.get() + offset, double_ptr, sizeof(double));
    offset += sizeof(double);
  }

  CHECK(offset - startoffset < align_bytes);
}

int32_t Executor::getStringId(const std::string& table_name,
                              const std::string& col_name,
                              const std::string& col_val,
                              const ::QueryRenderer::QueryDataLayout* query_data_layout,
                              const ResultRows* results) const {
  const auto td = catalog_->getMetadataForTable(table_name);
  CHECK(td);
  CHECK(query_data_layout);
  const auto col_real_name_it = query_data_layout->attrAliasToName.find(col_name);
  CHECK(col_real_name_it != query_data_layout->attrAliasToName.end());
  const auto cd = catalog_->getMetadataForColumn(td->tableId, col_real_name_it->second);
  CHECK(cd);
  CHECK(cd->columnType.is_string() && cd->columnType.get_compression() == kENCODING_DICT);
  const int dict_id = cd->columnType.get_comp_param();
  std::shared_ptr<RowSetMemoryOwner> row_set_mem_owner;
  if (results) {
    row_set_mem_owner = results->getRowSetMemOwner();
  }

  if (!row_set_mem_owner) {
    row_set_mem_owner = row_set_mem_owner_;
  }

  auto sdp = getStringDictionaryProxy(dict_id, row_set_mem_owner, false);
  CHECK(sdp);
  return sdp->getIdOfStringNoGeneration(col_val);
}

#endif  // HAVE_RENDERING

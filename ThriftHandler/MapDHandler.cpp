/*
 * cool mapd license
 */

/*
 * File:   MapDHandler.cpp
 * Author: michael
 *
 * Created on Jan 1, 2017, 12:40 PM
 */

#include "MapDHandler.h"
#include "DistributedLoader.h"
#include "MapDServer.h"
#ifdef HAVE_PROFILER
#include <gperftools/heap-profiler.h>
#endif  // HAVE_PROFILER
#include <thrift/concurrency/ThreadManager.h>
#include <thrift/concurrency/PlatformThreadFactory.h>
#include <thrift/protocol/TBinaryProtocol.h>
#include <thrift/protocol/TJSONProtocol.h>
#include <thrift/server/TThreadPoolServer.h>
#include <thrift/server/TThreadedServer.h>
#include <thrift/transport/THttpServer.h>
#include <thrift/transport/TServerSocket.h>
#include <thrift/transport/TBufferTransports.h>

#include "MapDRelease.h"

#ifdef HAVE_CALCITE
#include "Calcite/Calcite.h"
#endif  // HAVE_CALCITE

#ifdef HAVE_RAVM
#include "QueryEngine/PendingExecutionClosure.h"
#include "QueryEngine/RelAlgExecutor.h"
#endif  // HAVE_RAVM

#include "Catalog/Catalog.h"
#include "Fragmenter/InsertOrderFragmenter.h"
#include "Import/Importer.h"
#include "Parser/parser.h"
#include "Parser/ParserWrapper.h"
#include "Parser/ReservedKeywords.h"
#include "Planner/Planner.h"
#include "QueryEngine/CalciteAdapter.h"
#include "QueryEngine/Execute.h"
#include "QueryEngine/GpuMemUtils.h"
#include "QueryEngine/ExtensionFunctionsWhitelist.h"
#include "QueryEngine/JsonAccessors.h"
#include "Shared/geosupport.h"
#include "Shared/mapd_shared_mutex.h"
#include "Shared/measure.h"
#include "Shared/scope.h"
#include "Shared/StringTransform.h"
#include "Shared/MapDParameters.h"

#include <boost/algorithm/string.hpp>
#include <boost/algorithm/string/replace.hpp>
#include <boost/algorithm/string/trim.hpp>
#include <boost/filesystem.hpp>
#include <boost/make_shared.hpp>
#include <boost/program_options.hpp>
#include <boost/regex.hpp>
#include <boost/tokenizer.hpp>
#include <future>
#include <memory>
#include <string>
#include <fstream>
#include <sys/time.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <random>
#include <map>
#include <cmath>
#include <typeinfo>
#include <thread>
#include <glog/logging.h>
#include <signal.h>
#include <unistd.h>
#include <fcntl.h>
#include <regex>

#ifdef HAVE_RENDERING
#include "QueryRenderer/QueryRenderManager.h"
#endif  // HAVE_RENDERING

#define INVALID_SESSION_ID -1

std::shared_ptr<const rapidjson::Value> get_poly_render_data(rapidjson::Document& render_config) {
  // DEPRECATED, can be removed when MapDHandler::render() is removed
  const auto& data_descs = field(render_config, "data");
  CHECK(data_descs.IsArray());
  CHECK_EQ(unsigned(1), data_descs.Size());
  auto& data_desc = *(data_descs.Begin());
  if (data_desc.HasMember("format")) {
    CHECK_EQ("polys", json_str(field(data_desc, "format")));
    std::shared_ptr<rapidjson::Value> data_ptr(new rapidjson::Value);
    rapidjson::Document::AllocatorType& a = render_config.GetAllocator();
    data_ptr->CopyFrom(data_desc, a);
    return data_ptr;
  }
  return nullptr;
}

std::string build_poly_render_query(const rapidjson::Value& data_desc) {
  // DEPRECATED, can be removed when MapDHandler::render() is removed
  CHECK_EQ("polys", json_str(field(data_desc, "format")));
  const auto polyTableName = json_str(field(data_desc, "dbTableName"));
  const auto factsTableName = json_str(field(data_desc, "factsTableName"));
  const auto filterExpr = json_str(field(data_desc, "filterExpr"));
  const auto aggExpr = json_str(field(data_desc, "aggExpr"));
  const auto factsKey = json_str(field(data_desc, "factsKey"));
  const auto polysKey = json_str(field(data_desc, "polysKey"));
  return "SELECT " + polyTableName + ".rowid, " + aggExpr + " FROM " + factsTableName + ", " + polyTableName +
         " WHERE " + filterExpr + (filterExpr.empty() ? "" : " AND ") + factsTableName + "." + factsKey + " = " +
         polyTableName + "." + polysKey + " GROUP BY " + polyTableName + ".rowid;";
}

std::string transform_to_poly_render_query(const std::string& query_str, const rapidjson::Value& data_desc) {
  // DEPRECATED, can be removed when MapDHandler::render() is removed
  CHECK_EQ("polys", json_str(field(data_desc, "format")));
  auto result = query_str;
  {
    boost::regex aliased_group_expr{R"(\s+([^\s]+)\s+as\s+([^(\s|,)]+))", boost::regex::extended | boost::regex::icase};
    boost::smatch what;
    std::string what1, what2;
    if (boost::regex_search(result, what, aliased_group_expr)) {
      what1 = std::string(what[1]);
      what2 = std::string(what[2]);
      result.replace(what.position(), what.length(), " " + what1);
    } else {
      what1 = std::string(what[1]);
      what2 = std::string(what[2]);
    }
    boost::ireplace_all(result, what2, what1);
  }
  const auto polyTableName = json_str(field(data_desc, "dbTableName"));
  const auto polysKey = json_str(field(data_desc, "polysKey"));
  std::string groupby_expr;
  {
    boost::regex group_expr{R"(group\s+by\s+([^(\s|;|,)]+))", boost::regex::extended | boost::regex::icase};
    boost::smatch what;
    CHECK(boost::regex_search(result, what, group_expr));
    groupby_expr = what[1];
    boost::ireplace_all(result, std::string(what[1]), polyTableName + ".rowid");
  }
  CHECK(!groupby_expr.empty());
  const auto join_filter = groupby_expr + " = " + polyTableName + "." + polysKey;
  {
    boost::regex where_expr(R"(\s+where\s+(.*)\s+group\s+by)", boost::regex::extended | boost::regex::icase);
    boost::smatch what_where;
    boost::regex from_expr{R"(\s+from\s+([^\s]+)\s+)", boost::regex::extended | boost::regex::icase};
    boost::smatch what_from;
    if (boost::regex_search(result, what_where, where_expr)) {
      result.replace(
          what_where.position(), what_where.length(), " WHERE " + what_where[1] + " AND " + join_filter + " GROUP BY");
      CHECK(boost::regex_search(result, what_from, from_expr));
      result.replace(
          what_from.position(), what_from.length(), " FROM " + std::string(what_from[1]) + ", " + polyTableName + " ");
    } else {
      CHECK(boost::regex_search(result, what_from, from_expr));
      result.replace(what_from.position(),
                     what_from.length(),
                     " FROM " + std::string(what_from[1]) + ", " + polyTableName + " WHERE " + join_filter + " ");
    }
  }
  return result;
}

#ifdef HAVE_RENDERING
bool is_poly_table(const decltype(TableDescriptor::tableId) table_id, const Catalog_Namespace::Catalog& cat) {
  auto table_desc = cat.getMetadataForTable(table_id);
  if (!table_desc) {
    return false;
  }

  return (cat.getMetadataForColumn(table_id, "mapd_geo_coords") &&
          cat.getMetadataForColumn(table_id, "mapd_geo_indices") &&
          cat.getMetadataForColumn(table_id, "mapd_geo_linedrawinfo") &&
          cat.getMetadataForColumn(table_id, "mapd_geo_polydrawinfo"));
}
#endif  // HAVE_RENDERING

std::string image_from_rendered_rows(const ResultRows& rendered_results) {
  const auto img_row = rendered_results.getNextRow(false, false);
  CHECK_EQ(size_t(1), img_row.size());
  const auto& img_tv = img_row.front();
  const auto scalar_tv = boost::get<ScalarTargetValue>(&img_tv);
  const auto nullable_sptr = boost::get<NullableString>(scalar_tv);
  CHECK(nullable_sptr);
  auto sptr = boost::get<std::string>(nullable_sptr);
  CHECK(sptr);
  return *sptr;
}

MapDHandler::MapDHandler(const std::vector<LeafHostInfo>& db_leaves,
                         const std::vector<LeafHostInfo>& string_leaves,
                         const std::string& base_data_path,
                         const std::string& executor_device,
                         const bool allow_multifrag,
                         const bool jit_debug,
                         const bool read_only,
                         const bool allow_loop_joins,
                         const bool enable_rendering,
                         const size_t cpu_buffer_mem_bytes,
                         const size_t render_mem_bytes,
                         const int num_gpus,
                         const int start_gpu,
                         const size_t reserved_gpu_mem,
                         const size_t num_reader_threads,
                         const int start_epoch,
                         const LdapMetadata ldapMetadata,
                         const MapDParameters& mapd_parameters,
                         const std::string& db_convert_dir,
#ifdef HAVE_CALCITE
                         const int calcite_port,
                         const bool legacy_syntax)
#else
                         const int /* calcite_port */,
                         const bool /* legacy_syntax */)
#endif  // HAVE_CALCITE
    : leaf_aggregator_(db_leaves),
      string_leaves_(string_leaves),
      base_data_path_(base_data_path),
      random_gen_(std::random_device{}()),
      session_id_dist_(0, INT32_MAX),
      jit_debug_(jit_debug),
      allow_multifrag_(allow_multifrag),
      read_only_(read_only),
      allow_loop_joins_(allow_loop_joins),
      mapd_parameters_(mapd_parameters),
#ifdef HAVE_CALCITE
      enable_rendering_(enable_rendering),
      legacy_syntax_(legacy_syntax) {
#else
      enable_rendering_(enable_rendering) {
#endif  // HAVE_CALCITE
  LOG(INFO) << "MapD Server " << MapDRelease;
  if (executor_device == "gpu") {
#ifdef HAVE_CUDA
    executor_device_type_ = ExecutorDeviceType::GPU;
    cpu_mode_only_ = false;
#else
    executor_device_type_ = ExecutorDeviceType::CPU;
    LOG(ERROR) << "This build isn't CUDA enabled, will run on CPU";
    cpu_mode_only_ = true;
#endif  // HAVE_CUDA
  } else if (executor_device == "hybrid") {
    executor_device_type_ = ExecutorDeviceType::Hybrid;
    cpu_mode_only_ = false;
  } else {
    executor_device_type_ = ExecutorDeviceType::CPU;
    cpu_mode_only_ = true;
  }
  const auto data_path = boost::filesystem::path(base_data_path_) / "mapd_data";
  // calculate the total amount of memory we need to reserve from each gpu that the Buffer manage cannot ask for
  size_t total_reserved = reserved_gpu_mem;
  if (enable_rendering_) {
    total_reserved += render_mem_bytes;
  }
  data_mgr_.reset(new Data_Namespace::DataMgr(data_path.string(),
                                              cpu_buffer_mem_bytes,
                                              !cpu_mode_only_,
                                              num_gpus,
                                              db_convert_dir,
                                              start_gpu,
                                              total_reserved,
                                              start_epoch,
                                              num_reader_threads));
#ifdef HAVE_CALCITE
  calcite_.reset(new Calcite(calcite_port, base_data_path_, mapd_parameters_.calcite_max_mem));
#ifdef HAVE_RAVM
  ExtensionFunctionsWhitelist::add(calcite_->getExtensionFunctionWhitelist());
#endif  // HAVE_RAVM
#endif  // HAVE_CALCITE

  if (!data_mgr_->gpusPresent()) {
    executor_device_type_ = ExecutorDeviceType::CPU;
    LOG(ERROR) << "No GPUs detected, falling back to CPU mode";
    cpu_mode_only_ = true;
  }

  switch (executor_device_type_) {
    case ExecutorDeviceType::GPU:
      LOG(INFO) << "Started in GPU mode" << std::endl;
      break;
    case ExecutorDeviceType::CPU:
      LOG(INFO) << "Started in CPU mode" << std::endl;
      break;
    case ExecutorDeviceType::Hybrid:
      LOG(INFO) << "Started in Hybrid mode" << std::endl;
  }

#ifdef HAVE_RENDERING
  // TODO(croot): remove the cpu_mode_only_ rendering restriction
  // when cpu-mode with hardware accelerated rendering
  // or cpu-mode with software rendering is supported.
  if (enable_rendering_ && !cpu_mode_only_) {
    try {
      render_manager_.reset(
          new ::QueryRenderer::QueryRenderManager(data_mgr_->cudaMgr_, num_gpus, start_gpu, render_mem_bytes, 500));
    } catch (const std::exception& e) {
      enable_rendering_ = false;
      LOG(ERROR) << "Backend rendering disabled: " << e.what();
    }
  } else if (enable_rendering_ && cpu_mode_only_) {
    enable_rendering_ = false;
    LOG(ERROR) << "Backend rendering disabled: running in CPU mode";
  }
#endif  // HAVE_RENDERING

  sys_cat_.reset(new Catalog_Namespace::SysCatalog(base_data_path_,
                                                   data_mgr_,
                                                   ldapMetadata
#ifdef HAVE_CALCITE
                                                   ,
                                                   calcite_
#endif  // HAVE_CALCITE
                                                   ));
  import_path_ = boost::filesystem::path(base_data_path_) / "mapd_import";
  start_time_ = std::time(nullptr);
}

MapDHandler::~MapDHandler() {
  LOG(INFO) << "mapd_server exits." << std::endl;
}

void MapDHandler::check_read_only(const std::string& str) {
  if (MapDHandler::read_only_) {
    TMapDException ex;
    ex.error_msg = str + " disabled: server running in read-only mode.";
    LOG(ERROR) << ex.error_msg;
    throw ex;
  }
}

TSessionId MapDHandler::connect(const std::string& user, const std::string& passwd, const std::string& dbname) {
  mapd_lock_guard<mapd_shared_mutex> write_lock(sessions_mutex_);
  Catalog_Namespace::UserMetadata user_meta;
  if (!sys_cat_->getMetadataForUser(user, user_meta)) {
    TMapDException ex;
    ex.error_msg = std::string("User ") + user + " does not exist.";
    LOG(ERROR) << ex.error_msg;
    throw ex;
  }
  if (!sys_cat_->checkPasswordForUser(passwd, user_meta)) {
    TMapDException ex;
    ex.error_msg = std::string("Password for User ") + user + " is incorrect.";
    LOG(ERROR) << ex.error_msg;
    throw ex;
  }
  Catalog_Namespace::DBMetadata db_meta;
  if (!sys_cat_->getMetadataForDB(dbname, db_meta)) {
    TMapDException ex;
    ex.error_msg = std::string("Database ") + dbname + " does not exist.";
    LOG(ERROR) << ex.error_msg;
    throw ex;
  }
  // insert privilege is being treated as access allowed for now
  Privileges privs;
  privs.insert_ = true;
  privs.select_ = false;
  if (!sys_cat_->checkPrivileges(user_meta, db_meta, privs)) {
    TMapDException ex;
    ex.error_msg = std::string("User ") + user + " is not authorized to access database " + dbname;
    LOG(ERROR) << ex.error_msg;
    throw ex;
  }
  TSessionId session = INVALID_SESSION_ID;
  while (true) {
    session = session_id_dist_(random_gen_);
    auto session_it = sessions_.find(session);
    if (session_it == sessions_.end())
      break;
  }
  auto cat_it = cat_map_.find(dbname);
  if (cat_it == cat_map_.end()) {
    Catalog_Namespace::Catalog* cat = new Catalog_Namespace::Catalog(base_data_path_,
                                                                     db_meta,
                                                                     data_mgr_
#ifdef HAVE_CALCITE
                                                                     ,
                                                                     string_leaves_,
                                                                     calcite_
#endif  // HAVE_CALCITE
                                                                     );
    cat_map_[dbname].reset(cat);
    sessions_[session].reset(
        new Catalog_Namespace::SessionInfo(cat_map_[dbname], user_meta, executor_device_type_, session));
  } else
    sessions_[session].reset(
        new Catalog_Namespace::SessionInfo(cat_it->second, user_meta, executor_device_type_, session));
  if (leaf_aggregator_.leafCount() > 0) {
    const auto parent_session_info_ptr = sessions_[session];
    CHECK(parent_session_info_ptr);
    leaf_aggregator_.connect(*parent_session_info_ptr, user, passwd, dbname);
    return session;
  }
  LOG(INFO) << "User " << user << " connected to database " << dbname << std::endl;
  return session;
}

void MapDHandler::disconnect(const TSessionId session) {
  mapd_lock_guard<mapd_shared_mutex> write_lock(sessions_mutex_);
  if (leaf_aggregator_.leafCount() > 0) {
    leaf_aggregator_.disconnect(session);
  }
  auto session_it = MapDHandler::get_session_it(session);
  const auto dbname = session_it->second->get_catalog().get_currentDB().dbName;
  LOG(INFO) << "User " << session_it->second->get_currentUser().userName << " disconnected from database " << dbname
            << std::endl;
  sessions_.erase(session_it);
}

void MapDHandler::interrupt(const TSessionId session) {
  if (g_enable_dynamic_watchdog) {
    mapd_lock_guard<mapd_shared_mutex> read_lock(sessions_mutex_);
    if (leaf_aggregator_.leafCount() > 0) {
      leaf_aggregator_.interrupt(session);
    }
    auto session_it = get_session_it(session);
    const auto dbname = session_it->second->get_catalog().get_currentDB().dbName;
    auto session_info_ptr = session_it->second.get();
    auto& cat = session_info_ptr->get_catalog();
    auto executor = Executor::getExecutor(
        cat.get_currentDB().dbId, jit_debug_ ? "/tmp" : "", jit_debug_ ? "mapdquery" : "", mapd_parameters_, nullptr);
    CHECK(executor);

    VLOG(1) << "Received interrupt: "
            << "Session " << session << ", Executor " << executor << ", leafCount " << leaf_aggregator_.leafCount()
            << ", User " << session_it->second->get_currentUser().userName << ", Database " << dbname << std::endl;

    executor->interrupt();

    LOG(INFO) << "User " << session_it->second->get_currentUser().userName << " interrupted session with database "
              << dbname << std::endl;
  }
}

void MapDHandler::get_server_status(TServerStatus& _return, const TSessionId session) {
  _return.read_only = read_only_;
  _return.version = MapDRelease;
  _return.rendering_enabled = enable_rendering_;
  _return.start_time = start_time_;
}

void MapDHandler::value_to_thrift_column(const TargetValue& tv, const SQLTypeInfo& ti, TColumn& column) {
  const auto scalar_tv = boost::get<ScalarTargetValue>(&tv);
  if (!scalar_tv) {
    const auto list_tv = boost::get<std::vector<ScalarTargetValue>>(&tv);
    CHECK(list_tv);
    CHECK(ti.is_array());
    TColumn tColumn;
    for (const auto& elem_tv : *list_tv) {
      value_to_thrift_column(elem_tv, ti.get_elem_type(), tColumn);
    }
    column.data.arr_col.push_back(tColumn);
    column.nulls.push_back(list_tv->size() == 0);
  } else {
    if (boost::get<int64_t>(scalar_tv)) {
      int64_t data = *(boost::get<int64_t>(scalar_tv));
      column.data.int_col.push_back(data);
      switch (ti.get_type()) {
        case kBOOLEAN:
          column.nulls.push_back(data == NULL_BOOLEAN);
          break;
        case kSMALLINT:
          column.nulls.push_back(data == NULL_SMALLINT);
          break;
        case kINT:
          column.nulls.push_back(data == NULL_INT);
          break;
        case kBIGINT:
          column.nulls.push_back(data == NULL_BIGINT);
          break;
        case kTIME:
        case kTIMESTAMP:
        case kDATE:
        case kINTERVAL_DAY_TIME:
        case kINTERVAL_YEAR_MONTH:
          if (sizeof(time_t) == 4)
            column.nulls.push_back(data == NULL_INT);
          else
            column.nulls.push_back(data == NULL_BIGINT);
          break;
        default:
          column.nulls.push_back(false);
      }
    } else if (boost::get<double>(scalar_tv)) {
      double data = *(boost::get<double>(scalar_tv));
      column.data.real_col.push_back(data);
      if (ti.get_type() == kFLOAT) {
        column.nulls.push_back(data == NULL_FLOAT);
      } else {
        column.nulls.push_back(data == NULL_DOUBLE);
      }
    } else if (boost::get<float>(scalar_tv)) {
      CHECK_EQ(kFLOAT, ti.get_type());
      float data = *(boost::get<float>(scalar_tv));
      column.data.real_col.push_back(data);
      column.nulls.push_back(data == NULL_FLOAT);
    } else if (boost::get<NullableString>(scalar_tv)) {
      auto s_n = boost::get<NullableString>(scalar_tv);
      auto s = boost::get<std::string>(s_n);
      if (s) {
        column.data.str_col.push_back(*s);
      } else {
        column.data.str_col.push_back("");  // null string
        auto null_p = boost::get<void*>(s_n);
        CHECK(null_p && !*null_p);
      }
      column.nulls.push_back(!s);
    } else {
      CHECK(false);
    }
  }
}

TDatum MapDHandler::value_to_thrift(const TargetValue& tv, const SQLTypeInfo& ti) {
  TDatum datum;
  const auto scalar_tv = boost::get<ScalarTargetValue>(&tv);
  if (!scalar_tv) {
    const auto list_tv = boost::get<std::vector<ScalarTargetValue>>(&tv);
    CHECK(list_tv);
    CHECK(ti.is_array());
    for (const auto& elem_tv : *list_tv) {
      const auto scalar_col_val = value_to_thrift(elem_tv, ti.get_elem_type());
      datum.val.arr_val.push_back(scalar_col_val);
    }
    datum.is_null = datum.val.arr_val.empty();
    return datum;
  }
  if (boost::get<int64_t>(scalar_tv)) {
    datum.val.int_val = *(boost::get<int64_t>(scalar_tv));
    switch (ti.get_type()) {
      case kBOOLEAN:
        datum.is_null = (datum.val.int_val == NULL_BOOLEAN);
        break;
      case kSMALLINT:
        datum.is_null = (datum.val.int_val == NULL_SMALLINT);
        break;
      case kINT:
        datum.is_null = (datum.val.int_val == NULL_INT);
        break;
      case kBIGINT:
        datum.is_null = (datum.val.int_val == NULL_BIGINT);
        break;
      case kTIME:
      case kTIMESTAMP:
      case kDATE:
      case kINTERVAL_DAY_TIME:
      case kINTERVAL_YEAR_MONTH:
        if (sizeof(time_t) == 4)
          datum.is_null = (datum.val.int_val == NULL_INT);
        else
          datum.is_null = (datum.val.int_val == NULL_BIGINT);
        break;
      default:
        datum.is_null = false;
    }
  } else if (boost::get<double>(scalar_tv)) {
    datum.val.real_val = *(boost::get<double>(scalar_tv));
    if (ti.get_type() == kFLOAT) {
      datum.is_null = (datum.val.real_val == NULL_FLOAT);
    } else {
      datum.is_null = (datum.val.real_val == NULL_DOUBLE);
    }
  } else if (boost::get<float>(scalar_tv)) {
    CHECK_EQ(kFLOAT, ti.get_type());
    datum.val.real_val = *(boost::get<float>(scalar_tv));
    datum.is_null = (datum.val.real_val == NULL_FLOAT);
  } else if (boost::get<NullableString>(scalar_tv)) {
    auto s_n = boost::get<NullableString>(scalar_tv);
    auto s = boost::get<std::string>(s_n);
    if (s) {
      datum.val.str_val = *s;
    } else {
      auto null_p = boost::get<void*>(s_n);
      CHECK(null_p && !*null_p);
    }
    datum.is_null = !s;
  } else {
    CHECK(false);
  }
  return datum;
}

void MapDHandler::sql_execute(TQueryResult& _return,
                              const TSessionId session,
                              const std::string& query_str,
                              const bool column_format,
                              const std::string& nonce,
                              const int32_t first_n) {
  const auto session_info = MapDHandler::get_session(session);
  if (leaf_aggregator_.leafCount() > 0) {
#ifdef HAVE_RAVM
    cluster_execute(_return, session_info, query_str, column_format, nonce, first_n);
    _return.nonce = nonce;
#else
    CHECK(false);
#endif  // HAVE_RAVM
  } else {
    MapDHandler::sql_execute_impl(
        _return, session_info, query_str, column_format, nonce, session_info.get_executor_device_type(), first_n);
  }
}

namespace {

std::string apply_copy_to_shim(const std::string& query_str) {
  auto result = query_str;
  {
    // boost::regex copy_to{R"(COPY\s\((.*)\)\sTO\s(.*))", boost::regex::extended | boost::regex::icase};
    boost::regex copy_to{R"(COPY\s*\(([^#])(.+)\)\s+TO\s)", boost::regex::extended | boost::regex::icase};
    apply_shim(result, copy_to, [](std::string& result, const boost::smatch& what) {
      result.replace(what.position(), what.length(), "COPY (#~#" + what[1] + what[2] + "#~#) TO  ");
    });
  }
  return result;
}

}  // namespace

#ifdef HAVE_RAVM
namespace {

TQueryResult aggregate_execution_times(const std::vector<TQueryResult>& all_results) {
  for (const auto& result : all_results) {
    CHECK(result.row_set.rows.empty() && result.row_set.columns.empty());
  }
  const auto max_execution_time_it =
      std::max_element(all_results.begin(), all_results.end(), [](const TQueryResult& lhs, const TQueryResult& rhs) {
        return lhs.execution_time_ms < rhs.execution_time_ms;
      });
  const auto max_total_time_it =
      std::max_element(all_results.begin(), all_results.end(), [](const TQueryResult& lhs, const TQueryResult& rhs) {
        return lhs.total_time_ms < rhs.total_time_ms;
      });
  TQueryResult agggregated_result;
  agggregated_result.execution_time_ms = max_execution_time_it->execution_time_ms;
  agggregated_result.total_time_ms = max_total_time_it->total_time_ms;
  return agggregated_result;
}

}  // namespace

void MapDHandler::cluster_execute(TQueryResult& _return,
                                  const Catalog_Namespace::SessionInfo& session_info,
                                  const std::string& query_str,
                                  const bool column_format,
                                  const std::string& nonce,
                                  const int32_t first_n) {
  try {
    ParserWrapper pw{query_str};
    if (!pw.is_ddl && !pw.is_update_dml && !pw.is_other_explain) {
      const auto query_ra = MapDHandler::parse_to_ra(query_str, session_info);
      if (pw.is_select_calcite_explain) {
        // return the ra as the result
        convert_explain(_return, ResultRows(query_ra, 0), true);
        return;
      }
      ExecutionOptions eo = {false,
                             allow_multifrag_,
                             pw.is_select_explain,
                             allow_loop_joins_,
                             g_enable_watchdog,
                             jit_debug_,
                             false,
                             g_enable_dynamic_watchdog,
                             g_dynamic_watchdog_time_limit};
      const auto clock_begin = timer_start();
      const auto result = leaf_aggregator_.execute(session_info, query_ra, eo);
      _return.total_time_ms = timer_stop(clock_begin);
      _return.execution_time_ms = _return.total_time_ms - result.rs->getQueueTime();
      convert_rows(_return, result.targets_meta, *(result.rs), column_format, first_n);
    } else if (pw.is_update_dml) {
      std::unique_ptr<Planner::RootPlan> plan_ptr(
          parse_to_plan_legacy(apply_copy_to_shim(query_str), session_info, "validate"));
      const auto stmt_type = plan_ptr->get_stmt_type();
      CHECK_EQ(kINSERT, stmt_type);
      const auto td = session_info.get_catalog().getMetadataForTable(plan_ptr->get_result_table_id());
      CHECK(td);
      if (td->partitions == "REPLICATED") {
        // This is just a placeholder, we need real replication.
        auto all_results = leaf_aggregator_.forwardQueryToLeaves(session_info, query_str);
        _return = aggregate_execution_times(all_results);
      } else {
        std::random_device rd;
        std::mt19937_64 gen(rd());
        std::uniform_int_distribution<size_t> dis;
        const auto leaf_idx = dis(gen) % leaf_aggregator_.leafCount();
        _return = leaf_aggregator_.forwardQueryToLeaf(session_info, query_str, leaf_idx);
      }
    } else {
      if (pw.is_copy) {
        MapDHandler::sql_execute_impl(
            _return, session_info, query_str, column_format, nonce, session_info.get_executor_device_type(), first_n);
        return;
      }
      std::future<TQueryResult> aggregator_future{
          std::async(std::launch::async, [column_format, first_n, &nonce, &query_str, &session_info, this] {
            TQueryResult result;
            MapDHandler::sql_execute_impl(result,
                                          session_info,
                                          query_str,
                                          column_format,
                                          nonce,
                                          session_info.get_executor_device_type(),
                                          first_n);
            return result;
          })};
      auto all_results = leaf_aggregator_.forwardQueryToLeaves(session_info, query_str);
      all_results.push_back(aggregator_future.get());
      _return = aggregate_execution_times(all_results);
    }
  } catch (std::exception& e) {
    const auto mapd_exception = dynamic_cast<const TMapDException*>(&e);
    TMapDException ex;
    ex.error_msg = mapd_exception ? mapd_exception->error_msg : (std::string("Exception: ") + e.what());
    LOG(ERROR) << ex.error_msg;
    throw ex;
  }
}
#endif  // HAVE_RAVM

void MapDHandler::sql_validate(TTableDescriptor& _return, const TSessionId session, const std::string& query_str) {
  std::unique_ptr<const Planner::RootPlan> root_plan;
  const auto session_info = get_session(session);
#ifdef HAVE_CALCITE
  ParserWrapper pw{query_str};
  if (pw.is_select_explain || pw.is_other_explain || pw.is_ddl || pw.is_update_dml) {
    TMapDException ex;
    ex.error_msg = "Can only validate SELECT statements.";
    LOG(ERROR) << ex.error_msg;
    throw ex;
  }
#ifdef HAVE_RAVM
  MapDHandler::validate_rel_alg(_return, query_str, session_info);
#else   // HAVE_RAVM
  root_plan.reset(parse_to_plan(query_str, session_info));
#endif  // !HAVE_RAVM
#else   // HAVE_CALCITE
  root_plan.reset(parse_to_plan_legacy(query_str, session_info, "validate"));
#endif  // !HAVE_CALCITE
#ifndef HAVE_RAVM
  CHECK(root_plan);
  CHECK(root_plan->get_plan());
  const auto& target_list = root_plan->get_plan()->get_targetlist();
  for (const auto& target : target_list) {
    const auto& target_ti = target->get_expr()->get_type_info();
    TColumnType col_type;
    col_type.col_type.type = type_to_thrift(target_ti);
    col_type.col_type.encoding = encoding_to_thrift(target_ti);
    col_type.col_type.nullable = !target_ti.get_notnull();
    col_type.col_type.is_array = target_ti.get_type() == kARRAY;
    col_type.col_name = target->get_resname();
    col_type.col_type.precision = target_ti.get_precision();
    col_type.col_type.scale = target_ti.get_scale();
    const auto it_ok = _return.insert(std::make_pair(col_type.col_name, col_type));
    if (!it_ok.second) {
      TMapDException ex;
      ex.error_msg = "Duplicate alias: " + col_type.col_name;
      LOG(ERROR) << ex.error_msg;
      throw ex;
    }
  }
#endif  // !HAVE_RAVM
}

#ifdef HAVE_RAVM
void MapDHandler::validate_rel_alg(TTableDescriptor& _return,
                                   const std::string& query_str,
                                   const Catalog_Namespace::SessionInfo& session_info) {
  try {
    const auto query_ra = parse_to_ra(query_str, session_info);
    TQueryResult result;
    MapDHandler::execute_rel_alg(result, query_ra, true, session_info, ExecutorDeviceType::CPU, -1, false, true);
    const auto& row_desc = fixup_row_descriptor(result.row_set.row_desc, session_info.get_catalog());
    for (const auto& col_desc : row_desc) {
      const auto it_ok = _return.insert(std::make_pair(col_desc.col_name, col_desc));
      CHECK(it_ok.second);
    }
  } catch (std::exception& e) {
    TMapDException ex;
    ex.error_msg = std::string("Exception: ") + e.what();
    LOG(ERROR) << ex.error_msg;
    throw ex;
  }
}
#endif  // HAVE_RAVM

// DEPRECATED - use get_row_for_pixel()
void MapDHandler::get_rows_for_pixels(TPixelResult& _return,
                                      const TSessionId session,
                                      const int64_t widget_id,
                                      const std::vector<TPixel>& pixels,
                                      const std::string& table_name,
                                      const std::vector<std::string>& col_names,
                                      const bool column_format,
                                      const std::string& nonce) {
  _return.nonce = nonce;
  if (!enable_rendering_) {
    TMapDException ex;
    ex.error_msg = "Backend rendering is disabled.";
    LOG(ERROR) << ex.error_msg;
    throw ex;
  }
#ifdef HAVE_RENDERING
  std::lock_guard<std::mutex> render_lock(render_mutex_);
  mapd_lock_guard<mapd_shared_mutex> write_lock(sessions_mutex_);
  auto session_it = get_session_it(session);
  auto session_info_ptr = session_it->second.get();
  auto& cat = session_info_ptr->get_catalog();
  auto executor = Executor::getExecutor(cat.get_currentDB().dbId, "", "", mapd_parameters_, render_manager_.get());
  CHECK(executor);
  CHECK(ExecutorDeviceType::GPU == session_info_ptr->get_executor_device_type());

  for (const auto& pixel : pixels) {
    const auto rowid = executor->getRowidForPixel(
        pixel.x, pixel.y, session_info_ptr->get_session_id(), 1);  // TODO(alex): de-hardcode user widget

    if (rowid < 0) {
      continue;
    }

    // TODO(alex): fix potential SQL injection issues?
    const auto projection = boost::algorithm::join(col_names, ", ");
    const auto query_str =
        "SELECT " + projection + " FROM " + table_name + " WHERE rowid = " + std::to_string(rowid) + ";";
    TQueryResult ret;
    sql_execute_impl(ret, *session_info_ptr, query_str, column_format, nonce, ExecutorDeviceType::CPU, -1);
    TPixelRows pixel_rows;
    pixel_rows.pixel = pixel;
    pixel_rows.row_set = ret.row_set;
    _return.pixel_rows.push_back(pixel_rows);
  }
#endif  // HAVE_RENDERING
}

// DEPRECATED - use get_result_row_for_pixel()
void MapDHandler::get_row_for_pixel(TPixelRowResult& _return,
                                    const TSessionId session,
                                    const int64_t widget_id,
                                    const TPixel& pixel,
                                    const std::string& table_name,
                                    const std::vector<std::string>& col_names,
                                    const bool column_format,
                                    const int32_t pixelRadius,
                                    const std::string& nonce) {
  _return.nonce = nonce;
  if (!enable_rendering_) {
    TMapDException ex;
    ex.error_msg = "Backend rendering is disabled.";
    LOG(ERROR) << ex.error_msg;
    throw ex;
  }
#ifdef HAVE_RENDERING
  std::lock_guard<std::mutex> render_lock(render_mutex_);
  auto session_it = get_session_it(session);
  auto session_info_ptr = session_it->second.get();
  auto& cat = session_info_ptr->get_catalog();
  auto executor = Executor::getExecutor(cat.get_currentDB().dbId, "", "", mapd_parameters_, render_manager_.get());
  CHECK(executor);
  CHECK(ExecutorDeviceType::GPU == session_info_ptr->get_executor_device_type());
  const auto rowid = executor->getRowidForPixel(
      pixel.x, pixel.y, session_info_ptr->get_session_id(), 1, pixelRadius);  // TODO(alex): de-hardcode user widget

  _return.pixel = pixel;
  _return.row_id = rowid;

  if (rowid >= 0) {
    // TODO(alex): fix potential SQL injection issues?
    const auto projection = boost::algorithm::join(col_names, ", ");
    const auto query_str =
        "SELECT " + projection + " FROM " + table_name + " WHERE rowid = " + std::to_string(rowid) + ";";
    TQueryResult ret;
    sql_execute_impl(ret, *session_info_ptr, query_str, column_format, nonce, ExecutorDeviceType::CPU, -1);
    _return.row_set = ret.row_set;
  }
#endif  // HAVE_RENDERING
}

void MapDHandler::get_result_row_for_pixel(TPixelTableRowResult& _return,
                                           const TSessionId session,
                                           const int64_t widget_id,
                                           const TPixel& pixel,
                                           const std::map<std::string, std::vector<std::string>>& table_col_names,
                                           const bool column_format,
                                           const int32_t pixelRadius,
                                           const std::string& nonce) {
  _return.nonce = nonce;
  if (!enable_rendering_) {
    TMapDException ex;
    ex.error_msg = "Backend rendering is disabled.";
    LOG(ERROR) << ex.error_msg;
    throw ex;
  }
#ifdef HAVE_RENDERING
  if (leaf_aggregator_.leafCount() > 0) {
    const auto session_info = MapDHandler::get_session(session);
    _return = leaf_aggregator_.getResultRowForPixel(
        session_info, widget_id, pixel, table_col_names, column_format, pixelRadius);
    _return.nonce = nonce;
    return;
  }
  try {
    std::lock_guard<std::mutex> render_lock(render_mutex_);
    auto session_it = get_session_it(session);
    auto session_info_ptr = session_it->second.get();
    auto& cat = session_info_ptr->get_catalog();
    auto executor = Executor::getExecutor(cat.get_currentDB().dbId, "", "", mapd_parameters_, render_manager_.get());
    CHECK(executor && render_manager_);
    CHECK(ExecutorDeviceType::GPU == session_info_ptr->get_executor_device_type());

    _return.pixel = pixel;

    if (!render_manager_->hasUserWidget(session_info_ptr->get_session_id(), widget_id)) {
      _return.table_id = -1;
      _return.row_id = -1;
      _return.vega_table_name = "";
      return;
    }

    render_manager_->setActiveUserWidget(session_info_ptr->get_session_id(), widget_id);

    int32_t table_id;
    int64_t row_id;
    std::string vega_table_name;
    std::tie(table_id, row_id, vega_table_name) = render_manager_->getIdAt(pixel.x, pixel.y, pixelRadius);

    _return.vega_table_name = vega_table_name;
    _return.table_id = table_id;
    _return.row_id = row_id;

    // NOTE: -1 for table_id or row_id indicates nothing was hit
    // Table ids < -1 are for cached non-projection queries.
    // Table ids >= 0 are for projection queries and actually
    // reference database table ids.
    if (table_id != -1 && row_id >= 0) {
      TQueryResult ret;
      ret.row_set.is_columnar = column_format;

      std::string table_name;
      bool isCache = (table_id < 0);  // Table ids < -1 indicate rendered non-projection queries.
                                      // The results of these queries may be cached, and if not
                                      // cached, then the queries will be re-run here.

      if (isCache) {
        // Query is a non-projection query and may be cached.
        auto total_time = measure<>::execution([&]() {
          auto cache_table_id = table_id;

          // retrieve the table_id and table_name of the primary table from the query
          std::tie(table_id, table_name) = render_manager_->getPrimaryQueryCacheTableInfo(cache_table_id);

          auto itr = table_col_names.find(vega_table_name);
          if (itr == table_col_names.end() || itr->second.empty()) {
            return;
          }

#if defined(HAVE_CALCITE) && defined(HAVE_RAVM)
          ExecutionResult result{ResultRows({}, {}, nullptr, nullptr, {}, ExecutorDeviceType::CPU), {}};
#endif
          const ResultRows* resultRowPtr;
          const std::vector<TargetMetaInfo>* resultRowShapePtr;
          std::tie(resultRowPtr, resultRowShapePtr) = render_manager_->getQueryCacheResults(cache_table_id);

          if (!resultRowShapePtr->size()) {
            // cached results don't exist for the query. The cache may have been cleared or the query itself
            // may be fast enough to run per-hit-test.
            // Rerun query here.
            std::string query_str = render_manager_->getQueryForQueryCache(cache_table_id);
#if defined(HAVE_CALCITE) && defined(HAVE_RAVM)
            ParserWrapper pw{query_str};
            if (!pw.is_ddl && !pw.is_update_dml && !pw.is_other_explain) {
              std::string query_ra = parse_to_ra(query_str, *session_info_ptr);
              CompilationOptions co = {session_info_ptr->get_executor_device_type(),
                                       true,
                                       ExecutorOptLevel::Default,
                                       g_enable_dynamic_watchdog};
              ExecutionOptions eo = {false,
                                     allow_multifrag_,
                                     false,
                                     allow_loop_joins_,
                                     g_enable_watchdog,
                                     jit_debug_,
                                     false,
                                     g_enable_dynamic_watchdog,
                                     g_dynamic_watchdog_time_limit};
              RelAlgExecutor ra_executor(executor.get(), cat);
              result = ra_executor.executeRelAlgQuery(query_ra, co, eo, nullptr);

              resultRowPtr = &(result.getRows());
              resultRowShapePtr = &(result.getTargetsMeta());

              // TODO(croot): if the query takes too long here, should we add the results to
              // the cache? If so, an API in QueryRenderManager to add query results
              // to its cache would need to be authored.
            }
#else
            throw std::runtime_error(
                "Running non-projection queries to resolve hit-testing is only supported on a Calcite/RAVM build.");
#endif  // HAVE_CALCITE && HAVE_RAVM
          }

          std::regex projectAsRegex("\\s*(\\w+)\\s+as\\s+(\\w+)\\s*",
                                    std::regex_constants::ECMAScript | std::regex_constants::icase);
          std::regex funcProjectAsRegex("\\s*(\\w+\\s*\\(.*\\))\\s+as\\s+(\\w+)\\s*",
                                        std::regex_constants::ECMAScript | std::regex_constants::icase);
          std::smatch projectAsMatch;

          size_t i;
          int backref_rowid_idx = -1;
          TColumnType col_info;
          TRow trow;
          std::vector<std::pair<std::string, size_t>> unusedProjIdx;
          bool isPolyCache = render_manager_->isPolyQueryCache(cache_table_id);

          // iterate through the columns requested by the user, and look for
          // it in the generated results (results are either cached or generated
          // by the above query)
          // If the particular column is not found, then keep those as we'll
          // run a final query to get those columns for a specific rowid of
          // the original table
          for (size_t idx = 0; idx < itr->second.size(); ++idx) {
            // support the different ways columns can be projected, such as
            // AS and Functions/Expressions
            // For example, "conv_4326_900913_x(lon) as x" is a legal
            // column projection, and we need to support it here.
            // TODO(croot): may need to properly support expressions
            std::string colName = itr->second[idx];
            if (std::regex_match(colName, projectAsMatch, projectAsRegex)) {
              colName = projectAsMatch[1];
            } else if (std::regex_match(colName, projectAsMatch, funcProjectAsRegex)) {
              colName = projectAsMatch[2];
            }

            // iterate through results to see if we've already generated results
            // for this particular column.
            for (i = 0; i < resultRowShapePtr->size(); ++i) {
              // TODO(croot): do I need to ignore case or not?
              if (backref_rowid_idx < 0 && (*resultRowShapePtr)[i].get_resname() == "rowid") {
                // found rowid in the results -- need this to be able to actually
                // backrefence the true rowid of a particular table
                backref_rowid_idx = static_cast<int>(i);
              }

              if ((*resultRowShapePtr)[i].get_resname() == colName) {
                // found the column
                break;
              }
            }

            if (i != resultRowShapePtr->size()) {
              // Found the colum in the generated results, so convert it
              // into the return format expected

              if (ret.row_set.row_desc.empty()) {
                ret.row_set.row_desc.resize(itr->second.size());
              }
              auto target = (*resultRowShapePtr)[i];
              col_info.col_name = (projectAsMatch.empty() ? target.get_resname() : projectAsMatch[2]);
              if (col_info.col_name.empty()) {
                col_info.col_name = "result_" + std::to_string(idx + 1);
              }
              const auto& target_ti = target.get_type_info();
              col_info.col_type.type = type_to_thrift(target_ti);
              col_info.col_type.encoding = encoding_to_thrift(target_ti);
              col_info.col_type.nullable = !target_ti.get_notnull();
              col_info.col_type.is_array = target_ti.get_type() == kARRAY;
              ret.row_set.row_desc[idx] = col_info;

              if (column_format) {
                if (ret.row_set.columns.empty()) {
                  ret.row_set.columns.resize(itr->second.size());
                }
                const auto col_val = resultRowPtr->getRowAt(row_id, i, true);
                TColumn tcol;
                value_to_thrift_column(col_val, (*resultRowShapePtr)[i].get_type_info(), tcol);
                ret.row_set.columns[idx] = tcol;
              } else {
                if (trow.cols.empty()) {
                  trow.cols.resize(itr->second.size());
                }
                const auto col_val = resultRowPtr->getRowAt(row_id, i, true);
                trow.cols[idx] = value_to_thrift(col_val, (*resultRowShapePtr)[i].get_type_info());
              }
            } else if (isPolyCache) {
              unusedProjIdx.push_back(std::make_pair(itr->second[idx], idx));
            } else {
              throw std::runtime_error("get_result_row_for_pixel(): cannot find column " + colName +
                                       " in query cache.");
            }
          }

          // Run a final request to generate results of the columns that weren't found
          // in the original render query.
          // Requires a rowid column of a referenc-able table for backreferencing.
          // TODO(croot): only support poly tables right now, but extend to anything else
          // as long as the original table is back-referencable.
          if (isPolyCache) {
            CHECK(is_poly_table(table_id, cat)) << "Table: " << table_id << " is not a poly table";

            if (unusedProjIdx.size()) {
              auto poly_row_id = row_id;
              if (!ret.row_set.row_desc.empty()) {
                // rowid hasn't been found yet, look for it
                if (backref_rowid_idx < 0) {
                  for (i = 0; i < resultRowShapePtr->size(); ++i) {
                    // TODO(croot): do I need to ignore case or not?
                    if ((*resultRowShapePtr)[i].get_resname() == "rowid") {
                      backref_rowid_idx = i;
                      break;
                    }
                  }
                }

                // need a rowid
                // TODO(croot): throw error
                CHECK(backref_rowid_idx >= 0) << "A rowid is currently required for poly-table rendering";

                // get the true row id of the primary table
                const auto col_val = resultRowPtr->getRowAt(row_id, backref_rowid_idx, true);
                const auto scalar_val = boost::get<ScalarTargetValue>(&col_val);
                CHECK(scalar_val) << "\"rowid\" is not a scalar column";
                const auto int_val = boost::get<int64_t>(scalar_val);
                CHECK(int_val) << "\"rowid\" must be an integer";
                poly_row_id = *int_val;
              }

              // Compile final query to retrieve extra columns
              // TODO(alex): fix potential SQL injection issues?
              std::string projection = "";
              std::for_each(unusedProjIdx.begin(),
                            unusedProjIdx.end(),
                            [&projection](const std::pair<std::string, size_t>& item) {
                              if (!projection.empty()) {
                                projection += ", ";
                              }
                              projection += item.first;
                            });

              // TODO(croot): what about non-projection queries?
              // TODO(croot): what about poly tables?
              const auto query_str = "SELECT " + projection + " FROM " + table_name + " WHERE rowid = " +
                                     std::to_string(poly_row_id) + ";";
              TQueryResult tmpResult;
              sql_execute_impl(
                  tmpResult, *session_info_ptr, query_str, column_format, nonce, ExecutorDeviceType::CPU, -1);

              CHECK(tmpResult.row_set.row_desc.size() == unusedProjIdx.size() &&
                    (column_format || tmpResult.row_set.rows.size() == 1));

              // convert query results into the appropriate return struct
              if (ret.row_set.row_desc.empty()) {
                ret = std::move(tmpResult);
              } else {
                for (i = 0; i < unusedProjIdx.size(); ++i) {
                  auto idx = unusedProjIdx[i].second;
                  ret.row_set.row_desc[idx] = tmpResult.row_set.row_desc[i];

                  if (column_format) {
                    CHECK(!ret.row_set.columns.empty());
                    ret.row_set.columns[idx] = tmpResult.row_set.columns[i];
                  } else {
                    // NOTE: already checked that the number of rows in tmp result is 1
                    CHECK(!trow.cols.empty());
                    trow.cols[idx] = tmpResult.row_set.rows[0].cols[i];
                  }
                }
              }
            }
          }

          if (!trow.cols.empty()) {
            ret.row_set.rows.push_back(trow);
          }
        });
        LOG(INFO) << "Resolving hit test for non-projection query. Total: " << total_time << " (ms)";
      } else {
        auto td = cat.getMetadataForTable(table_id);
        CHECK(td) << "Table doesn't exist for table_id: " << table_id;
        CHECK(!isCache || td->tableName == table_name) << "Table names don't match for table_id " << table_id << ", "
                                                       << table_name << " != " << td->tableName;

        if (!table_name.size()) {
          table_name = td->tableName;
        }

        auto itr = table_col_names.find(vega_table_name);
        if (itr == table_col_names.end()) {
          return;
        }

        // TODO(alex): fix potential SQL injection issues?
        auto projection = boost::algorithm::join(itr->second, ", ");

        // TODO(croot): what about non-projection queries?
        // TODO(croot): what about poly tables?
        const auto query_str =
            "SELECT " + projection + " FROM " + table_name + " WHERE rowid = " + std::to_string(row_id) + ";";
        sql_execute_impl(ret, *session_info_ptr, query_str, column_format, nonce, ExecutorDeviceType::CPU, -1);
      }
      _return.row_set = ret.row_set;
    }
  } catch (std::exception& e) {
    TMapDException ex;
    ex.error_msg = std::string("Exception: ") + e.what();
    LOG(ERROR) << ex.error_msg;
    throw ex;
  }
#endif  // HAVE_RENDERING
}

TColumnType MapDHandler::populateThriftColumnType(const Catalog_Namespace::Catalog* cat, const ColumnDescriptor* cd) {
  TColumnType col_type;
  col_type.col_name = cd->columnName;
  col_type.src_name = cd->sourceName;
  col_type.col_type.type = type_to_thrift(cd->columnType);
  col_type.col_type.encoding = encoding_to_thrift(cd->columnType);
  col_type.col_type.nullable = !cd->columnType.get_notnull();
  col_type.col_type.is_array = cd->columnType.get_type() == kARRAY;
  col_type.col_type.precision = cd->columnType.get_precision();
  col_type.col_type.scale = cd->columnType.get_scale();
  if (cd->columnType.get_compression() == EncodingType::kENCODING_DICT && cat != nullptr) {
    // have to get the actual size of the encoding from the dictionary definition
    auto dd = cat->getMetadataForDict(cd->columnType.get_comp_param(), false);
    if (!dd) {
      TMapDException ex;
      ex.error_msg = "Dictionary doesn't exist";
      LOG(ERROR) << ex.error_msg;
      throw ex;
    }
    col_type.col_type.comp_param = dd->dictNBits;
  } else {
    col_type.col_type.comp_param = cd->columnType.get_comp_param();
  }
  return col_type;
}

void MapDHandler::get_table_descriptor(TTableDescriptor& _return,
                                       const TSessionId session,
                                       const std::string& table_name) {
  TRowDescriptor rd;
  TTableDetails table_details;
  get_table_details(table_details, session, table_name);
  for (const auto cd : table_details.table_desc) {
    _return.insert(std::make_pair(cd.col_name, cd));
  }
}

void MapDHandler::get_table_details(TTableDetails& _return, const TSessionId session, const std::string& table_name) {
  const auto session_info = get_session(session);
  auto& cat = session_info.get_catalog();
  auto td = cat.getMetadataForTable(table_name);
  if (!td) {
    TMapDException ex;
    ex.error_msg = "Table doesn't exist";
    LOG(ERROR) << ex.error_msg;
    throw ex;
  }
  if (td->isView) {
#ifdef HAVE_CALCITE
    try {
      const auto query_ra = parse_to_ra(td->viewSQL, session_info);
      TQueryResult result;
      execute_rel_alg(result, query_ra, true, session_info, ExecutorDeviceType::CPU, false, true, -1);
      _return.table_desc = fixup_row_descriptor(result.row_set.row_desc, cat);
    } catch (std::exception& e) {
      TMapDException ex;
      ex.error_msg = std::string("Exception: ") + e.what();
      LOG(ERROR) << ex.error_msg;
      throw ex;
    }
#else
    TMapDException ex;
    ex.error_msg = "Views not supported with legacy parser";
    LOG(ERROR) << ex.error_msg;
    throw ex;
#endif  // HAVE_CALCITE
  } else {
    const auto col_descriptors = cat.getAllColumnMetadataForTable(td->tableId, false, true);
    for (const auto cd : col_descriptors) {
      _return.table_desc.push_back(populateThriftColumnType(&cat, cd));
    }
  }
  _return.fragment_size = td->maxFragRows;
  _return.page_size = td->fragPageSize;
  _return.max_rows = td->maxRows;
  _return.view_sql = td->viewSQL;
}

void MapDHandler::get_row_descriptor(TRowDescriptor& _return, const TSessionId session, const std::string& table_name) {
  TTableDetails table_details;
  get_table_details(table_details, session, table_name);
  _return = table_details.table_desc;
}

void MapDHandler::get_frontend_view(TFrontendView& _return, const TSessionId session, const std::string& view_name) {
  const auto session_info = get_session(session);
  auto& cat = session_info.get_catalog();
  auto vd = cat.getMetadataForFrontendView(std::to_string(session_info.get_currentUser().userId), view_name);
  if (!vd) {
    TMapDException ex;
    ex.error_msg = "View " + view_name + " doesn't exist";
    LOG(ERROR) << ex.error_msg;
    throw ex;
  }
  _return.view_name = view_name;
  _return.view_state = vd->viewState;
  _return.image_hash = vd->imageHash;
  _return.update_time = vd->updateTime;
  _return.view_metadata = vd->viewMetadata;
}

void MapDHandler::get_link_view(TFrontendView& _return, const TSessionId session, const std::string& link) {
  const auto session_info = get_session(session);
  auto& cat = session_info.get_catalog();
  auto ld = cat.getMetadataForLink(std::to_string(cat.get_currentDB().dbId) + link);
  if (!ld) {
    TMapDException ex;
    ex.error_msg = "Link " + link + " is not valid.";
    LOG(ERROR) << ex.error_msg;
    throw ex;
  }
  _return.view_state = ld->viewState;
  _return.view_name = ld->link;
  _return.update_time = ld->updateTime;
  _return.view_metadata = ld->viewMetadata;
}

void MapDHandler::get_tables(std::vector<std::string>& table_names, const TSessionId session) {
  const auto session_info = get_session(session);
  auto& cat = session_info.get_catalog();
  const auto tables = cat.getAllTableMetadata();
  for (const auto td : tables) {
    table_names.push_back(td->tableName);
  }
}

void MapDHandler::get_users(std::vector<std::string>& user_names, const TSessionId session) {
  const auto session_info = get_session(session);
  std::list<Catalog_Namespace::UserMetadata> user_list = sys_cat_->getAllUserMetadata();
  for (auto u : user_list) {
    user_names.push_back(u.userName);
  }
}

void MapDHandler::get_version(std::string& version) {
  version = MapDRelease;
}

void MapDHandler::get_memory_gpu(std::string& memory, const TSessionId session) {
  const auto session_info = get_session(session);
  memory = sys_cat_->get_dataMgr().dumpLevel(MemoryLevel::GPU_LEVEL);
}

void MapDHandler::clear_gpu_memory(const TSessionId session) {
  const auto session_info = get_session(session);
  sys_cat_->get_dataMgr().clearMemory(MemoryLevel::GPU_LEVEL);
}

void MapDHandler::clear_cpu_memory(const TSessionId session) {
  const auto session_info = get_session(session);
  sys_cat_->get_dataMgr().clearMemory(MemoryLevel::CPU_LEVEL);
}

// void get_memory_summary(std::string& memory) { memory = sys_cat_->get_dataMgr().getMemorySummary(); }

void MapDHandler::get_memory_summary(TMemorySummary& memory, const TSessionId session) {
  const auto session_info = get_session(session);
  Data_Namespace::memorySummary internal_memory = sys_cat_->get_dataMgr().getMemorySummary();
  memory.cpu_memory_in_use = internal_memory.cpuMemoryInUse;
  for (auto gpu : internal_memory.gpuSummary) {
    TGpuMemorySummary gs;
    gs.in_use = gpu.inUse;
    gs.max = gpu.max;
    gs.allocated = gpu.allocated;
    gs.is_allocation_capped = gpu.isAllocationCapped;
    memory.gpu_summary.push_back(gs);
  }
}

void MapDHandler::get_databases(std::vector<TDBInfo>& dbinfos, const TSessionId session) {
  const auto session_info = get_session(session);
  std::list<Catalog_Namespace::DBMetadata> db_list = sys_cat_->getAllDBMetadata();
  std::list<Catalog_Namespace::UserMetadata> user_list = sys_cat_->getAllUserMetadata();
  for (auto d : db_list) {
    TDBInfo dbinfo;
    dbinfo.db_name = d.dbName;
    for (auto u : user_list) {
      if (d.dbOwner == u.userId) {
        dbinfo.db_owner = u.userName;
        break;
      }
    }
    dbinfos.push_back(dbinfo);
  }
}

void MapDHandler::get_frontend_views(std::vector<TFrontendView>& view_names, const TSessionId session) {
  const auto session_info = get_session(session);
  auto& cat = session_info.get_catalog();
  const auto views = cat.getAllFrontendViewMetadata();
  for (const auto vd : views) {
    if (vd->userId == session_info.get_currentUser().userId) {
      TFrontendView fv;
      fv.view_name = vd->viewName;
      fv.image_hash = vd->imageHash;
      fv.update_time = vd->updateTime;
      fv.view_metadata = vd->viewMetadata;
      view_names.push_back(fv);
    }
  }
}

void MapDHandler::set_execution_mode(const TSessionId session, const TExecuteMode::type mode) {
  mapd_lock_guard<mapd_shared_mutex> write_lock(sessions_mutex_);
  auto session_it = get_session_it(session);
  if (leaf_aggregator_.leafCount() > 0) {
    leaf_aggregator_.set_execution_mode(session, mode);
    try {
      MapDHandler::set_execution_mode_nolock(session_it->second.get(), mode);
    } catch (const TMapDException& e) {
      LOG(INFO) << "Aggregator failed to set execution mode: " << e.error_msg;
    }
    return;
  }
  MapDHandler::set_execution_mode_nolock(session_it->second.get(), mode);
}

void MapDHandler::load_table_binary(const TSessionId session,
                                    const std::string& table_name,
                                    const std::vector<TRow>& rows) {
  check_read_only("load_table_binary");
  const auto session_info = get_session(session);
  auto& cat = session_info.get_catalog();
  const TableDescriptor* td = cat.getMetadataForTable(table_name);
  if (td == nullptr) {
    TMapDException ex;
    ex.error_msg = "Table " + table_name + " does not exist.";
    LOG(ERROR) << ex.error_msg;
    throw ex;
  }
  Importer_NS::Loader loader(cat, td);
  // TODO(andrew): nColumns should be number of non-virtual/non-system columns.
  //               Subtracting 1 (rowid) until TableDescriptor is updated.
  if (rows.front().cols.size() != static_cast<size_t>(td->nColumns) - 1) {
    TMapDException ex;
    ex.error_msg = "Wrong number of columns to load into Table " + table_name;
    LOG(ERROR) << ex.error_msg;
    throw ex;
  }
  auto col_descs = loader.get_column_descs();
  std::vector<std::unique_ptr<Importer_NS::TypedImportBuffer>> import_buffers;
  for (auto cd : col_descs) {
    import_buffers.push_back(std::unique_ptr<Importer_NS::TypedImportBuffer>(
        new Importer_NS::TypedImportBuffer(cd, loader.get_string_dict(cd))));
  }
  for (auto row : rows) {
    try {
      int col_idx = 0;
      for (auto cd : col_descs) {
        import_buffers[col_idx]->add_value(cd, row.cols[col_idx], row.cols[col_idx].is_null);
        col_idx++;
      }
    } catch (const std::exception& e) {
      LOG(WARNING) << "load_table exception thrown: " << e.what() << ". Row discarded.";
    }
  }
  loader.load(import_buffers, rows.size());
}

void MapDHandler::load_table(const TSessionId session,
                             const std::string& table_name,
                             const std::vector<TStringRow>& rows) {
  check_read_only("load_table");
  const auto session_info = get_session(session);
  auto& cat = session_info.get_catalog();
  const TableDescriptor* td = cat.getMetadataForTable(table_name);
  if (td == nullptr) {
    TMapDException ex;
    ex.error_msg = "Table " + table_name + " does not exist.";
    LOG(ERROR) << ex.error_msg;
    throw ex;
  }
  Importer_NS::Loader loader(cat, td);
  Importer_NS::CopyParams copy_params;
  // TODO(andrew): nColumns should be number of non-virtual/non-system columns.
  //               Subtracting 1 (rowid) until TableDescriptor is updated.
  if (rows.front().cols.size() != static_cast<size_t>(td->nColumns) - 1) {
    TMapDException ex;
    ex.error_msg = "Wrong number of columns to load into Table " + table_name + " (" +
                   std::to_string(rows.front().cols.size()) + " vs " + std::to_string(td->nColumns - 1) + ")";
    LOG(ERROR) << ex.error_msg;
    throw ex;
  }
  auto col_descs = loader.get_column_descs();
  std::vector<std::unique_ptr<Importer_NS::TypedImportBuffer>> import_buffers;
  for (auto cd : col_descs) {
    import_buffers.push_back(std::unique_ptr<Importer_NS::TypedImportBuffer>(
        new Importer_NS::TypedImportBuffer(cd, loader.get_string_dict(cd))));
  }
  for (auto row : rows) {
    try {
      int col_idx = 0;
      for (auto cd : col_descs) {
        import_buffers[col_idx]->add_value(cd, row.cols[col_idx].str_val, row.cols[col_idx].is_null, copy_params);
        col_idx++;
      }
    } catch (const std::exception& e) {
      LOG(WARNING) << "load_table exception thrown: " << e.what() << ". Row discarded.";
    }
  }
  loader.load(import_buffers, rows.size());
}

char MapDHandler::unescape_char(std::string str) {
  char out = str[0];
  if (str.size() == 2 && str[0] == '\\') {
    if (str[1] == 't')
      out = '\t';
    else if (str[1] == 'n')
      out = '\n';
    else if (str[1] == '0')
      out = '\0';
    else if (str[1] == '\'')
      out = '\'';
    else if (str[1] == '\\')
      out = '\\';
  }
  return out;
}

Importer_NS::CopyParams MapDHandler::thrift_to_copyparams(const TCopyParams& cp) {
  Importer_NS::CopyParams copy_params;
  copy_params.has_header = cp.has_header;
  copy_params.quoted = cp.quoted;
  if (cp.delimiter.length() > 0)
    copy_params.delimiter = unescape_char(cp.delimiter);
  else
    copy_params.delimiter = '\0';
  if (cp.null_str.length() > 0)
    copy_params.null_str = cp.null_str;
  if (cp.quote.length() > 0)
    copy_params.quote = unescape_char(cp.quote);
  if (cp.escape.length() > 0)
    copy_params.escape = unescape_char(cp.escape);
  if (cp.line_delim.length() > 0)
    copy_params.line_delim = unescape_char(cp.line_delim);
  if (cp.array_delim.length() > 0)
    copy_params.array_delim = unescape_char(cp.array_delim);
  if (cp.array_begin.length() > 0)
    copy_params.array_begin = unescape_char(cp.array_begin);
  if (cp.array_end.length() > 0)
    copy_params.array_end = unescape_char(cp.array_end);
  if (cp.threads != 0)
    copy_params.threads = cp.threads;
  switch (cp.table_type) {
    case TTableType::POLYGON:
      copy_params.table_type = Importer_NS::TableType::POLYGON;
      break;
    default:
      copy_params.table_type = Importer_NS::TableType::DELIMITED;
      break;
  }
  return copy_params;
}

TCopyParams MapDHandler::copyparams_to_thrift(const Importer_NS::CopyParams& cp) {
  TCopyParams copy_params;
  copy_params.delimiter = cp.delimiter;
  copy_params.null_str = cp.null_str;
  copy_params.has_header = cp.has_header;
  copy_params.quoted = cp.quoted;
  copy_params.quote = cp.quote;
  copy_params.escape = cp.escape;
  copy_params.line_delim = cp.line_delim;
  copy_params.array_delim = cp.array_delim;
  copy_params.array_begin = cp.array_begin;
  copy_params.array_end = cp.array_end;
  copy_params.threads = cp.threads;
  switch (cp.table_type) {
    case Importer_NS::TableType::POLYGON:
      copy_params.table_type = TTableType::POLYGON;
      break;
    default:
      copy_params.table_type = TTableType::DELIMITED;
      break;
  }
  return copy_params;
}

void MapDHandler::detect_column_types(TDetectResult& _return,
                                      const TSessionId session,
                                      const std::string& file_name_in,
                                      const TCopyParams& cp) {
  check_read_only("detect_column_types");
  get_session(session);

  // Assume relative paths are relative to data_path / mapd_import / <session>
  std::string file_name{file_name_in};
  auto file_path = boost::filesystem::path(file_name);
  if (!boost::filesystem::path(file_name).is_absolute()) {
    file_path = import_path_ / std::to_string(session) / boost::filesystem::path(file_name).filename();
    file_name = file_path.string();
  }

  if (!boost::filesystem::exists(file_path)) {
    TMapDException ex;
    ex.error_msg = "File does not exist: " + file_path.string();
    LOG(ERROR) << ex.error_msg;
    throw ex;
  }

  Importer_NS::CopyParams copy_params = thrift_to_copyparams(cp);

  try {
    if (copy_params.table_type == Importer_NS::TableType::DELIMITED) {
      Importer_NS::Detector detector(file_path, copy_params);
      std::vector<SQLTypes> best_types = detector.best_sqltypes;
      std::vector<EncodingType> best_encodings = detector.best_encodings;
      std::vector<std::string> headers = detector.get_headers();
      copy_params = detector.get_copy_params();

      _return.copy_params = copyparams_to_thrift(copy_params);
      _return.row_set.row_desc.resize(best_types.size());
      TColumnType col;
      for (size_t col_idx = 0; col_idx < best_types.size(); col_idx++) {
        SQLTypes t = best_types[col_idx];
        EncodingType encodingType = best_encodings[col_idx];
        SQLTypeInfo* ti = new SQLTypeInfo(t, false, encodingType);
        col.col_type.type = type_to_thrift(*ti);
        col.col_type.encoding = encoding_to_thrift(*ti);
        col.col_name = headers[col_idx];
        col.is_reserved_keyword =
            reserved_keywords.find(boost::to_upper_copy<std::string>(col.col_name)) != reserved_keywords.end();
        _return.row_set.row_desc[col_idx] = col;
      }
      size_t num_samples = 100;
      auto sample_data = detector.get_sample_rows(num_samples);

      TRow sample_row;
      for (auto row : sample_data) {
        sample_row.cols.clear();
        for (const auto& s : row) {
          TDatum td;
          td.val.str_val = s;
          td.is_null = s.empty();
          sample_row.cols.push_back(td);
        }
        _return.row_set.rows.push_back(sample_row);
      }
    } else if (copy_params.table_type == Importer_NS::TableType::POLYGON) {
      check_geospatial_files(file_path);
      std::list<ColumnDescriptor> cds = Importer_NS::Importer::gdalToColumnDescriptors(file_path.string());
      for (auto cd : cds) {
        cd.columnName = sanitize_name(cd.columnName);
        _return.row_set.row_desc.push_back(populateThriftColumnType(nullptr, &cd));
      }
      std::map<std::string, std::vector<std::string>> sample_data;
      Importer_NS::Importer::readMetadataSampleGDAL(file_path.string(), sample_data, 100);
      if (sample_data.size() > 0) {
        for (size_t i = 0; i < sample_data.begin()->second.size(); i++) {
          TRow sample_row;
          for (auto cd : cds) {
            TDatum td;
            td.val.str_val = sample_data[cd.sourceName].at(i);
            td.is_null = td.val.str_val.empty();
            sample_row.cols.push_back(td);
          }
          _return.row_set.rows.push_back(sample_row);
        }
      }
    }
  } catch (const std::exception& e) {
    TMapDException ex;
    ex.error_msg = "detect_column_types error: " + std::string(e.what());
    LOG(ERROR) << ex.error_msg;
    throw ex;
  }
}

Planner::RootPlan* MapDHandler::parse_to_plan_legacy(const std::string& query_str,
                                                     const Catalog_Namespace::SessionInfo& session_info,
                                                     const std::string& action /* render or validate */) {
  auto& cat = session_info.get_catalog();
  LOG(INFO) << action << ": " << query_str;
  SQLParser parser;
  std::list<std::unique_ptr<Parser::Stmt>> parse_trees;
  std::string last_parsed;
  int num_parse_errors = 0;
  try {
    num_parse_errors = parser.parse(query_str, parse_trees, last_parsed);
  } catch (std::exception& e) {
    TMapDException ex;
    ex.error_msg = std::string("Exception: ") + e.what();
    LOG(ERROR) << ex.error_msg;
    throw ex;
  }
  if (num_parse_errors > 0) {
    TMapDException ex;
    ex.error_msg = "Syntax error at: " + last_parsed;
    LOG(ERROR) << ex.error_msg;
    throw ex;
  }
  if (parse_trees.size() != 1) {
    TMapDException ex;
    ex.error_msg = "Can only " + action + " a single query at a time.";
    LOG(ERROR) << ex.error_msg;
    throw ex;
  }
  Parser::Stmt* stmt = parse_trees.front().get();
  Parser::DDLStmt* ddl = dynamic_cast<Parser::DDLStmt*>(stmt);
  if (ddl != nullptr) {
    TMapDException ex;
    ex.error_msg = "Can only " + action + " SELECT statements.";
    LOG(ERROR) << ex.error_msg;
    throw ex;
  }
  auto dml = static_cast<Parser::DMLStmt*>(stmt);
  Analyzer::Query query;
  dml->analyze(cat, query);
  Planner::Optimizer optimizer(query, cat);
  return optimizer.optimize();
}

void MapDHandler::render(TRenderResult& _return,
                         const TSessionId session,
                         const std::string& query_str_in,
                         const std::string& render_type,
                         const std::string& nonce) {
  _return.total_time_ms = measure<>::execution([&]() {
    _return.nonce = nonce;
    if (!enable_rendering_) {
      TMapDException ex;
      ex.error_msg = "Backend rendering is disabled.";
      LOG(ERROR) << ex.error_msg;
      throw ex;
    }
    auto query_str = query_str_in;
    rapidjson::Document render_config;
    render_config.Parse(render_type.c_str());
    auto poly_data_desc = get_poly_render_data(render_config);
    bool is_projection_query = true;
    if (poly_data_desc) {
      if (poly_data_desc->HasMember("factsKey")) {
        is_projection_query = false;
        query_str = build_poly_render_query(*poly_data_desc);
      } else if (poly_data_desc->HasMember("polysKey")) {
        is_projection_query = false;
        query_str = transform_to_poly_render_query(query_str, *poly_data_desc);
      }
    }
    std::lock_guard<std::mutex> render_lock(render_mutex_);
    mapd_shared_lock<mapd_shared_mutex> read_lock(sessions_mutex_);
    auto session_it = get_session_it(session);
    auto session_info_ptr = session_it->second.get();
    try {
#ifdef HAVE_RAVM
      std::string query_ra;
      _return.execution_time_ms +=
          measure<>::execution([&]() { query_ra = parse_to_ra(query_str, *session_info_ptr); });
      render_rel_alg(_return, query_ra, query_str_in, *session_info_ptr, render_type, is_projection_query);
#else
#ifdef HAVE_CALCITE
      ParserWrapper pw{query_str};
      if (pw.is_select_explain || pw.is_other_explain || pw.is_ddl || pw.is_update_dml) {
        TMapDException ex;
        ex.error_msg = "Can only render SELECT statements.";
        LOG(ERROR) << ex.error_msg;
        throw ex;
      }
      auto root_plan = parse_to_plan(query_str, *session_info_ptr);
#else
      auto root_plan = parse_to_plan_legacy(query_str, *session_info_ptr, "render");
#endif  // HAVE_CALCITE
      CHECK(root_plan);
      std::unique_ptr<Planner::RootPlan> plan_ptr(root_plan);  // make sure it's deleted
      render_root_plan(_return, root_plan, query_str_in, *session_info_ptr, render_type, is_projection_query);
#endif  // HAVE_RAVM
    } catch (std::exception& e) {
      TMapDException ex;
      ex.error_msg = std::string("Exception: ") + e.what();
      LOG(ERROR) << ex.error_msg;
      throw ex;
    }
  });
  LOG(INFO) << "Total: " << _return.total_time_ms << " (ms), Execution: " << _return.execution_time_ms
            << " (ms), Render: " << _return.render_time_ms << " (ms)";
}

void MapDHandler::render_vega(TRenderResult& _return,
                              const TSessionId session,
                              const int64_t widget_id,
                              const std::string& vega_json,
                              const int compressionLevel,
                              const std::string& nonce) {
#ifdef HAVE_RENDERING
  if (leaf_aggregator_.leafCount() > 0) {
#ifdef HAVE_RAVM
    const auto session_info = MapDHandler::get_session(session);
    _return.image =
        leaf_aggregator_.render(session_info, vega_json, widget_id, compressionLevel, render_manager_.get());
    _return.nonce = nonce;
    return;
#else
    CHECK(false);
#endif  // HAVE_RAVM
  }
#endif  // HAVE_RENDERING
  _return.total_time_ms = measure<>::execution([&]() {
    _return.execution_time_ms = 0;
    _return.render_time_ms = 0;
    _return.nonce = nonce;
    if (!enable_rendering_) {
      TMapDException ex;
      ex.error_msg = "Backend rendering is disabled.";
      LOG(ERROR) << ex.error_msg;
      throw ex;
    }

    std::lock_guard<std::mutex> render_lock(render_mutex_);
    mapd_shared_lock<mapd_shared_mutex> read_lock(sessions_mutex_);
    auto session_it = get_session_it(session);
    auto session_info_ptr = session_it->second.get();

    const auto& cat = session_info_ptr->get_catalog();
    auto executor = Executor::getExecutor(cat.get_currentDB().dbId,
                                          jit_debug_ ? "/tmp" : "",
                                          jit_debug_ ? "mapdquery" : "",
                                          mapd_parameters_,
#ifdef HAVE_RENDERING
                                          render_manager_.get());
#else
                                          nullptr);
#endif

#ifdef HAVE_RENDERING
    try {
      const auto session_id = session_info_ptr->get_session_id();
      std::unique_ptr<RenderInfo> render_info(new RenderInfo(session_id, widget_id));

      std::string png_img_str;
      int64_t total_execution_time, total_render_time;
      std::tie(png_img_str, total_execution_time, total_render_time) = render_manager_->runRenderRequest(
          session_id,
          widget_id,
          vega_json,
          executor.get(),
          render_info.get(),
          [&](QueryRenderer::RenderQueryExecuteTimer& render_timer,
              Executor* executor,
              const std::string& query,
              const rapidjson::Value* dataObj,
              bool is_poly_query = false) {
            std::string query_str = query;

            // reset any layouts from prior executions
            render_info->vbo_result_query_data_layout = nullptr;

            if (is_poly_query) {
              render_info->do_render = false;  // do not perform the render post query
                                               // For polys, we call an explicit executor->renderPolygons()
            } else {
              render_info->do_render = true;
              if (!render_info->render_allocator_map_ptr) {
                const auto& catalog = session_info_ptr->get_catalog();

                // NOTE: the following code calculating block_size_x &
                // grid_size_x was copied from Executor::blockSize() &
                // Executor::gridSize() respectively.
                // Had to pull that code out as the Executor might not
                // have been properly initialized with a catalog yet,
                // which is required
                CHECK(catalog.get_dataMgr().cudaMgr_);
                const auto& dev_props = catalog.get_dataMgr().cudaMgr_->deviceProperties;

                size_t t_cuda_block_size = mapd_parameters_.cuda_block_size;
                size_t t_cuda_grid_size = mapd_parameters_.cuda_grid_size;
                if (!t_cuda_block_size) {
                  t_cuda_block_size = dev_props.front().maxThreadsPerBlock;
                }

                if (!t_cuda_grid_size) {
                  t_cuda_grid_size = 2 * dev_props.front().numMPs;
                }

                render_info->render_allocator_map_ptr.reset(new RenderAllocatorMap(
                    catalog.get_dataMgr().cudaMgr_, render_manager_.get(), t_cuda_block_size, t_cuda_grid_size));
              }
            }

#ifdef HAVE_RAVM
            std::string query_ra;
            _return.execution_time_ms +=
                measure<>::execution([&]() { query_ra = parse_to_ra(query_str, *session_info_ptr); });
            auto rtnData = execute_render_rel_alg(render_timer,
                                                  query_ra,
                                                  query_str,
                                                  *session_info_ptr,
                                                  executor,
                                                  dataObj,
                                                  render_info.get(),
                                                  is_poly_query);
#else
#ifdef HAVE_CALCITE
            ParserWrapper pw{query_str};
            if (pw.is_select_explain || pw.is_other_explain || pw.is_ddl || pw.is_update_dml) {
              std::runtime_error ex("Can only render SELECT statements.");
              LOG(ERROR) << ex.what();
              throw ex;
            }
            auto root_plan = parse_to_plan(query_str, *session_info_ptr);
#else
            auto root_plan = parse_to_plan_legacy(query_str, *session_info_ptr, "render");
#endif  // HAVE_CALCITE
            CHECK(root_plan);
            std::unique_ptr<Planner::RootPlan> plan_ptr(root_plan);  // make sure it's deleted

            auto rtnData = execute_render_root_plan(render_timer,
                                                    root_plan,
                                                    query_str,
                                                    *session_info_ptr,
                                                    executor,
                                                    dataObj,
                                                    render_info.get(),
                                                    is_poly_query);
#endif  // HAVE_RAVM

            auto& usedTables = std::get<3>(rtnData);
            CHECK(usedTables.size() > 0);

            if (is_poly_query) {
            } else if (render_info->render_allocator_map_ptr && render_info->vbo_result_query_data_layout) {
              render_info->render_allocator_map_ptr->setDataLayout(render_info->vbo_result_query_data_layout);
            }

            return std::make_tuple(std::move(std::get<0>(rtnData)),
                                   std::move(std::get<1>(rtnData)),
                                   std::get<2>(rtnData),
                                   std::move(usedTables),
                                   render_info->vbo_result_query_data_layout,
                                   render_info->ubo_result_query_data_layout);
          },
          compressionLevel,
          true);

      _return.execution_time_ms = total_execution_time;
      _return.render_time_ms = total_render_time;
      _return.image = png_img_str;
    } catch (std::exception& e) {
      TMapDException ex;
      ex.error_msg = std::string("Exception: ") + e.what();
      LOG(ERROR) << ex.error_msg;
      throw ex;
    }
#endif  // HAVE_RENDERING
  });
  LOG(INFO) << "Total: " << _return.total_time_ms << " (ms), Total Execution: " << _return.execution_time_ms
            << " (ms), Total Render: " << _return.render_time_ms << " (ms)";
}

void MapDHandler::create_frontend_view(const TSessionId session,
                                       const std::string& view_name,
                                       const std::string& view_state,
                                       const std::string& image_hash,
                                       const std::string& view_metadata) {
  check_read_only("create_frontend_view");
  const auto session_info = get_session(session);
  auto& cat = session_info.get_catalog();
  FrontendViewDescriptor vd;
  vd.viewName = view_name;
  vd.viewState = view_state;
  vd.imageHash = image_hash;
  vd.viewMetadata = view_metadata;
  vd.userId = session_info.get_currentUser().userId;

  try {
    cat.createFrontendView(vd);
  } catch (const std::exception& e) {
    TMapDException ex;
    ex.error_msg = std::string("Exception: ") + e.what();
    LOG(ERROR) << ex.error_msg;
    throw ex;
  }
}

void MapDHandler::delete_frontend_view(const TSessionId session, const std::string& view_name) {
  check_read_only("delete_frontend_view");
  const auto session_info = get_session(session);
  auto& cat = session_info.get_catalog();
  auto vd = cat.getMetadataForFrontendView(std::to_string(session_info.get_currentUser().userId), view_name);
  if (!vd) {
    TMapDException ex;
    ex.error_msg = "View " + view_name + " doesn't exist";
    LOG(ERROR) << ex.error_msg;
    throw ex;
  }
  try {
    cat.deleteMetadataForFrontendView(std::to_string(session_info.get_currentUser().userId), view_name);
  } catch (const std::exception& e) {
    TMapDException ex;
    ex.error_msg = std::string("Exception: ") + e.what();
    LOG(ERROR) << ex.error_msg;
    throw ex;
  }
}

void MapDHandler::create_link(std::string& _return,
                              const TSessionId session,
                              const std::string& view_state,
                              const std::string& view_metadata) {
  // check_read_only("create_link");
  const auto session_info = get_session(session);
  auto& cat = session_info.get_catalog();

  LinkDescriptor ld;
  ld.userId = session_info.get_currentUser().userId;
  ld.viewState = view_state;
  ld.viewMetadata = view_metadata;

  try {
    _return = cat.createLink(ld, 6);
  } catch (const std::exception& e) {
    TMapDException ex;
    ex.error_msg = std::string("Exception: ") + e.what();
    LOG(ERROR) << ex.error_msg;
    throw ex;
  }
}

std::string MapDHandler::sanitize_name(const std::string& name) {
  boost::regex invalid_chars{R"([^0-9a-z_])", boost::regex::extended | boost::regex::icase};

  std::string col_name = boost::regex_replace(name, invalid_chars, "");
  if (reserved_keywords.find(boost::to_upper_copy<std::string>(col_name)) != reserved_keywords.end()) {
    col_name += "_";
  }
  return col_name;
}

TColumnType MapDHandler::create_array_column(const TDatumType::type type, const std::string& name) {
  TColumnType ct;
  ct.col_name = name;
  ct.col_type.type = type;
  ct.col_type.is_array = true;
  return ct;
}

void MapDHandler::check_geospatial_files(const boost::filesystem::path file_path) {
  const std::list<std::string> shp_ext{".shp", ".shx", ".dbf"};
  if (std::find(shp_ext.begin(), shp_ext.end(), boost::algorithm::to_lower_copy(file_path.extension().string())) !=
      shp_ext.end()) {
    for (auto ext : shp_ext) {
      auto aux_file = file_path;
      if (!boost::filesystem::exists(aux_file.replace_extension(boost::algorithm::to_upper_copy(ext))) &&
          !boost::filesystem::exists(aux_file.replace_extension(ext))) {
        throw std::runtime_error("required file for shapefile does not exist: " + aux_file.filename().string());
      }
    }
  }
}

void MapDHandler::create_table(const TSessionId session,
                               const std::string& table_name,
                               const TRowDescriptor& rd,
                               const TTableType::type table_type) {
  check_read_only("create_table");

  if (table_name != sanitize_name(table_name)) {
    TMapDException ex;
    ex.error_msg = "Invalid characters in table name: " + table_name;
    LOG(ERROR) << ex.error_msg;
    throw ex;
  }

  auto rds = rd;

  if (table_type == TTableType::POLYGON) {
    rds.push_back(create_array_column(TDatumType::DOUBLE, MAPD_GEO_PREFIX + "coords"));
    rds.push_back(create_array_column(TDatumType::INT, MAPD_GEO_PREFIX + "indices"));
    rds.push_back(create_array_column(TDatumType::INT, MAPD_GEO_PREFIX + "linedrawinfo"));
    rds.push_back(create_array_column(TDatumType::INT, MAPD_GEO_PREFIX + "polydrawinfo"));
  }

  std::string stmt{"CREATE TABLE " + table_name};
  std::vector<std::string> col_stmts;

  for (auto col : rds) {
    if (col.col_name != sanitize_name(col.col_name)) {
      TMapDException ex;
      ex.error_msg = "Invalid characters in column name: " + col.col_name;
      LOG(ERROR) << ex.error_msg;
      throw ex;
    }
    if (col.col_type.type == TDatumType::INTERVAL_DAY_TIME || col.col_type.type == TDatumType::INTERVAL_YEAR_MONTH) {
      TMapDException ex;
      ex.error_msg = "Unsupported type: " + thrift_to_name(col.col_type) + " for column: " + col.col_name;
      LOG(ERROR) << ex.error_msg;
      throw ex;
    }
    // if no precision or scale passed in set to default 14,7
    if (col.col_type.precision == 0 && col.col_type.precision == 0) {
      col.col_type.precision = 14;
      col.col_type.scale = 7;
    }

    std::string col_stmt;
    col_stmt.append(col.col_name + " " + thrift_to_name(col.col_type) + " ");

    // As of 2016-06-27 the Immerse v1 frontend does not explicitly set the
    // `nullable` argument, leading this to default to false. Uncomment for v2.
    // if (!col.col_type.nullable) col_stmt.append("NOT NULL ");

    if (thrift_to_encoding(col.col_type.encoding) != kENCODING_NONE) {
      col_stmt.append("ENCODING " + thrift_to_encoding_name(col.col_type) + " ");
    }
    // deal with special case of non DICT encoded strings
    if (thrift_to_encoding(col.col_type.encoding) == kENCODING_NONE && col.col_type.type == TDatumType::STR) {
      col_stmt.append("ENCODING NONE");
    }
    col_stmts.push_back(col_stmt);
  }

  stmt.append(" (" + boost::algorithm::join(col_stmts, ", ") + ");");

  TQueryResult ret;
  sql_execute(ret, session, stmt, true, "", -1);
}

void MapDHandler::import_table(const TSessionId session,
                               const std::string& table_name,
                               const std::string& file_name,
                               const TCopyParams& cp) {
  check_read_only("import_table");
  LOG(INFO) << "import_table " << table_name << " from " << file_name;
  const auto session_info = get_session(session);
  auto& cat = session_info.get_catalog();

  const TableDescriptor* td = cat.getMetadataForTable(table_name);
  if (td == nullptr) {
    TMapDException ex;
    ex.error_msg = "Table " + table_name + " does not exist.";
    LOG(ERROR) << ex.error_msg;
    throw ex;
  }

  auto file_path = import_path_ / std::to_string(session) / boost::filesystem::path(file_name).filename();
  if (!boost::filesystem::exists(file_path)) {
    TMapDException ex;
    ex.error_msg = "File does not exist: " + file_path.filename().string();
    LOG(ERROR) << ex.error_msg << " at " << file_path.string();
    throw ex;
  }

  Importer_NS::CopyParams copy_params = thrift_to_copyparams(cp);

  // TODO(andrew): add delimiter detection to Importer
  if (copy_params.delimiter == '\0') {
    copy_params.delimiter = ',';
    if (boost::filesystem::extension(file_path) == ".tsv") {
      copy_params.delimiter = '\t';
    }
  }

  try {
    std::unique_ptr<Importer_NS::Importer> importer;
    if (leaf_aggregator_.leafCount() > 0) {
      importer.reset(new Importer_NS::Importer(
          new DistributedLoader(session_info, td, &leaf_aggregator_), file_path.string(), copy_params));
    } else {
      importer.reset(new Importer_NS::Importer(cat, td, file_path.string(), copy_params));
    }
    auto ms = measure<>::execution([&]() { importer->import(); });
    std::cout << "Total Import Time: " << (double)ms / 1000.0 << " Seconds." << std::endl;
  } catch (const std::exception& e) {
    TMapDException ex;
    ex.error_msg = "Exception: " + std::string(e.what());
    LOG(ERROR) << ex.error_msg;
    throw ex;
  }
}

void MapDHandler::import_geo_table(const TSessionId session,
                                   const std::string& table_name,
                                   const std::string& file_name_in,
                                   const TCopyParams& cp,
                                   const TRowDescriptor& row_desc) {
  check_read_only("import_table");
  const auto session_info = get_session(session);
  auto& cat = session_info.get_catalog();

  // Assume relative paths are relative to data_path / mapd_import / <session>
  std::string file_name{file_name_in};
  if (!boost::filesystem::path(file_name).is_absolute()) {
    auto file_path = import_path_ / std::to_string(session) / boost::filesystem::path(file_name).filename();
    file_name = file_path.string();
  }

  LOG(INFO) << "import_geo_table " << table_name << " from " << file_name;

  auto file_path = boost::filesystem::path(file_name);
  if (!boost::filesystem::exists(file_path)) {
    TMapDException ex;
    ex.error_msg = "File does not exist: " + file_path.filename().string();
    LOG(ERROR) << ex.error_msg << " at " << file_path.string();
    throw ex;
  }
  try {
    check_geospatial_files(file_path);
  } catch (const std::exception& e) {
    TMapDException ex;
    ex.error_msg = "import_geo_table error: " + std::string(e.what());
    LOG(ERROR) << ex.error_msg;
    throw ex;
  }

  TRowDescriptor rd;
  if (cat.getMetadataForTable(table_name) == nullptr) {
    TDetectResult cds;
    TCopyParams cp;
    cp.table_type = TTableType::POLYGON;
    detect_column_types(cds, session, file_name_in, cp);
    create_table(session, table_name, cds.row_set.row_desc, TTableType::POLYGON);
    rd = cds.row_set.row_desc;
  } else if (row_desc.size() > 0) {
    rd = row_desc;
  } else {
    TMapDException ex;
    ex.error_msg =
        "Could not append file " + file_path.filename().string() + " to " + table_name + ": not currently supported.";
    LOG(ERROR) << ex.error_msg;
    throw ex;
  }

  std::map<std::string, std::string> colname_to_src;
  for (auto r : rd) {
    colname_to_src[r.col_name] = r.src_name.length() > 0 ? r.src_name : sanitize_name(r.src_name);
  }

  const TableDescriptor* td = cat.getMetadataForTable(table_name);
  if (td == nullptr) {
    TMapDException ex;
    ex.error_msg = "Table " + table_name + " does not exist.";
    LOG(ERROR) << ex.error_msg;
    throw ex;
  }

  Importer_NS::CopyParams copy_params = thrift_to_copyparams(cp);

  try {
    std::unique_ptr<Importer_NS::Importer> importer;
    if (leaf_aggregator_.leafCount() > 0) {
      importer.reset(new Importer_NS::Importer(
          new DistributedLoader(session_info, td, &leaf_aggregator_), file_path.string(), copy_params));
    } else {
      importer.reset(new Importer_NS::Importer(cat, td, file_path.string(), copy_params));
    }
    auto ms = measure<>::execution([&]() { importer->importGDAL(colname_to_src); });
    std::cout << "Total Import Time: " << (double)ms / 1000.0 << " Seconds." << std::endl;
  } catch (const std::exception& e) {
    TMapDException ex;
    ex.error_msg = std::string("import_geo_table failed: ") + e.what();
    LOG(ERROR) << ex.error_msg;
    throw ex;
  }
}

void MapDHandler::import_table_status(TImportStatus& _return, const TSessionId session, const std::string& import_id) {
  LOG(INFO) << "import_table_status " << import_id;
  auto is = Importer_NS::Importer::get_import_status(import_id);
  _return.elapsed = is.elapsed.count();
  _return.rows_completed = is.rows_completed;
  _return.rows_estimated = is.rows_estimated;
  _return.rows_rejected = is.rows_rejected;
}

void MapDHandler::start_heap_profile(const TSessionId session) {
  const auto session_info = get_session(session);
#ifdef HAVE_PROFILER
  if (IsHeapProfilerRunning()) {
    throw_profile_exception("Profiler already started");
  }
  HeapProfilerStart("mapd");
#else
  throw_profile_exception("Profiler not enabled");
#endif  // HAVE_PROFILER
}

void MapDHandler::stop_heap_profile(const TSessionId session) {
  const auto session_info = get_session(session);
#ifdef HAVE_PROFILER
  if (!IsHeapProfilerRunning()) {
    throw_profile_exception("Profiler not running");
  }
  HeapProfilerStop();
#else
  throw_profile_exception("Profiler not enabled");
#endif  // HAVE_PROFILER
}

void MapDHandler::get_heap_profile(std::string& profile, const TSessionId session) {
  const auto session_info = get_session(session);
#ifdef HAVE_PROFILER
  if (!IsHeapProfilerRunning()) {
    throw_profile_exception("Profiler not running");
  }
  auto profile_buff = GetHeapProfile();
  profile = profile_buff;
  free(profile_buff);
#else
  throw_profile_exception("Profiler not enabled");
#endif  // HAVE_PROFILER
}

SessionMap::iterator MapDHandler::get_session_it(const TSessionId session) {
  auto session_it = sessions_.find(session);
  if (session_it == sessions_.end()) {
    TMapDException ex;
    ex.error_msg = "Session not valid.";
    LOG(ERROR) << ex.error_msg;
    throw ex;
  }
  session_it->second->update_time();
  return session_it;
}

Catalog_Namespace::SessionInfo MapDHandler::get_session(const TSessionId session) {
  mapd_shared_lock<mapd_shared_mutex> read_lock(sessions_mutex_);
  return *get_session_it(session)->second;
}

void MapDHandler::set_execution_mode_nolock(Catalog_Namespace::SessionInfo* session_ptr,
                                            const TExecuteMode::type mode) {
  const std::string& user_name = session_ptr->get_currentUser().userName;
  switch (mode) {
    case TExecuteMode::GPU:
      if (cpu_mode_only_) {
        TMapDException e;
        e.error_msg = "Cannot switch to GPU mode in a server started in CPU-only mode.";
        throw e;
      }
      session_ptr->set_executor_device_type(ExecutorDeviceType::GPU);
      LOG(INFO) << "User " << user_name << " sets GPU mode.";
      break;
    case TExecuteMode::CPU:
      session_ptr->set_executor_device_type(ExecutorDeviceType::CPU);
      LOG(INFO) << "User " << user_name << " sets CPU mode.";
      break;
    case TExecuteMode::HYBRID:
      if (cpu_mode_only_) {
        TMapDException e;
        e.error_msg = "Cannot switch to Hybrid mode in a server started in CPU-only mode.";
        throw e;
      }
      session_ptr->set_executor_device_type(ExecutorDeviceType::Hybrid);
      LOG(INFO) << "User " << user_name << " sets HYBRID mode.";
  }
}

#ifdef HAVE_RAVM
void MapDHandler::execute_rel_alg(TQueryResult& _return,
                                  const std::string& query_ra,
                                  const bool column_format,
                                  const Catalog_Namespace::SessionInfo& session_info,
                                  const ExecutorDeviceType executor_device_type,
                                  const int32_t first_n,
                                  const bool just_explain,
                                  const bool just_validate) const {
  const auto& cat = session_info.get_catalog();
  CompilationOptions co = {executor_device_type, true, ExecutorOptLevel::Default, g_enable_dynamic_watchdog};
  ExecutionOptions eo = {false,
                         allow_multifrag_,
                         just_explain,
                         allow_loop_joins_ || just_validate,
                         g_enable_watchdog,
                         jit_debug_,
                         just_validate,
                         g_enable_dynamic_watchdog,
                         g_dynamic_watchdog_time_limit};
  auto executor = Executor::getExecutor(
      cat.get_currentDB().dbId, jit_debug_ ? "/tmp" : "", jit_debug_ ? "mapdquery" : "", mapd_parameters_, nullptr);
  RelAlgExecutor ra_executor(executor.get(), cat);
  ExecutionResult result{ResultRows({}, {}, nullptr, nullptr, {}, executor_device_type), {}};
  _return.execution_time_ms +=
      measure<>::execution([&]() { result = ra_executor.executeRelAlgQuery(query_ra, co, eo, nullptr); });
  // reduce execution time by the time spent during queue waiting
  _return.execution_time_ms -= result.getRows().getQueueTime();
  if (just_explain) {
    convert_explain(_return, result.getRows(), column_format);
  } else {
    convert_rows(_return, result.getTargetsMeta(), result.getRows(), column_format, first_n);
  }
}
#endif  // HAVE_RAVM

void MapDHandler::execute_root_plan(TQueryResult& _return,
                                    const Planner::RootPlan* root_plan,
                                    const bool column_format,
                                    const Catalog_Namespace::SessionInfo& session_info,
                                    const ExecutorDeviceType executor_device_type,
                                    const int32_t first_n) const {
  auto executor = Executor::getExecutor(root_plan->get_catalog().get_currentDB().dbId,
                                        jit_debug_ ? "/tmp" : "",
                                        jit_debug_ ? "mapdquery" : "",
                                        mapd_parameters_,
#ifdef HAVE_RENDERING
                                        render_manager_.get());
#else
                                        nullptr);
#endif
  ResultRows results({}, {}, nullptr, nullptr, {}, executor_device_type);
  _return.execution_time_ms += measure<>::execution([&]() {
    results = executor->execute(root_plan,
                                session_info,
                                true,
                                executor_device_type,
                                ExecutorOptLevel::Default,
                                allow_multifrag_,
                                allow_loop_joins_);
  });
  // reduce execution time by the time spent during queue waiting
  _return.execution_time_ms -= results.getQueueTime();
  if (root_plan->get_plan_dest() == Planner::RootPlan::Dest::kEXPLAIN) {
    convert_explain(_return, results, column_format);
    return;
  }
  const auto plan = root_plan->get_plan();
  CHECK(plan);
  const auto& targets = plan->get_targetlist();
  convert_rows(_return, getTargetMetaInfo(targets), results, column_format, -1);
}

#ifdef HAVE_RENDERING
std::tuple<std::shared_ptr<ResultRows>,
           std::vector<TargetMetaInfo>,
           int64_t,
           std::vector<std::pair<decltype(TableDescriptor::tableId), decltype(TableDescriptor::tableName)>>>
MapDHandler::execute_render_root_plan(QueryRenderer::RenderQueryExecuteTimer& render_timer,
                                      Planner::RootPlan* root_plan,
                                      const std::string& query_str,
                                      const Catalog_Namespace::SessionInfo& session_info,
                                      Executor* executor,
                                      const rapidjson::Value* data_desc,
                                      RenderInfo* render_info,
                                      bool is_poly_query = false) {
  if (!is_poly_query) {
    root_plan->set_plan_dest(Planner::RootPlan::Dest::kRENDER);
  }

  auto clock_begin = timer_start();
  auto results = executor->execute(root_plan,
                                   session_info,
                                   true,
                                   session_info.get_executor_device_type(),
                                   ExecutorOptLevel::Default,
                                   allow_multifrag_,
                                   false,
                                   render_info);

  auto plan = root_plan->get_plan();
  CHECK(plan);
  const auto& targets = plan->get_targetlist();
  auto targetMetaInfo = getTargetMetaInfo(targets);

  // reduce execution time by the time spent during queue waiting
  int64_t execute_time_ms = timer_stop(clock_begin) - results.getQueueTime();
  render_timer.queue_time_ms += results.getQueueTime();

  auto tableId = root_plan->get_result_table_id();
  auto tableName = session_info.get_catalog().getMetadataForTable(tableId)->tableName;

  bool is_projection_query = true;
  while (plan) {
    if (dynamic_cast<const Planner::AggPlan*>(plan) || dynamic_cast<const Planner::Join*>(plan)) {
      is_projection_query = false;
      break;
    }
    plan = plan->get_child_plan();
  }

  if (is_poly_query) {
    auto rendered_results = executor->renderPolygons(query_str,
                                                     results,
                                                     targetMetaInfo,
                                                     session_info,
                                                     1,
                                                     *data_desc,
                                                     nullptr,
                                                     is_projection_query,
                                                     tableName,
                                                     render_info);
    render_timer.queue_time_ms += rendered_results.getQueueTime();
    render_timer.render_time_ms += rendered_results.getRenderTime();
  } else {
    // reduce execution time by time spend rendering
    execute_time_ms -= results.getRenderTime();
    render_timer.render_time_ms += results.getRenderTime();
  }

  render_timer.execution_time_ms += execute_time_ms;

  return std::make_tuple(
      std::shared_ptr<ResultRows>(new ResultRows(results)),
      std::move(targetMetaInfo),
      execute_time_ms,
      std::vector<std::pair<decltype(TableDescriptor::tableId), std::string>>({std::make_pair(tableId, tableName)}));
}
#endif  // HAVE_RENDERING

void MapDHandler::render_root_plan(TRenderResult& _return,
                                   Planner::RootPlan* root_plan,
                                   const std::string& query_str,
                                   const Catalog_Namespace::SessionInfo& session_info,
                                   const std::string& render_type,
                                   const bool is_projection_query) {
  rapidjson::Document render_config;
  render_config.Parse(render_type.c_str());

  auto poly_data_desc = get_poly_render_data(render_config);
  std::unique_ptr<RenderInfo> render_info(
      new RenderInfo(session_info.get_session_id(), 1, render_type, (poly_data_desc == nullptr)));
  if (!poly_data_desc) {
    root_plan->set_plan_dest(Planner::RootPlan::Dest::kRENDER);
  }
  auto executor = Executor::getExecutor(root_plan->get_catalog().get_currentDB().dbId,
                                        jit_debug_ ? "/tmp" : "",
                                        jit_debug_ ? "mapdquery" : "",
                                        mapd_parameters_,
#ifdef HAVE_RENDERING
                                        render_manager_.get());
#else
                                        nullptr);
#endif

  auto clock_begin = timer_start();
  auto results = executor->execute(root_plan,
                                   session_info,
                                   true,
                                   session_info.get_executor_device_type(),
                                   ExecutorOptLevel::Default,
                                   allow_multifrag_,
                                   false,
                                   render_info.get());
  if (poly_data_desc) {
#ifdef HAVE_RENDERING
    const auto plan = root_plan->get_plan();
    CHECK(plan);
    const auto& targets = plan->get_targetlist();
    auto rendered_results = executor->renderPolygons(query_str,
                                                     results,
                                                     getTargetMetaInfo(targets),
                                                     session_info,
                                                     1,
                                                     *poly_data_desc,
                                                     &render_type,
                                                     is_projection_query);
    _return.execution_time_ms =
        timer_stop(clock_begin) - rendered_results.getQueueTime() - rendered_results.getRenderTime();
    _return.render_time_ms = rendered_results.getRenderTime();
    _return.image = image_from_rendered_rows(rendered_results);
    return;
#else
    CHECK(false);
#endif
  }
  // reduce execution time by the time spent during queue waiting
  _return.execution_time_ms = timer_stop(clock_begin) - results.getQueueTime() - results.getRenderTime();
  _return.render_time_ms = results.getRenderTime();
  _return.image = image_from_rendered_rows(results);
}

#ifdef HAVE_RAVM

#ifdef HAVE_RENDERING
std::tuple<std::shared_ptr<ResultRows>,
           std::vector<TargetMetaInfo>,
           int64_t,
           std::vector<std::pair<decltype(TableDescriptor::tableId), decltype(TableDescriptor::tableName)>>>
MapDHandler::execute_render_rel_alg(QueryRenderer::RenderQueryExecuteTimer& render_timer,
                                    const std::string& query_ra,
                                    const std::string& query_str,
                                    const Catalog_Namespace::SessionInfo& session_info,
                                    Executor* executor,
                                    const rapidjson::Value* data_desc,
                                    RenderInfo* render_info,
                                    bool is_poly_query = false) {
  const auto& cat = session_info.get_catalog();

  RelAlgExecutor ra_executor(executor, cat);
  auto clock_begin = timer_start();
  CompilationOptions co = {
      session_info.get_executor_device_type(), true, ExecutorOptLevel::Default, g_enable_dynamic_watchdog};
  ExecutionOptions eo = {false,
                         MapDHandler::allow_multifrag_,
                         false,
                         MapDHandler::allow_loop_joins_,
                         g_enable_watchdog,
                         MapDHandler::jit_debug_,
                         false,
                         g_enable_dynamic_watchdog,
                         g_dynamic_watchdog_time_limit};
  const auto exe_result = ra_executor.executeRelAlgQuery(query_ra, co, eo, render_info);

  const auto& results = exe_result.getRows();

  // reduce execution time by the time spent during queue waiting
  int64_t execute_time_ms = timer_stop(clock_begin) - results.getQueueTime();
  render_timer.queue_time_ms += results.getQueueTime();

  auto tables = ra_executor.getScanTableNamesInRelAlgSeq();
  std::vector<std::pair<decltype(TableDescriptor::tableId), std::string>> rtn(tables.size());
  size_t cnt = 0;
  for (auto& tableName : tables) {
    rtn[cnt++] = std::make_pair(cat.getMetadataForTable(tableName)->tableId, std::move(tableName));
  }

  if (is_poly_query) {
    CHECK(is_poly_table(rtn[0].first, cat))
        << "Table: " << rtn[0].second << ", id: " << rtn[0].first
        << " is not a poly table but the query is part of a poly render query. sql: " << query_str;

    // TODO(croot): is the number of tables in a query enough to determine if its a projection query
    // or not?
    auto rendered_results = executor->renderPolygons(query_str,
                                                     results,
                                                     exe_result.getTargetsMeta(),
                                                     session_info,
                                                     1,
                                                     *data_desc,
                                                     nullptr,
                                                     tables.size() == 1,
                                                     rtn[0].second,
                                                     render_info);
    render_timer.render_time_ms += rendered_results.getRenderTime();
    render_timer.queue_time_ms += rendered_results.getQueueTime();
  } else {
    // reduce execution time by time spend rendering
    execute_time_ms -= results.getRenderTime();
    render_timer.render_time_ms += results.getRenderTime();
  }

  render_timer.execution_time_ms += execute_time_ms;

  return std::make_tuple(std::shared_ptr<ResultRows>(new ResultRows(results)),
                         std::move(exe_result.getTargetsMeta()),
                         execute_time_ms,
                         rtn);
}
#endif  // HAVE_RENDERING

void MapDHandler::render_rel_alg(TRenderResult& _return,
                                 const std::string& query_ra,
                                 const std::string& query_str,
                                 const Catalog_Namespace::SessionInfo& session_info,
                                 const std::string& render_type,
                                 const bool is_projection_query) {
  const auto& cat = session_info.get_catalog();
  auto executor = Executor::getExecutor(cat.get_currentDB().dbId,
                                        jit_debug_ ? "/tmp" : "",
                                        jit_debug_ ? "mapdquery" : "",
                                        mapd_parameters_,
#ifdef HAVE_RENDERING
                                        render_manager_.get());
#else
                                        nullptr);
#endif
  RelAlgExecutor ra_executor(executor.get(), cat);
  auto clock_begin = timer_start();
  CompilationOptions co = {
      session_info.get_executor_device_type(), true, ExecutorOptLevel::Default, g_enable_dynamic_watchdog};
  ExecutionOptions eo = {false,
                         allow_multifrag_,
                         false,
                         allow_loop_joins_,
                         g_enable_watchdog,
                         jit_debug_,
                         false,
                         g_enable_dynamic_watchdog,
                         g_dynamic_watchdog_time_limit};
  rapidjson::Document render_config;
  render_config.Parse(render_type.c_str());

  auto poly_data_desc = get_poly_render_data(render_config);

  std::unique_ptr<RenderInfo> render_info(
      new RenderInfo(session_info.get_session_id(), 1, render_type, (poly_data_desc == nullptr)));
  const auto exe_result = ra_executor.executeRelAlgQuery(query_ra, co, eo, render_info.get());
  const auto& results = exe_result.getRows();
  if (poly_data_desc) {
#ifdef HAVE_RENDERING
    auto rendered_results = executor->renderPolygons(query_str,
                                                     results,
                                                     exe_result.getTargetsMeta(),
                                                     session_info,
                                                     1,
                                                     *poly_data_desc,
                                                     &render_type,
                                                     is_projection_query);
    _return.execution_time_ms =
        timer_stop(clock_begin) - rendered_results.getQueueTime() - rendered_results.getRenderTime();
    _return.render_time_ms = rendered_results.getRenderTime();
    _return.image = image_from_rendered_rows(rendered_results);
    return;
#else
    CHECK(false);
#endif
  }
  // reduce execution time by the time spent during queue waiting
  _return.execution_time_ms = timer_stop(clock_begin) - results.getQueueTime() - results.getRenderTime();
  _return.render_time_ms = results.getRenderTime();
  _return.image = image_from_rendered_rows(results);
}
#endif  // HAVE_RAVM

std::vector<TargetMetaInfo> MapDHandler::getTargetMetaInfo(
    const std::vector<std::shared_ptr<Analyzer::TargetEntry>>& targets) const {
  std::vector<TargetMetaInfo> result;
  for (const auto target : targets) {
    CHECK(target);
    CHECK(target->get_expr());
    result.emplace_back(target->get_resname(), target->get_expr()->get_type_info());
  }
  return result;
}

TRowDescriptor MapDHandler::convert_target_metainfo(const std::vector<TargetMetaInfo>& targets) const {
  TRowDescriptor row_desc;
  TColumnType proj_info;
  size_t i = 0;
  for (const auto target : targets) {
    proj_info.col_name = target.get_resname();
    if (proj_info.col_name.empty()) {
      proj_info.col_name = "result_" + std::to_string(i + 1);
    }
    const auto& target_ti = target.get_type_info();
    proj_info.col_type.type = type_to_thrift(target_ti);
    proj_info.col_type.encoding = encoding_to_thrift(target_ti);
    proj_info.col_type.nullable = !target_ti.get_notnull();
    proj_info.col_type.is_array = target_ti.get_type() == kARRAY;
    proj_info.col_type.precision = target_ti.get_precision();
    proj_info.col_type.scale = target_ti.get_scale();
    proj_info.col_type.comp_param = target_ti.get_comp_param();
    row_desc.push_back(proj_info);
    ++i;
  }
  return row_desc;
}

template <class R>
void MapDHandler::convert_rows(TQueryResult& _return,
                               const std::vector<TargetMetaInfo>& targets,
                               const R& results,
                               const bool column_format,
                               const int32_t first_n) const {
  _return.row_set.row_desc = convert_target_metainfo(targets);
  int32_t fetched{0};
  if (column_format) {
    _return.row_set.is_columnar = true;
    std::vector<TColumn> tcolumns(results.colCount());
    while (first_n == -1 || fetched < first_n) {
      const auto crt_row = results.getNextRow(true, true);
      if (crt_row.empty()) {
        break;
      }
      ++fetched;
      for (size_t i = 0; i < results.colCount(); ++i) {
        const auto agg_result = crt_row[i];
        value_to_thrift_column(agg_result, targets[i].get_type_info(), tcolumns[i]);
      }
    }
    for (size_t i = 0; i < results.colCount(); ++i) {
      _return.row_set.columns.push_back(tcolumns[i]);
    }
  } else {
    _return.row_set.is_columnar = false;
    while (first_n == -1 || fetched < first_n) {
      const auto crt_row = results.getNextRow(true, true);
      if (crt_row.empty()) {
        break;
      }
      ++fetched;
      TRow trow;
      trow.cols.reserve(results.colCount());
      for (size_t i = 0; i < results.colCount(); ++i) {
        const auto agg_result = crt_row[i];
        trow.cols.push_back(value_to_thrift(agg_result, targets[i].get_type_info()));
      }
      _return.row_set.rows.push_back(trow);
    }
  }
}

TRowDescriptor MapDHandler::fixup_row_descriptor(const TRowDescriptor& row_desc,
                                                 const Catalog_Namespace::Catalog& cat) {
  TRowDescriptor fixedup_row_desc;
  for (const TColumnType& col_desc : row_desc) {
    auto fixedup_col_desc = col_desc;
    if (col_desc.col_type.encoding == TEncodingType::DICT && col_desc.col_type.comp_param > 0) {
      const auto dd = cat.getMetadataForDict(col_desc.col_type.comp_param, false);
      fixedup_col_desc.col_type.comp_param = dd->dictNBits;
    }
    fixedup_row_desc.push_back(fixedup_col_desc);
  }
  return fixedup_row_desc;
}

// create simple result set to return a single column result
void MapDHandler::create_simple_result(TQueryResult& _return,
                                       const ResultRows& results,
                                       const bool column_format,
                                       const std::string label) const {
  CHECK_EQ(size_t(1), results.rowCount());
  TColumnType proj_info;
  proj_info.col_name = label;
  proj_info.col_type.type = TDatumType::STR;
  proj_info.col_type.nullable = false;
  proj_info.col_type.is_array = false;
  _return.row_set.row_desc.push_back(proj_info);
  const auto crt_row = results.getNextRow(true, true);
  const auto tv = crt_row[0];
  CHECK(results.getNextRow(true, true).empty());
  const auto scalar_tv = boost::get<ScalarTargetValue>(&tv);
  CHECK(scalar_tv);
  const auto s_n = boost::get<NullableString>(scalar_tv);
  CHECK(s_n);
  const auto s = boost::get<std::string>(s_n);
  CHECK(s);
  if (column_format) {
    TColumn tcol;
    tcol.data.str_col.push_back(*s);
    tcol.nulls.push_back(false);
    _return.row_set.is_columnar = true;
    _return.row_set.columns.push_back(tcol);
  } else {
    TDatum explanation;
    explanation.val.str_val = *s;
    explanation.is_null = false;
    TRow trow;
    trow.cols.push_back(explanation);
    _return.row_set.is_columnar = false;
    _return.row_set.rows.push_back(trow);
  }
}

void MapDHandler::convert_explain(TQueryResult& _return, const ResultRows& results, const bool column_format) const {
  create_simple_result(_return, results, column_format, "Explanation");
}

void MapDHandler::convert_result(TQueryResult& _return, const ResultRows& results, const bool column_format) const {
  create_simple_result(_return, results, column_format, "Result");
}

void MapDHandler::sql_execute_impl(TQueryResult& _return,
                                   const Catalog_Namespace::SessionInfo& session_info,
                                   const std::string& query_str,
                                   const bool column_format,
                                   const std::string& nonce,
                                   const ExecutorDeviceType executor_device_type,
                                   const int32_t first_n) {
  _return.nonce = nonce;
  _return.execution_time_ms = 0;
  auto& cat = session_info.get_catalog();
  LOG(INFO) << query_str;
  _return.total_time_ms = measure<>::execution([&]() {
    SQLParser parser;
    std::list<std::unique_ptr<Parser::Stmt>> parse_trees;
    std::string last_parsed;
    int num_parse_errors = 0;
    Planner::RootPlan* root_plan{nullptr};
#ifdef HAVE_CALCITE
    try {
#ifdef HAVE_RAVM
      ParserWrapper pw{query_str};
      if (!pw.is_ddl && !pw.is_update_dml && !pw.is_other_explain) {
        std::string query_ra;
        _return.execution_time_ms += measure<>::execution([&]() { query_ra = parse_to_ra(query_str, session_info); });
        if (pw.is_select_calcite_explain) {
          // return the ra as the result
          convert_explain(_return, ResultRows(query_ra, 0), true);
          return;
        }
        execute_rel_alg(
            _return, query_ra, column_format, session_info, executor_device_type, first_n, pw.is_select_explain, false);
        return;
      }
#else
      std::unique_ptr<const Planner::RootPlan> plan_ptr;
      _return.execution_time_ms +=
          measure<>::execution([&]() { plan_ptr.reset(parse_to_plan(query_str, session_info)); });
      if (plan_ptr) {
        execute_root_plan(_return, plan_ptr.get(), column_format, session_info, executor_device_type, first_n);
        return;
      }
#endif  // HAVE_RAVM
      LOG(INFO) << "passing query to legacy processor";
    } catch (std::exception& e) {
      TMapDException ex;
      ex.error_msg = std::string("Exception: ") + e.what();
      LOG(ERROR) << ex.error_msg;
      throw ex;
    }
#endif  // HAVE_CALCITE
    try {
      // check for COPY TO stmt replace as required parser expects #~# markers
      const auto result = apply_copy_to_shim(query_str);
      num_parse_errors = parser.parse(result, parse_trees, last_parsed);
    } catch (std::exception& e) {
      TMapDException ex;
      ex.error_msg = std::string("Exception: ") + e.what();
      LOG(ERROR) << ex.error_msg;
      throw ex;
    }
    if (num_parse_errors > 0) {
      TMapDException ex;
      ex.error_msg = "Syntax error at: " + last_parsed;
      LOG(ERROR) << ex.error_msg;
      throw ex;
    }
    for (const auto& stmt : parse_trees) {
      try {
        auto select_stmt = dynamic_cast<Parser::SelectStmt*>(stmt.get());
        if (!select_stmt) {
          check_read_only("Non-SELECT statements");
        }
        Parser::DDLStmt* ddl = dynamic_cast<Parser::DDLStmt*>(stmt.get());
        Parser::ExplainStmt* explain_stmt = nullptr;
        if (ddl != nullptr)
          explain_stmt = dynamic_cast<Parser::ExplainStmt*>(ddl);
        if (ddl != nullptr && explain_stmt == nullptr) {
          const auto copy_stmt = dynamic_cast<Parser::CopyTableStmt*>(ddl);
          if (copy_stmt && leaf_aggregator_.leafCount() > 0) {
            _return.execution_time_ms +=
                measure<>::execution([&]() { execute_distributed_copy_statement(copy_stmt, session_info); });
          } else {
            _return.execution_time_ms += measure<>::execution([&]() { ddl->execute(session_info); });
          }
          // check if it was a copy statement gather response message
          if (copy_stmt) {
            convert_result(_return, ResultRows(*copy_stmt->return_message.get(), 0), true);
          }
        } else {
          const Parser::DMLStmt* dml;
          if (explain_stmt != nullptr)
            dml = explain_stmt->get_stmt();
          else
            dml = dynamic_cast<Parser::DMLStmt*>(stmt.get());
          Analyzer::Query query;
          dml->analyze(cat, query);
          Planner::Optimizer optimizer(query, cat);
          root_plan = optimizer.optimize();
          CHECK(root_plan);
          std::unique_ptr<Planner::RootPlan> plan_ptr(root_plan);  // make sure it's deleted
          if (explain_stmt != nullptr) {
            root_plan->set_plan_dest(Planner::RootPlan::Dest::kEXPLAIN);
          }
          execute_root_plan(_return, root_plan, column_format, session_info, executor_device_type, first_n);
        }
      } catch (std::exception& e) {
        TMapDException ex;
        ex.error_msg = std::string("Exception: ") + e.what();
        LOG(ERROR) << ex.error_msg;
        throw ex;
      }
    }
  });
  LOG(INFO) << "Total: " << _return.total_time_ms << " (ms), Execution: " << _return.execution_time_ms << " (ms)";
}

void MapDHandler::execute_distributed_copy_statement(Parser::CopyTableStmt* copy_stmt,
                                                     const Catalog_Namespace::SessionInfo& session_info) {
  auto importer_factory = [&session_info, this](const Catalog_Namespace::Catalog& catalog,
                                                const TableDescriptor* td,
                                                const std::string& file_path,
                                                const Importer_NS::CopyParams& copy_params) {
    return boost::make_unique<Importer_NS::Importer>(
        new DistributedLoader(session_info, td, &leaf_aggregator_), file_path, copy_params);
  };
  copy_stmt->execute(session_info, importer_factory);
}

#ifdef HAVE_CALCITE
Planner::RootPlan* MapDHandler::parse_to_plan(const std::string& query_str,
                                              const Catalog_Namespace::SessionInfo& session_info) {
  auto& cat = session_info.get_catalog();
  ParserWrapper pw{query_str};
  // if this is a calcite select or explain select run in calcite
  if (!pw.is_ddl && !pw.is_update_dml && !pw.is_other_explain) {
    const std::string actual_query{pw.is_select_explain || pw.is_select_calcite_explain ? pw.actual_query : query_str};
    const auto query_ra = calcite_->process(session_info.get_currentUser().userName,
                                            session_info.get_currentUser().passwd,
                                            cat.get_currentDB().dbName,
                                            legacy_syntax_ ? pg_shim(actual_query) : actual_query,
                                            legacy_syntax_,
                                            pw.is_select_calcite_explain);
    auto root_plan = translate_query(query_ra, cat);
    CHECK(root_plan);
    if (pw.is_select_explain) {
      root_plan->set_plan_dest(Planner::RootPlan::Dest::kEXPLAIN);
    }
    return root_plan;
  }
  return nullptr;
}

std::string MapDHandler::parse_to_ra(const std::string& query_str, const Catalog_Namespace::SessionInfo& session_info) {
  ParserWrapper pw{query_str};
  const std::string actual_query{pw.is_select_explain || pw.is_select_calcite_explain ? pw.actual_query : query_str};
  auto& cat = session_info.get_catalog();
  if (!pw.is_ddl && !pw.is_update_dml && !pw.is_other_explain) {
    return calcite_->process(session_info.get_currentUser().userName,
                             session_info.get_currentUser().passwd,
                             cat.get_currentDB().dbName,
                             legacy_syntax_ ? pg_shim(actual_query) : actual_query,
                             legacy_syntax_,
                             pw.is_select_calcite_explain);
  }
  return "";
}
#endif  // HAVE_CALCITE

void MapDHandler::execute_first_step(TStepResult& _return, const TPendingQuery& pending_query) {
#ifdef HAVE_RAVM
  try {
    const auto first_step_result = PendingExecutionClosure::executeNextStep(
        pending_query.id,
        column_ranges_from_thrift(pending_query.column_ranges),
        string_dictionary_generations_from_thrift(pending_query.dictionary_generations),
        table_generations_from_thrift(pending_query.table_generations));
    const auto& result_rows = first_step_result.result.getRows();
    auto result_set = result_rows.getResultSet();
    if (!result_set) {
      QueryMemoryDescriptor empty_query_mem_desc{};
      result_set = std::make_shared<ResultSet>(
          result_rows.getTargetInfos(), ExecutorDeviceType::CPU, empty_query_mem_desc, nullptr, nullptr);
    }
    _return.serialized_rows = result_set->serialize();
    _return.execution_finished = first_step_result.is_outermost_query;
    _return.merge_type = first_step_result.merge_type == MergeType::Reduce ? TMergeType::REDUCE : TMergeType::UNION;
    _return.sharded = true;  // TODO(alex)
    _return.row_desc = convert_target_metainfo(first_step_result.result.getTargetsMeta());
    _return.node_id = first_step_result.node_id;
  } catch (std::exception& e) {
    TMapDException ex;
    ex.error_msg = std::string("Exception: ") + e.what();
    LOG(ERROR) << ex.error_msg;
    throw ex;
  }
#else
  CHECK(false);
#endif  // HAVE_RAVM
}

void MapDHandler::start_query(TPendingQuery& _return,
                              const TSessionId session,
                              const std::string& query_ra,
                              const bool just_explain) {
#ifdef HAVE_RAVM
  const auto session_info = get_session(session);
  const auto& cat = session_info.get_catalog();
  CompilationOptions co = {
      session_info.get_executor_device_type(), true, ExecutorOptLevel::Default, g_enable_dynamic_watchdog};
  ExecutionOptions eo = {false,
                         allow_multifrag_,
                         just_explain,
                         allow_loop_joins_,
                         g_enable_watchdog,
                         jit_debug_,
                         false,
                         g_enable_dynamic_watchdog,
                         g_dynamic_watchdog_time_limit};
  RelAlgExecutionOptions ra_eo{co, eo, nullptr, 0};
  auto executor = Executor::getExecutor(
      cat.get_currentDB().dbId, jit_debug_ ? "/tmp" : "", jit_debug_ ? "mapdquery" : "", mapd_parameters_, nullptr);
  auto ra_executor = boost::make_unique<RelAlgExecutor>(executor.get(), cat);
  const auto ra = deserialize_ra_dag(query_ra, cat, ra_executor.get());
  auto closure = PendingExecutionClosure::create(ra, ra_executor, cat, ra_eo);
  _return.id = closure->getId();
  _return.column_ranges = column_ranges_to_thrift(closure->getColRangeCache());
  _return.dictionary_generations = string_dictionary_generations_to_thrift(closure->getStringDictionaryGenerations());
  _return.table_generations = table_generations_to_thrift(closure->getTableGenerations());
#else
  CHECK(false);
#endif  // HAVE_RAVM
}

std::vector<TColumnRange> MapDHandler::column_ranges_to_thrift(const AggregatedColRange& column_ranges) {
  std::vector<TColumnRange> thrift_column_ranges;
  const auto& column_ranges_map = column_ranges.asMap();
  for (const auto& kv : column_ranges_map) {
    TColumnRange thrift_column_range;
    thrift_column_range.col_id = kv.first.col_id;
    thrift_column_range.table_id = kv.first.table_id;
    const auto& expr_range = kv.second;
    switch (expr_range.getType()) {
      case ExpressionRangeType::Integer:
        thrift_column_range.type = TExpressionRangeType::INTEGER;
        thrift_column_range.int_min = expr_range.getIntMin();
        thrift_column_range.int_max = expr_range.getIntMax();
        thrift_column_range.bucket = expr_range.getBucket();
        thrift_column_range.has_nulls = expr_range.hasNulls();
        break;
      case ExpressionRangeType::Float:
      case ExpressionRangeType::Double:
        thrift_column_range.type = expr_range.getType() == ExpressionRangeType::Float ? TExpressionRangeType::FLOAT
                                                                                      : TExpressionRangeType::DOUBLE;
        thrift_column_range.fp_min = expr_range.getFpMin();
        thrift_column_range.fp_max = expr_range.getFpMax();
        thrift_column_range.has_nulls = expr_range.hasNulls();
        break;
      case ExpressionRangeType::Invalid:
        thrift_column_range.type = TExpressionRangeType::INVALID;
        break;
      default:
        CHECK(false);
    }
    thrift_column_ranges.push_back(thrift_column_range);
  }
  return thrift_column_ranges;
}

std::vector<TDictionaryGeneration> MapDHandler::string_dictionary_generations_to_thrift(
    const StringDictionaryGenerations& dictionary_generations) {
  std::vector<TDictionaryGeneration> thrift_dictionary_generations;
  for (const auto& kv : dictionary_generations.asMap()) {
    TDictionaryGeneration thrift_dictionary_generation;
    thrift_dictionary_generation.dict_id = kv.first;
    thrift_dictionary_generation.entry_count = kv.second;
    thrift_dictionary_generations.push_back(thrift_dictionary_generation);
  }
  return thrift_dictionary_generations;
}

std::vector<TTableGeneration> MapDHandler::table_generations_to_thrift(const TableGenerations& table_generations) {
  std::vector<TTableGeneration> thrift_table_generations;
  for (const auto& kv : table_generations.asMap()) {
    TTableGeneration table_generation;
    table_generation.table_id = kv.first;
    table_generation.start_rowid = kv.second.start_rowid;
    table_generation.tuple_count = kv.second.tuple_count;
    thrift_table_generations.push_back(table_generation);
  }
  return thrift_table_generations;
}

void MapDHandler::broadcast_serialized_rows(const std::string& serialized_rows,
                                            const TRowDescriptor& row_desc,
                                            const TQueryId query_id) {
#ifdef HAVE_RAVM
  auto result_set = ResultSet::unserialize(serialized_rows, PendingExecutionClosure::getExecutor(query_id));
  ResultRows rows(std::shared_ptr<ResultSet>(result_set.release()));
  const auto target_meta = target_meta_infos_from_thrift(row_desc);
  const auto subquery_result = std::make_shared<const ExecutionResult>(rows, target_meta);
  PendingExecutionClosure::setCurrentSubqueryResult(query_id, subquery_result);
#else
  CHECK(false);
#endif  // HAVE_RAVM
}

void MapDHandler::insert_data(const TSessionId session, const TInsertData& thrift_insert_data) {
  static std::mutex insert_mutex;  // TODO: split lock, make it per table
  CHECK_EQ(thrift_insert_data.column_ids.size(), thrift_insert_data.data.size());
  const auto session_info = get_session(session);
  auto& cat = session_info.get_catalog();
  Fragmenter_Namespace::InsertData insert_data;
  insert_data.databaseId = thrift_insert_data.db_id;
  insert_data.tableId = thrift_insert_data.table_id;
  insert_data.columnIds = thrift_insert_data.column_ids;
  std::vector<std::unique_ptr<std::vector<std::string>>> none_encoded_string_columns;
  std::vector<std::unique_ptr<std::vector<ArrayDatum>>> array_columns;
  for (size_t col_idx = 0; col_idx < insert_data.columnIds.size(); ++col_idx) {
    const int column_id = insert_data.columnIds[col_idx];
    DataBlockPtr p;
    const auto cd = cat.getMetadataForColumn(insert_data.tableId, column_id);
    CHECK(cd);
    const auto& ti = cd->columnType;
    if (ti.is_number() || ti.is_time() || ti.is_boolean()) {
      p.numbersPtr = (int8_t*)thrift_insert_data.data[col_idx].fixed_len_data.data();
    } else if (ti.is_string()) {
      if (ti.get_compression() == kENCODING_DICT) {
        p.numbersPtr = (int8_t*)thrift_insert_data.data[col_idx].fixed_len_data.data();
      } else {
        CHECK_EQ(kENCODING_NONE, ti.get_compression());
        none_encoded_string_columns.emplace_back(new std::vector<std::string>());
        auto& none_encoded_strings = none_encoded_string_columns.back();
        CHECK_EQ(static_cast<size_t>(thrift_insert_data.num_rows),
                 thrift_insert_data.data[col_idx].var_len_data.size());
        for (const auto& varlen_str : thrift_insert_data.data[col_idx].var_len_data) {
          none_encoded_strings->push_back(varlen_str.payload);
        }
        p.stringsPtr = none_encoded_strings.get();
      }
    } else {
      CHECK(ti.is_array());
      array_columns.emplace_back(new std::vector<ArrayDatum>());
      auto& array_column = array_columns.back();
      CHECK_EQ(static_cast<size_t>(thrift_insert_data.num_rows), thrift_insert_data.data[col_idx].var_len_data.size());
      for (const auto& t_arr_datum : thrift_insert_data.data[col_idx].var_len_data) {
        if (t_arr_datum.is_null) {
          array_column->emplace_back(0, nullptr, true);
        } else {
          ArrayDatum arr_datum;
          arr_datum.length = t_arr_datum.payload.size();
          arr_datum.pointer = (int8_t*)t_arr_datum.payload.data();
          arr_datum.is_null = false;
          array_column->push_back(arr_datum);
        }
      }
      p.arraysPtr = array_column.get();
    }
    insert_data.data.push_back(p);
  }
  insert_data.numRows = thrift_insert_data.num_rows;
  const auto td = cat.getMetadataForTable(insert_data.tableId);
  try {
    td->fragmenter->insertData(insert_data);
  } catch (const std::exception& e) {
    TMapDException ex;
    ex.error_msg = std::string("Exception: ") + e.what();
    LOG(ERROR) << ex.error_msg;
    throw ex;
  }
}

#ifdef HAVE_RENDERING
namespace {

void convert_raw_pixel_data(TRawPixelDataResult& thrift_raw_pixel_data,
                            const QueryRenderer::RawPixelData& raw_pixel_data) {
  thrift_raw_pixel_data.width = raw_pixel_data.width;
  thrift_raw_pixel_data.height = raw_pixel_data.height;
  thrift_raw_pixel_data.num_channels = raw_pixel_data.numChannels;
  const size_t num_pixels = static_cast<size_t>(raw_pixel_data.width) * raw_pixel_data.height;
  thrift_raw_pixel_data.pixels =
      std::string((char*)raw_pixel_data.pixels.get(), num_pixels * raw_pixel_data.numChannels);
  thrift_raw_pixel_data.row_ids_A =
      std::string((char*)raw_pixel_data.rowIdsA.get(), num_pixels * sizeof(*raw_pixel_data.rowIdsA));
  thrift_raw_pixel_data.row_ids_B =
      std::string((char*)raw_pixel_data.rowIdsB.get(), num_pixels * sizeof(*raw_pixel_data.rowIdsB));
  thrift_raw_pixel_data.table_ids =
      std::string((char*)raw_pixel_data.tableIds.get(), num_pixels * sizeof(*raw_pixel_data.tableIds));
}

}  // namespace
#endif  // HAVE_RENDERING

/*
 * There's a lot of code duplication between this endpoint and render_vega,
 * we need to do something about it ASAP. A helper to create the QueryExecCB
 * lambda looks like an option, but there might be better ones.
 */
void MapDHandler::render_vega_raw_pixels(TRawPixelDataResult& _return,
                                         const TSessionId session,
                                         const int64_t widget_id,
                                         const std::string& vega_json) {
  CHECK_EQ(size_t(0), leaf_aggregator_.leafCount());
  _return.total_time_ms = measure<>::execution([&]() {
    _return.execution_time_ms = 0;
    _return.render_time_ms = 0;
    if (!enable_rendering_) {
      TMapDException ex;
      ex.error_msg = "Backend rendering is disabled.";
      LOG(ERROR) << ex.error_msg;
      throw ex;
    }

    std::lock_guard<std::mutex> render_lock(render_mutex_);
    mapd_shared_lock<mapd_shared_mutex> read_lock(sessions_mutex_);
    auto session_it = get_session_it(session);
    auto session_info_ptr = session_it->second.get();

    const auto& cat = session_info_ptr->get_catalog();
    auto executor = Executor::getExecutor(cat.get_currentDB().dbId,
                                          jit_debug_ ? "/tmp" : "",
                                          jit_debug_ ? "mapdquery" : "",
                                          mapd_parameters_,
#ifdef HAVE_RENDERING
                                          render_manager_.get());
#else
                                          nullptr);
#endif

#ifdef HAVE_RENDERING
    try {
      const auto session_id = session_info_ptr->get_session_id();
      std::unique_ptr<RenderInfo> render_info(new RenderInfo(session_id, widget_id));

      const auto pixel_data_and_timing = render_manager_->runPixelDataRenderRequest(
          session_id,
          widget_id,
          vega_json,
          executor.get(),
          render_info.get(),
          [&](QueryRenderer::RenderQueryExecuteTimer& render_timer,
              Executor* executor,
              const std::string& query,
              const rapidjson::Value* dataObj,
              bool is_poly_query = false) {
            std::string query_str = query;

            // reset any layouts from prior executions
            render_info->vbo_result_query_data_layout = nullptr;

            if (is_poly_query) {
              render_info->do_render = false;  // do not perform the render post query
                                               // For polys, we call an explicit executor->renderPolygons()
            } else {
              render_info->do_render = true;
              if (!render_info->render_allocator_map_ptr) {
                const auto& catalog = session_info_ptr->get_catalog();

                // NOTE: the following code calculating block_size_x &
                // grid_size_x was copied from Executor::blockSize() &
                // Executor::gridSize() respectively.
                // Had to pull that code out as the Executor might not
                // have been properly initialized with a catalog yet,
                // which is required
                CHECK(catalog.get_dataMgr().cudaMgr_);
                const auto& dev_props = catalog.get_dataMgr().cudaMgr_->deviceProperties;

                size_t t_cuda_block_size = mapd_parameters_.cuda_block_size;
                size_t t_cuda_grid_size = mapd_parameters_.cuda_grid_size;
                if (!t_cuda_block_size) {
                  t_cuda_block_size = dev_props.front().maxThreadsPerBlock;
                }

                if (!t_cuda_grid_size) {
                  t_cuda_grid_size = 2 * dev_props.front().numMPs;
                }

                render_info->render_allocator_map_ptr.reset(new RenderAllocatorMap(
                    catalog.get_dataMgr().cudaMgr_, render_manager_.get(), t_cuda_block_size, t_cuda_grid_size));
              }
            }

#ifdef HAVE_RAVM
            std::string query_ra;
            _return.execution_time_ms +=
                measure<>::execution([&]() { query_ra = parse_to_ra(query_str, *session_info_ptr); });
            auto rtnData = execute_render_rel_alg(render_timer,
                                                  query_ra,
                                                  query_str,
                                                  *session_info_ptr,
                                                  executor,
                                                  dataObj,
                                                  render_info.get(),
                                                  is_poly_query);
#else
            CHECK(false);
#endif  // HAVE_RAVM

            auto& usedTables = std::get<3>(rtnData);
            CHECK(usedTables.size() > 0);

            if (is_poly_query) {
            } else if (render_info->render_allocator_map_ptr && render_info->vbo_result_query_data_layout) {
              render_info->render_allocator_map_ptr->setDataLayout(render_info->vbo_result_query_data_layout);
            }

            return std::make_tuple(std::move(std::get<0>(rtnData)),
                                   std::move(std::get<1>(rtnData)),
                                   std::get<2>(rtnData),
                                   std::move(usedTables),
                                   render_info->vbo_result_query_data_layout,
                                   render_info->ubo_result_query_data_layout);
          },
          true);

      convert_raw_pixel_data(_return, std::get<0>(pixel_data_and_timing));
      _return.execution_time_ms = std::get<1>(pixel_data_and_timing);
      _return.render_time_ms = std::get<2>(pixel_data_and_timing);
    } catch (std::exception& e) {
      TMapDException ex;
      ex.error_msg = std::string("Exception: ") + e.what();
      LOG(ERROR) << ex.error_msg;
      throw ex;
    }
#endif  // HAVE_RENDERING
  });
  LOG(INFO) << "Total: " << _return.total_time_ms << " (ms), Total Execution: " << _return.execution_time_ms
            << " (ms), Total Render: " << _return.render_time_ms << " (ms)";
}

void MapDHandler::throw_profile_exception(const std::string& error_msg) {
  TMapDException ex;
  ex.error_msg = error_msg;
  LOG(ERROR) << ex.error_msg;
  throw ex;
}

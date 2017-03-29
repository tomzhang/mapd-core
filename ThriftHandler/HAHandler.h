/*
 * cool mapd license
 */

/*
 * File:   HAHandler.h
 * Author: michael
 *
 * Created on Jan 1, 2017, 12:40 PM
 */

#ifndef HAHANDLER_H
#define HAHANDLER_H

#include "gen-cpp/MapD.h"
#include "MapDHandler.h"
#include "Shared/MapDParameters.h"

class HAHandler : public MapDIf {
 public:
  HAHandler(MapDParameters& mapd_parameters, boost::shared_ptr<MapDHandler>& mapd_handler);

  ~HAHandler();

  TSessionId connect(const std::string& user, const std::string& passwd, const std::string& dbname);
  void disconnect(const TSessionId session);
  void interrupt(const TSessionId session);
  void get_server_status(TServerStatus& _return, const TSessionId session);
  void sql_execute(TQueryResult& _return,
                   const TSessionId session,
                   const std::string& query,
                   const bool column_format,
                   const std::string& nonce,
                   const int32_t first_n);
  void sql_validate(TTableDescriptor& _return, const TSessionId session, const std::string& query);
  void get_table_descriptor(TTableDescriptor& _return, const TSessionId session, const std::string& table_name);
  void get_row_descriptor(TRowDescriptor& _return, const TSessionId session, const std::string& table_name);
  void get_frontend_view(TFrontendView& _return, const TSessionId session, const std::string& view_name);
  void delete_frontend_view(const TSessionId session, const std::string& view_name);
  void get_tables(std::vector<std::string>& _return, const TSessionId session);
  void get_users(std::vector<std::string>& _return, const TSessionId session);
  void get_databases(std::vector<TDBInfo>& _return, const TSessionId session);
  void get_frontend_views(std::vector<TFrontendView>& _return, const TSessionId session);
  void set_execution_mode(const TSessionId session, const TExecuteMode::type mode);
  void get_version(std::string& _return);
  void get_memory_gpu(std::string& _return, const TSessionId session);
  void get_memory_summary(TMemorySummary& _return, const TSessionId session);
  void load_table_binary(const TSessionId session, const std::string& table_name, const std::vector<TRow>& rows);
  void load_table(const TSessionId session, const std::string& table_name, const std::vector<TStringRow>& rows);
  void render(TRenderResult& _return,
              const TSessionId session,
              const std::string& query,
              const std::string& render_type,
              const std::string& nonce);
  void render_vega(TRenderResult& _return,
                   const TSessionId session,
                   const int64_t widget_id,
                   const std::string& vega_json,
                   const int32_t compression_level,
                   const std::string& nonce);
  void create_frontend_view(const TSessionId session,
                            const std::string& view_name,
                            const std::string& view_state,
                            const std::string& image_hash,
                            const std::string& view_metadata);
  void detect_column_types(TDetectResult& _return,
                           const TSessionId session,
                           const std::string& file_name,
                           const TCopyParams& copy_params);
  void create_table(const TSessionId session,
                    const std::string& table_name,
                    const TRowDescriptor& row_desc,
                    const TTableType::type table_type);
  void import_table(const TSessionId session,
                    const std::string& table_name,
                    const std::string& file_name,
                    const TCopyParams& copy_params);
  void import_table_status(TImportStatus& _return, const TSessionId session, const std::string& import_id);
  void get_link_view(TFrontendView& _return, const TSessionId session, const std::string& link);
  void create_link(std::string& _return,
                   const TSessionId session,
                   const std::string& view_state,
                   const std::string& view_metadata);
  void get_rows_for_pixels(TPixelResult& _return,
                           const TSessionId session,
                           const int64_t widget_id,
                           const std::vector<TPixel>& pixels,
                           const std::string& table_name,
                           const std::vector<std::string>& col_names,
                           const bool column_format,
                           const std::string& nonce);
  void get_row_for_pixel(TPixelRowResult& _return,
                         const TSessionId session,
                         const int64_t widget_id,
                         const TPixel& pixel,
                         const std::string& table_name,
                         const std::vector<std::string>& col_names,
                         const bool column_format,
                         const int32_t pixelRadius,
                         const std::string& nonce);
  void get_result_row_for_pixel(TPixelTableRowResult& _return,
                                const TSessionId session,
                                const int64_t widget_id,
                                const TPixel& pixel,
                                const std::map<std::string, std::vector<std::string>>& table_col_names,
                                const bool column_format,
                                const int32_t pixelRadius,
                                const std::string& nonce);
  void start_heap_profile(const TSessionId session);
  void stop_heap_profile(const TSessionId session);
  void get_heap_profile(std::string& _return, const TSessionId session);
  void import_geo_table(const TSessionId session,
                        const std::string& table_name,
                        const std::string& file_name,
                        const TCopyParams& copy_params,
                        const TRowDescriptor& row_desc);
  void start_query(TPendingQuery& _return,
                   const TSessionId session,
                   const std::string& query_ra,
                   const bool just_explain);
  void execute_first_step(TStepResult& _return, const TPendingQuery& pending_query);
  void broadcast_serialized_rows(const std::string& serialized_rows,
                                 const TRowDescriptor& row_desc,
                                 const TQueryId query_id);
  void insert_data(const TSessionId session, const TInsertData& insert_data);
  void render_vega_raw_pixels(TRawPixelDataResult& _return,
                              const TSessionId session,
                              const int64_t widget_id,
                              const std::string& vega_json);
  void checkpoint(const TSessionId session, const int32_t db_id, const int32_t table_id);
  void get_table_details(TTableDetails& _return, const TSessionId session, const std::string& table_name);
  void clear_gpu_memory(const TSessionId session);
  void clear_cpu_memory(const TSessionId session);
};

#endif /* HAHANDLER_H */

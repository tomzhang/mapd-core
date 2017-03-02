/*
 * cool mapd license
 */

/*
 * File:   HAHandler.cpp
 * Author: michael
 *
 * Created on Jan 1, 2017, 12:40 PM
 */

#include "HAHandler.h"

HAHandler::HAHandler(MapDParameters& mapd_parameters, boost::shared_ptr<MapDHandler>& mapd_handler) {
  LOG(INFO) << " HAHandler starting";
};

HAHandler::~HAHandler() {
  LOG(INFO) << " HAHandler stopping";
};

TSessionId HAHandler::connect(const std::string& user, const std::string& passwd, const std::string& dbname) {
  return 1;
};

void HAHandler::disconnect(const TSessionId session){};

void HAHandler::get_server_status(TServerStatus& _return, const TSessionId session){};

void HAHandler::sql_execute(TQueryResult& _return,
                            const TSessionId session,
                            const std::string& query,
                            const bool column_format,
                            const std::string& nonce){};

void HAHandler::sql_validate(TTableDescriptor& _return, const TSessionId session, const std::string& query){};

void HAHandler::get_table_descriptor(TTableDescriptor& _return,
                                     const TSessionId session,
                                     const std::string& table_name){};

void HAHandler::get_row_descriptor(TRowDescriptor& _return, const TSessionId session, const std::string& table_name){};

void HAHandler::get_frontend_view(TFrontendView& _return, const TSessionId session, const std::string& view_name){};

void HAHandler::delete_frontend_view(const TSessionId session, const std::string& view_name){};

void HAHandler::get_tables(std::vector<std::string>& _return, const TSessionId session){};

void HAHandler::get_users(std::vector<std::string>& _return, const TSessionId session){};

void HAHandler::get_databases(std::vector<TDBInfo>& _return, const TSessionId session){};

void HAHandler::get_frontend_views(std::vector<TFrontendView>& _return, const TSessionId session){};

void HAHandler::set_execution_mode(const TSessionId session, const TExecuteMode::type mode){};

void HAHandler::get_version(std::string& _return){};

void HAHandler::get_memory_gpu(std::string& _return, const TSessionId session){};

void HAHandler::get_memory_summary(TMemorySummary& _return, const TSessionId session){};

void HAHandler::load_table_binary(const TSessionId session,
                                  const std::string& table_name,
                                  const std::vector<TRow>& rows){};

void HAHandler::load_table(const TSessionId session,
                           const std::string& table_name,
                           const std::vector<TStringRow>& rows){};

void HAHandler::render(TRenderResult& _return,
                       const TSessionId session,
                       const std::string& query,
                       const std::string& render_type,
                       const std::string& nonce){};

void HAHandler::render_vega(TRenderResult& _return,
                            const TSessionId session,
                            const int64_t widget_id,
                            const std::string& vega_json,
                            const int32_t compression_level,
                            const std::string& nonce){};

void HAHandler::create_frontend_view(const TSessionId session,
                                     const std::string& view_name,
                                     const std::string& view_state,
                                     const std::string& image_hash,
                                     const std::string& view_metadata){};

void HAHandler::detect_column_types(TDetectResult& _return,
                                    const TSessionId session,
                                    const std::string& file_name,
                                    const TCopyParams& copy_params){};

void HAHandler::create_table(const TSessionId session,
                             const std::string& table_name,
                             const TRowDescriptor& row_desc,
                             const TTableType::type table_type){};

void HAHandler::import_table(const TSessionId session,
                             const std::string& table_name,
                             const std::string& file_name,
                             const TCopyParams& copy_params){};

void HAHandler::import_table_status(TImportStatus& _return, const TSessionId session, const std::string& import_id){};

void HAHandler::get_link_view(TFrontendView& _return, const TSessionId session, const std::string& link){};

void HAHandler::create_link(std::string& _return,
                            const TSessionId session,
                            const std::string& view_state,
                            const std::string& view_metadata){};

void HAHandler::get_rows_for_pixels(TPixelResult& _return,
                                    const TSessionId session,
                                    const int64_t widget_id,
                                    const std::vector<TPixel>& pixels,
                                    const std::string& table_name,
                                    const std::vector<std::string>& col_names,
                                    const bool column_format,
                                    const std::string& nonce){};

void HAHandler::get_row_for_pixel(TPixelRowResult& _return,
                                  const TSessionId session,
                                  const int64_t widget_id,
                                  const TPixel& pixel,
                                  const std::string& table_name,
                                  const std::vector<std::string>& col_names,
                                  const bool column_format,
                                  const int32_t pixelRadius,
                                  const std::string& nonce){};

void HAHandler::get_result_row_for_pixel(TPixelTableRowResult& _return,
                                         const TSessionId session,
                                         const int64_t widget_id,
                                         const TPixel& pixel,
                                         const std::map<std::string, std::vector<std::string>>& table_col_names,
                                         const bool column_format,
                                         const int32_t pixelRadius,
                                         const std::string& nonce){};

void HAHandler::start_heap_profile(const TSessionId session){};

void HAHandler::stop_heap_profile(const TSessionId session){};

void HAHandler::get_heap_profile(std::string& _return, const TSessionId session){};

void HAHandler::import_geo_table(const TSessionId session,
                                 const std::string& table_name,
                                 const std::string& file_name,
                                 const TCopyParams& copy_params,
                                 const TRowDescriptor& row_desc){};

void HAHandler::start_query(TPendingQuery& _return, const TSessionId session, const std::string& query_ra){};

void HAHandler::execute_first_step(TStepResult& _return, const TPendingQuery& pending_query){};

void HAHandler::broadcast_serialized_rows(const std::string& serialized_rows,
                                          const TRowDescriptor& row_desc,
                                          const TQueryId query_id){};

void HAHandler::get_table_details(TTableDetails& _return, const TSessionId session, const std::string& table_name){};

void HAHandler::clear_gpu_memory(const TSessionId session){};

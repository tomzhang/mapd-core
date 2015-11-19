#include "MapDGL.h"
#include "QueryRenderer.h"
#include "RapidJSONUtils.h"
#include "../QueryEngine/Execute.h"
#include <glog/logging.h>
#include <utility>  // std::pair
#include <stdexcept>
#include "rapidjson/error/en.h"

using namespace MapD_Renderer;

QueryRenderer::QueryRenderer(const Executor* executor,
                             const QueryResultVertexBufferShPtr& queryResultVBOPtr,
                             bool doHitTest,
                             bool doDepthTest,
                             GLFWwindow* win)
    : _ctx(new QueryRendererContext(executor, queryResultVBOPtr, doHitTest, doDepthTest)), _framebufferPtr(nullptr) {
}

QueryRenderer::QueryRenderer(const Executor* executor,
                             const rapidjson::Document& jsonDocument,
                             const QueryResultVertexBufferShPtr& queryResultVBOPtr,
                             bool doHitTest,
                             bool doDepthTest,
                             GLFWwindow* win)
    : _ctx(new QueryRendererContext(executor, queryResultVBOPtr, doHitTest, doDepthTest)), _framebufferPtr(nullptr) {
  _initFromJSON(jsonDocument, win);
}

QueryRenderer::QueryRenderer(const Executor* executor,
                             const std::string& configJSON,
                             const QueryResultVertexBufferShPtr& queryResultVBOPtr,
                             bool doHitTest,
                             bool doDepthTest,
                             GLFWwindow* win)
    : _ctx(new QueryRendererContext(executor, queryResultVBOPtr, doHitTest, doDepthTest)), _framebufferPtr(nullptr) {
  _initFromJSON(configJSON, win);
}

QueryRenderer::~QueryRenderer() {
  _clear();
}

void QueryRenderer::_clear() {
  _ctx->_clear();
}

void QueryRenderer::_initFramebuffer(int width, int height) {
  if (_framebufferPtr == nullptr) {
    _framebufferPtr.reset(new QueryFramebuffer(width, height, _ctx->doHitTest(), _ctx->doDepthTest()));
  } else {
    _framebufferPtr->resize(width, height);
  }
}

void QueryRenderer::_initFromJSON(const std::string& configJSON, GLFWwindow* win) {
  rapidjson::Document obj;
  obj.Parse(configJSON.c_str());

  // TODO(croot): this can be removed if the executor will handle the initial parse.
  RUNTIME_EX_ASSERT(!obj.HasParseError(),
                    "JSON parse error - " + std::to_string(obj.GetErrorOffset()) + ", error: " +
                        rapidjson::GetParseError_En(obj.GetParseError()));

  _initFromJSON(obj, win);
}

void QueryRenderer::_initFromJSON(const rapidjson::Value& obj, GLFWwindow* win) {
  // clear out the previous state? Or do we want to maintain the previous state in case of an error?
  _clear();

  RUNTIME_EX_ASSERT(obj.IsObject(), "JSON parse error - Root object is not a JSON object.");

  rapidjson::Value::ConstMemberIterator mitr;
  rapidjson::Value::ConstValueIterator vitr;

  RUNTIME_EX_ASSERT((mitr = obj.FindMember("width")) != obj.MemberEnd(),
                    "JSON parse error - \"width\" is not defined.");
  RUNTIME_EX_ASSERT(mitr->value.IsInt(), "JSON parse error - \"width\" is not an integer.");
  int width = mitr->value.GetInt();

  RUNTIME_EX_ASSERT((mitr = obj.FindMember("height")) != obj.MemberEnd(),
                    "JSON parse error - \"height\" is not defined.");
  RUNTIME_EX_ASSERT(mitr->value.IsInt(), "JSON parse error - \"height\" is not an integer.");
  int height = mitr->value.GetInt();

  setWidthHeight(width, height, win);

  mitr = obj.FindMember("data");
  if (mitr != obj.MemberEnd()) {
    RUNTIME_EX_ASSERT(mitr->value.IsArray(), "JSON parse error - the \"data\" member must be an array.");

    DataVBOShPtr dataTablePtr;

    for (vitr = mitr->value.Begin(); vitr != mitr->value.End(); ++vitr) {
      // dataTablePtr.reset(
      //     new DataTable(*vitr, _doHitTest, DataTable::VboType::INTERLEAVED));  // NOTE: uses a SEQUENTIAL vbo by
      // default
      // new DataTable(*vitr, _doHitTest, DataTable::VboType::SEQUENTIAL));  // NOTE: uses a SEQUENTIAL vbo by default

      dataTablePtr = createDataTable(*vitr, _ctx);

      _ctx->_dataTableMap.insert(std::make_pair(dataTablePtr->getName(), dataTablePtr));
    }
  }

  mitr = obj.FindMember("scales");
  if (mitr != obj.MemberEnd()) {
    RUNTIME_EX_ASSERT(mitr->value.IsArray(), "JSON parse error - the \"scales\" member must be an array.");

    for (vitr = mitr->value.Begin(); vitr != mitr->value.End(); ++vitr) {
      ScaleShPtr scaleConfig = createScale(*vitr, _ctx);
      _ctx->_scaleConfigMap.insert(std::make_pair(scaleConfig->name, scaleConfig));
    }
  }

  mitr = obj.FindMember("marks");
  if (mitr != obj.MemberEnd()) {
    RUNTIME_EX_ASSERT(mitr->value.IsArray(), "JSON parse error - the \"marks\" member must be an array");

    for (vitr = mitr->value.Begin(); vitr != mitr->value.End(); ++vitr) {
      GeomConfigShPtr geomConfigPtr = createMark(*vitr, _ctx);

      _ctx->_geomConfigs.push_back(geomConfigPtr);
    }
  }
}

int QueryRenderer::getWidth() {
  return _ctx->_width;
}

int QueryRenderer::getHeight() {
  return _ctx->_height;
}

void QueryRenderer::setWidthHeight(int width, int height, GLFWwindow* win) {
  _ctx->_width = width;
  _ctx->_height = height;

  if (win) {
    // pass a window in debug mode

    // resize the window
    glfwSetWindowSize(win, width, height);

    // now get the actual framebuffer dimensions
    glfwGetFramebufferSize(win, &_ctx->_width, &_ctx->_height);
  }

  _initFramebuffer(_ctx->_width, _ctx->_height);
}

const QueryFramebufferUqPtr& QueryRenderer::getFramebuffer() {
  return _framebufferPtr;
}

void QueryRenderer::setJSONConfig(const std::string& configJSON, GLFWwindow* win) {
  _initFromJSON(configJSON, win);
}

void QueryRenderer::setJSONDocument(const rapidjson::Document& jsonDocument, GLFWwindow* win) {
  _initFromJSON(jsonDocument, win);
}

void QueryRenderer::updateQueryResultBufferPostQuery(const BufferLayoutShPtr& layoutPtr,
                                                     const int numRows,
                                                     const int64_t invalid_key) {
  _ctx->_queryResultBufferLayout = layoutPtr;
  _ctx->_queryResultVBOPtr->updatePostQuery(layoutPtr, numRows);
  _ctx->_invalidKey = invalid_key;

  // TODO(croot): Should each mark have its own invalid key? Probably, if we support
  // multiple sqls.
  // But if we deal with multiple sqls, then the context probably shouldn't
  // hold onto the invalidKey - the QueryResultVertexBuffer probably should, but
  // then how do we deal with setting that uniform?
  for (size_t i = 0; i < _ctx->_geomConfigs.size(); ++i) {
    _ctx->_geomConfigs[i]->setInvalidKey(_ctx->_invalidKey);
  }
}

void QueryRenderer::render() {
  RUNTIME_EX_ASSERT(_framebufferPtr != nullptr, "QueryRenderer: The framebuffer is not defined. Cannot render.");

  _framebufferPtr->bindToRenderer();

  MAPD_CHECK_GL_ERROR(glEnable(GL_BLEND));
  MAPD_CHECK_GL_ERROR(glBlendEquationSeparate(GL_FUNC_ADD, GL_FUNC_ADD));
  MAPD_CHECK_GL_ERROR(glBlendFuncSeparate(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA, GL_ONE, GL_ZERO));

  MAPD_CHECK_GL_ERROR(glClearColor(0, 0, 0, 0));
  MAPD_CHECK_GL_ERROR(glViewport(0, 0, _ctx->_width, _ctx->_height));
  MAPD_CHECK_GL_ERROR(glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT));

  for (size_t i = 0; i < _ctx->_geomConfigs.size(); ++i) {
    _ctx->_geomConfigs[i]->draw();
  }
}

unsigned int QueryRenderer::getIdAt(int x, int y) {
  RUNTIME_EX_ASSERT(_framebufferPtr != nullptr,
                    "QueryRenderer: The framebuffer is not defined. Cannot retrieve id at pixel.");

  // TODO(croot): develop an API for reading from specific fbo buffers
  _framebufferPtr->bindToRenderer();
  MAPD_CHECK_GL_ERROR(glReadBuffer(GL_COLOR_ATTACHMENT1));

  // TODO(croot): support a wider pixel check for a hit test and take a weighted avg
  // of the results to get a more stable result at boundaries
  unsigned int id;
  MAPD_CHECK_GL_ERROR(glReadPixels(int(x), int(y), 1, 1, GL_RED_INTEGER, GL_UNSIGNED_INT, &id));

  return id;
}

bool QueryRendererContext::hasDataTable(const std::string& tableName) const {
  return (_dataTableMap.find(tableName) != _dataTableMap.end());
}

DataVBOShPtr QueryRendererContext::getDataTable(const std::string& tableName) const {
  DataVBOShPtr rtn(nullptr);

  auto itr = _dataTableMap.find(tableName);
  if (itr != _dataTableMap.end()) {
    rtn = itr->second;
  }

  return rtn;
}

bool QueryRendererContext::hasScale(const std::string& scaleConfigName) const {
  return (_scaleConfigMap.find(scaleConfigName) != _scaleConfigMap.end());
}

ScaleShPtr QueryRendererContext::getScale(const std::string& scaleConfigName) const {
  ScaleShPtr rtn(nullptr);

  auto itr = _scaleConfigMap.find(scaleConfigName);
  if (itr != _scaleConfigMap.end()) {
    rtn = itr->second;
  }

  return rtn;
}

// void QueryRenderer::_buildShaderFromGeomConfig(const GeomConfigPtr& geomConfigPtr) {

// }

DataVBOShPtr MapD_Renderer::createDataTable(const rapidjson::Value& obj, const QueryRendererContextShPtr& ctx) {
  RUNTIME_EX_ASSERT(obj.IsObject(), "createDataTable: A data object in the JSON must be an object.");

  rapidjson::Value::ConstMemberIterator itr;

  RUNTIME_EX_ASSERT((itr = obj.FindMember("name")) != obj.MemberEnd() && itr->value.IsString(),
                    "createDataTable: A data object must contain a \"name\" property and it must be a string");
  std::string tableName = itr->value.GetString();

  if ((itr = obj.FindMember("sql")) != obj.MemberEnd()) {
    RUNTIME_EX_ASSERT(itr->value.IsString(),
                      "createDataTable: The sql property for \"" + tableName + "\" must be a string.");
    return DataVBOShPtr(
        new SqlQueryDataTable(tableName, obj, ctx->getQueryResultVertexBuffer(), itr->value.GetString()));
  } else if ((itr = obj.FindMember("values")) != obj.MemberEnd()) {
    return DataVBOShPtr(new DataTable(tableName, obj, ctx->doHitTest(), DataTable::VboType::INTERLEAVED));
  } else if ((itr = obj.FindMember("url")) != obj.MemberEnd()) {
    return DataVBOShPtr(new DataTable(tableName, obj, ctx->doHitTest(), DataTable::VboType::INTERLEAVED));
  }
  THROW_RUNTIME_EX("createDataTable: There is not valid data property in the table \"" + tableName +
                   "\". Must have an \"sql\", \"values\" or \"url\" property.");

  return DataVBOShPtr(nullptr);
}

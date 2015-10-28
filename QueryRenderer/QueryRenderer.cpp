#include "QueryRenderer.h"
#include "RapidJSONUtils.h"
#include <glog/logging.h>
#include <utility>  // std::pair
#include <stdexcept>
#include "rapidjson/error/en.h"

using namespace MapD_Renderer;

QueryRenderer::QueryRenderer(const QueryResultVertexBufferShPtr& queryResultVBOPtr,
                             bool doHitTest,
                             bool doDepthTest,
                             GLFWwindow* win)
    : _doHitTest(doHitTest),
      _doDepthTest(doDepthTest),
      _ctx(new QueryRendererContext(queryResultVBOPtr)),
      _framebufferPtr(nullptr) {
}

QueryRenderer::QueryRenderer(const rapidjson::Document& jsonDocument,
                             const QueryResultVertexBufferShPtr& queryResultVBOPtr,
                             bool doHitTest,
                             bool doDepthTest,
                             GLFWwindow* win)
    : _doHitTest(doHitTest),
      _doDepthTest(doDepthTest),
      _ctx(new QueryRendererContext(queryResultVBOPtr)),
      _framebufferPtr(nullptr) {
  _initFromJSON(jsonDocument, win);
}

QueryRenderer::QueryRenderer(const std::string& configJSON,
                             const QueryResultVertexBufferShPtr& queryResultVBOPtr,
                             bool doHitTest,
                             bool doDepthTest,
                             GLFWwindow* win)
    : _doHitTest(doHitTest),
      _doDepthTest(doDepthTest),
      _ctx(new QueryRendererContext(queryResultVBOPtr)),
      _framebufferPtr(nullptr) {
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
    _framebufferPtr.reset(new QueryFramebuffer(width, height, _doHitTest, _doDepthTest));
  } else {
    _framebufferPtr->resize(width, height);
  }
}

void QueryRenderer::_initFromJSON(const std::string& configJSON, GLFWwindow* win) {
  rapidjson::Document obj;
  obj.Parse(configJSON.c_str());

  // TODO(croot): this can be removed if the executor will handle the initial parse.
  if (obj.HasParseError()) {
    throw RJMapDException("Error parsing the json config: offset: " + std::to_string(obj.GetErrorOffset()) +
                          ", error: " + rapidjson::GetParseError_En(obj.GetParseError()));
  }

  _initFromJSON(obj, win);
}

void QueryRenderer::_initFromJSON(const rapidjson::Value& obj, GLFWwindow* win) {
  // clear out the previous state? Or do we want to maintain the previous state in case of an error?
  _clear();

  if (!obj.IsObject()) {
    throw RJMapDException("Root object is not a JSON object.");
  }

  rapidjson::Value::ConstMemberIterator mitr;
  rapidjson::Value::ConstValueIterator vitr;

  if ((mitr = obj.FindMember("width")) == obj.MemberEnd()) {
    throw RJMapDException("\"width\" is not defined.");
  } else if (!mitr->value.IsInt()) {
    throw RJMapDException("\"width\" is not an integer.");
  }
  int width = mitr->value.GetInt();

  if ((mitr = obj.FindMember("height")) == obj.MemberEnd()) {
    throw RJMapDException("\"height\" is not defined.");
  } else if (!mitr->value.IsInt()) {
    throw RJMapDException("\"width\" is not an integer.");
  }
  int height = mitr->value.GetInt();

  setWidthHeight(width, height, win);

  mitr = obj.FindMember("data");
  if (mitr != obj.MemberEnd()) {
    if (!mitr->value.IsArray()) {
      throw RJMapDException("the \"data\" member must be an array.");
    }

    DataTableShPtr dataTablePtr;

    for (vitr = mitr->value.Begin(); vitr != mitr->value.End(); ++vitr) {
      dataTablePtr.reset(
          new DataTable(*vitr, _doHitTest, DataTable::VboType::INTERLEAVED));  // NOTE: uses a SEQUENTIAL vbo by
                                                                               // default
      // new DataTable(*vitr, _doHitTest, DataTable::VboType::SEQUENTIAL));  // NOTE: uses a SEQUENTIAL vbo by default
      _ctx->_dataTableMap.insert(std::make_pair(dataTablePtr->getName(), dataTablePtr));
    }
  }

  mitr = obj.FindMember("scales");
  if (mitr != obj.MemberEnd()) {
    if (!mitr->value.IsArray()) {
      throw RJMapDException("the \"scales\" member must be an array.");
    }

    for (vitr = mitr->value.Begin(); vitr != mitr->value.End(); ++vitr) {
      ScaleShPtr scaleConfig = createScale(*vitr, _ctx);
      _ctx->_scaleConfigMap.insert(std::make_pair(scaleConfig->name, scaleConfig));
    }
  }

  mitr = obj.FindMember("marks");
  if (mitr != obj.MemberEnd()) {
    if (!mitr->value.IsArray()) {
      throw RJMapDException("the \"marks\" member must be an array");
    }

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

void QueryRenderer::render() {
  if (!_framebufferPtr) {
    throw std::runtime_error("QueryRenderer: The framebuffer is not defined. Cannot render.");
  }

  _framebufferPtr->bindToRenderer();

  glClearColor(0, 0, 0, 0);
  glViewport(0, 0, _ctx->_width, _ctx->_height);
  glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

  for (size_t i = 0; i < _ctx->_geomConfigs.size(); ++i) {
    _ctx->_geomConfigs[i]->draw();
  }
}

unsigned int QueryRenderer::getIdAt(int x, int y) {
  if (!_framebufferPtr) {
    throw std::runtime_error("QueryRenderer: The framebuffer is not defined. Cannot render.");
  }

  // TODO(croot): develop an API for reading from specific fbo buffers
  _framebufferPtr->bindToRenderer();
  glReadBuffer(GL_COLOR_ATTACHMENT1);

  // TODO(croot): support a wider pixel check for a hit test and take a weighted avg
  // of the results to get a more stable result at boundaries
  unsigned int id;
  glReadPixels(int(x), int(y), 1, 1, GL_RED_INTEGER, GL_UNSIGNED_INT, &id);

  return id;
}

bool QueryRendererContext::hasDataTable(const std::string& tableName) const {
  return (_dataTableMap.find(tableName) != _dataTableMap.end());
}

DataTableShPtr QueryRendererContext::getDataTable(const std::string& tableName) const {
  DataTableShPtr rtn(nullptr);

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

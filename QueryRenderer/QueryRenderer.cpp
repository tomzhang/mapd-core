#include "QueryRenderer.h"
#include <utility> // std::pair


using namespace MapD_Renderer;

QueryRenderer::QueryRenderer(const std::string& configJSON, bool doHitTest, bool doDepthTest, GLFWwindow *win) : _doHitTest(doHitTest), _doDepthTest(doDepthTest), _ctx(new QueryRendererContext()), _framebufferPtr(nullptr) {
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

void QueryRenderer::_initFromJSON(const std::string& configJSON, GLFWwindow *win) {

    // clear out the previous state? Or do we want to maintain the previous state in case of an error?
    _clear();

    rapidjson::Document obj;
    obj.Parse(configJSON.c_str());

    if (obj.HasParseError()) {
        // TODO: throw an error and gracefully deal
        // with an un-initialized renderer object
        //
        // Use the following to get the error type
        // for useful error info & logging:
        // rapidjson::ParseErrorCode obj.GetParseError();
        // see http://rapidjson.org/md_doc_dom.html#ParseError for more
        assert(false);
    }

    // TODO: throw exceptions instead of asserts
    assert(obj.IsObject());

    rapidjson::Value::ConstMemberIterator mitr;
    rapidjson::Value::ConstValueIterator vitr;

    assert((mitr = obj.FindMember("width")) != obj.MemberEnd() &&
           mitr->value.IsInt());
    int width = mitr->value.GetInt();

    assert((mitr = obj.FindMember("height")) != obj.MemberEnd() &&
           mitr->value.IsInt());
    int height = mitr->value.GetInt();


    setWidthHeight(width, height, win);


    mitr = obj.FindMember("data");
    if (mitr != obj.MemberEnd()) {
        assert(mitr->value.IsArray());

        DataTableShPtr dataTablePtr;

        for (vitr = mitr->value.Begin(); vitr != mitr->value.End(); ++vitr) {
            dataTablePtr.reset(new DataTable(*vitr, _doHitTest)); // NOTE: uses a SEQUENTIAL vbo by default
            _ctx->_dataTableMap.insert(std::make_pair(dataTablePtr->getName(), dataTablePtr));
        }
    }

    mitr = obj.FindMember("scales");
    if (mitr != obj.MemberEnd()) {
        assert(mitr->value.IsArray());

        for (vitr = mitr->value.Begin(); vitr != mitr->value.End(); ++vitr) {
            ScaleShPtr scaleConfig = createScale(*vitr, _ctx);
            _ctx->_scaleConfigMap.insert(std::make_pair(scaleConfig->name, scaleConfig));
        }
    }

    mitr = obj.FindMember("marks");
    if (mitr != obj.MemberEnd()) {
        assert(mitr->value.IsArray());

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

void QueryRenderer::setWidthHeight(int width, int height, GLFWwindow *win) {
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

void QueryRenderer::setJSONConfig(const std::string& configJSON, GLFWwindow *win) {
    _initFromJSON(configJSON);
}

void QueryRenderer::render() {
    assert(_framebufferPtr != nullptr);

    _framebufferPtr->bindToRenderer();

    glClearColor(0,0,0,0);
    glViewport(0, 0, _ctx->_width, _ctx->_height);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    for (int i=0; i<_ctx->_geomConfigs.size(); ++i) {
        _ctx->_geomConfigs[i]->draw();
    }
}

unsigned int QueryRenderer::getIdAt(int x, int y) {
    // TODO: develop an API for reading from specific fbo buffers
    _framebufferPtr->bindToRenderer();
    glReadBuffer(GL_COLOR_ATTACHMENT1);

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

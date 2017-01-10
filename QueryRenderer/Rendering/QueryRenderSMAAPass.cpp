#include "QueryRenderSMAAPass.h"

#include "shaders/SMAAPassThru_vert.h"
#include "shaders/SMAAEdgeDetection_frag.h"
#include "shaders/SMAABlendingWeightCalculation_frag.h"
#include "shaders/SMAANeighborhoodBlending_frag.h"

#include "textures/AreaTex.h"
#include "textures/SearchTex.h"

#include "../PngData.h"

#include <Rendering/Renderer/GL/GLRenderer.h>
#include <Rendering/Renderer/GL/GLResourceManager.h>
#include <Rendering/Renderer/GL/Resources/GLShader.h>

#include <regex>
#include <iostream>

namespace QueryRenderer {

QueryRenderSMAAPass::QueryRenderSMAAPass(::Rendering::GL::GLRendererShPtr& rendererPtr,
                                         const size_t width,
                                         const size_t height,
                                         const size_t numSamples,
                                         SMAA_QUALITY_PRESET qualityPreset,
                                         SMAA_EDGE_DETECTION_TYPE edgeDetectType)
    : _rendererPtr(rendererPtr),
      _qualityPreset(qualityPreset),
      _edgeDetectType(edgeDetectType),
      _usePredication(false),
      _useReprojection(false) {
  RUNTIME_EX_ASSERT(numSamples == 1,
                    "SMAA Anti-aliasing is currently only supported for 1 sample per-pixel, not " +
                        std::to_string(numSamples) + " samples.");
  _initializeRsrcs(width, height, numSamples);
}

QueryRenderSMAAPass::~QueryRenderSMAAPass() {}

void QueryRenderSMAAPass::_initializeRsrcs(const size_t width, const size_t height, const size_t numSamples) {
  CHECK(_rendererPtr);

  // TODO(croot): restore active renderer state?
  _rendererPtr->makeActiveOnCurrentThread();

  auto rsrcMgr = _rendererPtr->getResourceManager();

  auto samplingParams =
      ::Rendering::GL::Resources::GLTexture2dSampleProps(GL_LINEAR, GL_LINEAR, GL_CLAMP_TO_EDGE, GL_CLAMP_TO_EDGE);

  _edgeDetectionTex =
      rsrcMgr->createTexture2d(width, height, GL_RGBA8, GL_RGBA, GL_UNSIGNED_BYTE, numSamples, samplingParams);
  _edgeDetectionFbo = rsrcMgr->createFramebuffer({{GL_COLOR_ATTACHMENT0, _edgeDetectionTex}});

  _blendingWeightTex =
      rsrcMgr->createTexture2d(width, height, GL_RGBA8, GL_RGBA, GL_UNSIGNED_BYTE, numSamples, samplingParams);
  _blendingWeightFbo = rsrcMgr->createFramebuffer({{GL_COLOR_ATTACHMENT0, _blendingWeightTex}});

  _buildShaders(rsrcMgr);

  float fwidth = static_cast<float>(width);
  float fheight = static_cast<float>(height);

  ::Rendering::GL::Resources::GLInterleavedBufferLayoutShPtr bufferLayout(
      new ::Rendering::GL::Resources::GLInterleavedBufferLayout(_rendererPtr->getSupportedExtensions()));
  bufferLayout->addAttribute<float, 2>("pos");
  bufferLayout->addAttribute<float, 2>("texcoord");
  _rectvbo = rsrcMgr->createVertexBuffer<float>(
      {0.0, 0.0, 0.0, 0.0, fwidth, 0.0, 1.0, 0.0, 0.0, fheight, 0.0, 1.0, fwidth, fheight, 1.0, 1.0}, bufferLayout);

  _rendererPtr->bindShader(_edgeDetectionShader);
  _vao = rsrcMgr->createVertexArray({{_rectvbo, {}}});

  _areaTex = rsrcMgr->createTexture2d(
      AREATEX_WIDTH, AREATEX_HEIGHT, GL_RG8, GL_RG, GL_UNSIGNED_BYTE, 1, samplingParams, areaTexBytes);

  _searchTex = rsrcMgr->createTexture2d(
      SEARCHTEX_WIDTH, SEARCHTEX_HEIGHT, GL_R8, GL_RED, GL_UNSIGNED_BYTE, 1, samplingParams, searchTexBytes);
}

size_t QueryRenderSMAAPass::getWidth() const {
  return _edgeDetectionFbo->getWidth();
}

size_t QueryRenderSMAAPass::getHeight() const {
  return _edgeDetectionFbo->getHeight();
}

void QueryRenderSMAAPass::_buildShaders(::Rendering::GL::GLResourceManagerShPtr& rsrcMgr) {
  std::regex versionRegex("^#version\\s+(\\d+)\\s+core\\s*");
  std::smatch versionMatch;

  std::string predefineStr = "\n#define SMAA_PRESET_";
  switch (_qualityPreset) {
    case SMAA_QUALITY_PRESET::LOW:
      predefineStr += "LOW\n";
      break;
    case SMAA_QUALITY_PRESET::MEDIUM:
      predefineStr += "MEDIUM\n";
      break;
    case SMAA_QUALITY_PRESET::HIGH:
      predefineStr += "HIGH\n";
      break;
    case SMAA_QUALITY_PRESET::ULTRA:
      predefineStr += "ULTRA\n";
      break;
  }

  std::string vertSrc = SMAAPassThru_vert::source;

  {
    // build the edge detection shader first
    std::string edgeDetectPredefineStr = predefineStr;

    if (_usePredication) {
      edgeDetectPredefineStr += "#define SMAA_PREDICATION\n";
    }

    edgeDetectPredefineStr += "\n";

    std::string fragSrc = SMAAEdgeDetection_frag::source;

    RUNTIME_EX_ASSERT(std::regex_search(fragSrc, versionMatch, versionRegex),
                      "QueryRenderSMAAPass: Cannot find a GLSL version string in the SMAAEdgeDetection shader");

    fragSrc.insert(versionMatch.position() + versionMatch.length(), edgeDetectPredefineStr);

    _edgeDetectionShader = rsrcMgr->createShader(vertSrc, fragSrc);

    _rendererPtr->bindShader(_edgeDetectionShader);
    _edgeDetectionShader->setSamplerTextureImageUnit("colorTex", GL_TEXTURE0);
  }

  {
    // build the blending weight calculation shader
    std::string blendPredefineStr = predefineStr;

    blendPredefineStr += "\n";

    std::string fragSrc = SMAABlendingWeightCalculation_frag::source;

    RUNTIME_EX_ASSERT(
        std::regex_search(fragSrc, versionMatch, versionRegex),
        "QueryRenderSMAAPass: Cannot find a GLSL version string in the SMAABlendingWeightCalculation shader");

    fragSrc.insert(versionMatch.position() + versionMatch.length(), blendPredefineStr);

    _blendingWeightShader = rsrcMgr->createShader(vertSrc, fragSrc);
    _rendererPtr->bindShader(_blendingWeightShader);
    _blendingWeightShader->setSamplerTextureImageUnit("edgeTex", GL_TEXTURE0);
    _blendingWeightShader->setSamplerTextureImageUnit("areaTex", GL_TEXTURE1);
    _blendingWeightShader->setSamplerTextureImageUnit("searchTex", GL_TEXTURE2);
  }

  {
    // build the neighborhood blending shader
    std::string neighborPredefineStr = predefineStr;

    if (_useReprojection) {
      neighborPredefineStr += "#define SMAA_REPROJECTION\n";
    }

    neighborPredefineStr += "\n";

    std::string fragSrc = SMAANeighborhoodBlending_frag::source;

    RUNTIME_EX_ASSERT(std::regex_search(fragSrc, versionMatch, versionRegex),
                      "QueryRenderSMAAPass: Cannot find a GLSL version string in the SMAANeighborhoodBlending shader");

    fragSrc.insert(versionMatch.position() + versionMatch.length(), neighborPredefineStr);

    _neighborhoodBlendShader = rsrcMgr->createShader(vertSrc, fragSrc);
    _rendererPtr->bindShader(_neighborhoodBlendShader);
    _neighborhoodBlendShader->setSamplerTextureImageUnit("colorTex", GL_TEXTURE0);
    _neighborhoodBlendShader->setSamplerTextureImageUnit("blendTex", GL_TEXTURE1);
  }
}

void QueryRenderSMAAPass::setQuality(SMAA_QUALITY_PRESET quality) {
  if (quality != _qualityPreset) {
    // need to reset the shaders

    // TODO(croot): restore active renderer state?
    _rendererPtr->makeActiveOnCurrentThread();

    auto rsrcMgr = _rendererPtr->getResourceManager();

    _qualityPreset = quality;
    _buildShaders(rsrcMgr);
  }
}

void QueryRenderSMAAPass::resize(size_t width, size_t height) {
  _edgeDetectionFbo->resize(width, height);
  _blendingWeightFbo->resize(width, height);

  auto rsrcMgr = _rendererPtr->getResourceManager();

  // rebuild the vbo -- this is done because we may only render within a subimage of the
  // full fbo in a multi-user environment. Therefore we should do the
  // NDC transform in the vertex shader to keep pixels in the right place.
  float fwidth = static_cast<float>(width);
  float fheight = static_cast<float>(height);
  ::Rendering::GL::Resources::GLInterleavedBufferLayoutShPtr bufferLayout(
      new ::Rendering::GL::Resources::GLInterleavedBufferLayout(_rendererPtr->getSupportedExtensions()));
  bufferLayout->addAttribute<float, 2>("pos");
  bufferLayout->addAttribute<float, 2>("texcoord");
  _rectvbo = rsrcMgr->createVertexBuffer<float>(
      {0.0, 0.0, 0.0, 0.0, fwidth, 0.0, 1.0, 0.0, 0.0, fheight, 0.0, 1.0, fwidth, fheight, 1.0, 1.0}, bufferLayout);

  _rendererPtr->bindShader(_edgeDetectionShader);
  _vao->initialize({{_rectvbo, {}}});
}

void QueryRenderSMAAPass::runPass(size_t width,
                                  size_t height,
                                  ::Rendering::GL::GLRenderer* renderer,
                                  ::Rendering::GL::Resources::GLFramebufferShPtr& inputFbo,
                                  ::Rendering::GL::Resources::GLFramebufferShPtr& outputFbo) {
  CHECK(renderer == _rendererPtr.get() && inputFbo && outputFbo);

  CHECK(width <= inputFbo->getWidth() && width <= outputFbo->getWidth() && width <= _edgeDetectionFbo->getWidth() &&
        height <= inputFbo->getHeight() && height <= outputFbo->getHeight() && height <= _edgeDetectionFbo->getHeight())
      << "width: " << width << ", height: " << height << ", inputFbo: [" << inputFbo->getWidth() << ", "
      << inputFbo->getHeight() << "], outputFbo: [" << outputFbo->getWidth() << ", " << outputFbo->getHeight()
      << "], SMAA fbos: [" << _edgeDetectionFbo->getWidth() << ", " << _edgeDetectionFbo->getHeight() << "]";

  auto inputRsrcTex = inputFbo->getAttachmentResource(GL_COLOR_ATTACHMENT0);
  RUNTIME_EX_ASSERT(
      inputRsrcTex && inputRsrcTex->getResourceType() == ::Rendering::GL::Resources::GLResourceType::TEXTURE_2D,
      "GL Resource at GL_COLOR_ATTACHMENT0 attachment in input framebuffer is not a 2D texture. It must be a 2D "
      "texture");

  auto outputRsrcTex = outputFbo->getAttachmentResource(GL_COLOR_ATTACHMENT0);
  RUNTIME_EX_ASSERT(
      outputRsrcTex && outputRsrcTex->getResourceType() == ::Rendering::GL::Resources::GLResourceType::TEXTURE_2D,
      "GL Resource at GL_COLOR_ATTACHMENT0 attachment in output framebuffer is not a 2D texture. It must be a 2D "
      "texture");

  auto fwidth = static_cast<float>(width);
  auto fheight = static_cast<float>(height);

  auto fullwidth = static_cast<float>(_edgeDetectionTex->getWidth());
  auto fullheight = static_cast<float>(_edgeDetectionTex->getHeight());

  std::array<float, 4> viewportMetrics({1.0f / fwidth, 1.0f / fheight, fwidth, fheight});
  std::array<float, 4> fullviewportMetrics({1.0f / fullwidth, 1.0f / fullheight, fullwidth, fullheight});

  renderer->makeActiveOnCurrentThread();
  renderer->disable(GL_BLEND);

  // run the edge detection pass first
  // TODO(croot): use a stencil buffer here to use in
  // the second pass to improve performance a bit
  renderer->bindFramebuffer(_edgeDetectionFbo, ::Rendering::GL::Resources::FboBind::READ_AND_DRAW);
  renderer->setViewport(0, 0, width, height);
  renderer->setClearColor(0, 0, 0, 0);
  renderer->clearAll();

  renderer->bindShader(_edgeDetectionShader);
  renderer->bindVertexArray(_vao);

  _edgeDetectionShader->setUniformAttribute<std::array<float, 4>>("SMAA_RT_METRICS", viewportMetrics);
  _edgeDetectionShader->setUniformAttribute<std::array<float, 4>>("FULL_SMAA_RT_METRICS", fullviewportMetrics);
  _edgeDetectionShader->setSamplerAttribute("colorTex", inputRsrcTex);

  switch (_edgeDetectType) {
    case SMAA_EDGE_DETECTION_TYPE::LUMA:
      _edgeDetectionShader->setSubroutine("detectEdges", "SMAALumaEdgeDetection");
      break;
    case SMAA_EDGE_DETECTION_TYPE::COLOR:
      _edgeDetectionShader->setSubroutine("detectEdges", "SMAAColorEdgeDetection");
      break;
    case SMAA_EDGE_DETECTION_TYPE::DEPTH:
      _edgeDetectionShader->setSubroutine("detectEdges", "SMAADepthEdgeDetection");
      break;
    default:
      THROW_RUNTIME_EX("Edge detection function for edge detect type " +
                       std::to_string(static_cast<int>(_edgeDetectType)) + " is not supported yet.");
  }

  renderer->drawVertexBuffers(GL_TRIANGLE_STRIP);

  // now run the blending weight calculation pass
  renderer->bindFramebuffer(_blendingWeightFbo, ::Rendering::GL::Resources::FboBind::READ_AND_DRAW);
  renderer->clearAll();
  renderer->bindShader(_blendingWeightShader);
  renderer->bindVertexArray(_vao);

  _blendingWeightShader->setUniformAttribute<std::array<float, 4>>("SMAA_RT_METRICS", viewportMetrics);
  _blendingWeightShader->setUniformAttribute<std::array<float, 4>>("FULL_SMAA_RT_METRICS", fullviewportMetrics);
  _blendingWeightShader->setSamplerAttribute("edgeTex", _edgeDetectionTex);
  _blendingWeightShader->setSamplerAttribute("areaTex", _areaTex);
  _blendingWeightShader->setSamplerAttribute("searchTex", _searchTex);

  renderer->drawVertexBuffers(GL_TRIANGLE_STRIP);

  // now do the neighborhood blend pass
  renderer->bindFramebuffer(outputFbo, ::Rendering::GL::Resources::FboBind::READ_AND_DRAW);
  renderer->clearAll();
  renderer->bindShader(_neighborhoodBlendShader);
  renderer->bindVertexArray(_vao);

  _neighborhoodBlendShader->setUniformAttribute<std::array<float, 4>>("SMAA_RT_METRICS", viewportMetrics);
  _neighborhoodBlendShader->setUniformAttribute<std::array<float, 4>>("FULL_SMAA_RT_METRICS", fullviewportMetrics);
  _neighborhoodBlendShader->setSamplerAttribute("colorTex", inputRsrcTex);
  _neighborhoodBlendShader->setSamplerAttribute("blendTex", _blendingWeightTex);

  renderer->drawVertexBuffers(GL_TRIANGLE_STRIP);
}

}  // namespace QueryRenderer

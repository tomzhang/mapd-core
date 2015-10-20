#include "QueryFramebuffer.h"
#include <iostream>
#include <assert.h>

using namespace MapD_Renderer;
using namespace std;

struct Texture2DSamplingProperties {
    GLint minFilter;
    GLint magFilter;
    GLint wrapS;
    GLint wrapT;
};

void resizeTexture2D(GLuint texture, GLint width, GLint height, GLint internalFormat, GLenum pixelFormat, GLenum pixelType) {
    glBindTexture(GL_TEXTURE_2D, texture);
    glTexImage2D(GL_TEXTURE_2D, 0, internalFormat, width, height, 0, pixelFormat, pixelType, NULL);
}

GLuint generateTexture2D(GLint width, GLint height, GLint internalFormat, GLenum pixelFormat, GLenum pixelType) {
    GLuint tex;
    glGenTextures(1, &tex);
    resizeTexture2D(tex, width, height, internalFormat, pixelFormat, pixelType);
    return tex;
}

GLuint generateTexture2D(GLint width, GLint height, GLint internalFormat, GLenum pixelFormat, GLenum pixelType, const Texture2DSamplingProperties& samplingProps) {
    GLuint tex = generateTexture2D(width, height, internalFormat, pixelFormat, pixelType);

    // TODO - use a sampler object instead
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, samplingProps.magFilter);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, samplingProps.minFilter);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, samplingProps.wrapS);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, samplingProps.wrapT);

    return tex;
}

void resizeRenderbuffer(GLuint rbo, GLint width, GLint height, GLint internalFormat) {
    glBindRenderbuffer(GL_RENDERBUFFER, rbo);
    glRenderbufferStorage(GL_RENDERBUFFER, internalFormat, width, height);
}

GLuint generateRenderBuffer(GLint width, GLint height, GLint internalFormat) {
    GLuint rbo;

    glGenRenderbuffers(1, &rbo);
    resizeRenderbuffer(rbo, width, height, internalFormat);

    return rbo;
}


int AttachmentContainer::numColorAttachments = -1;

AttachmentContainer::AttachmentContainer() {
    // need to initialize the number of color attachments for the system
    if (numColorAttachments < 0) {
        glGetIntegerv(GL_MAX_COLOR_ATTACHMENTS, &numColorAttachments);
    }
}

AttachmentContainer::~AttachmentContainer() {
    // std::cerr << "IN AttachmentContainer DESTRUCTOR" << std::endl;
}

inline bool AttachmentContainer::isColorAttachment(GLenum attachment) {
    return (attachment - GL_COLOR_ATTACHMENT0 < numColorAttachments);
}

void AttachmentContainer::addTexture2dAttachment(GLenum attachment, GLuint tex) {
    glFramebufferTexture2D(GL_FRAMEBUFFER, attachment, GL_TEXTURE_2D, tex, 0);

    AttachmentData data = {attachment, tex};
    _attachmentMap.insert(data);

    if (isColorAttachment(attachment)) {
        _dirty = true;
    }
}

void AttachmentContainer::addRenderbufferAttachment(GLenum attachment, GLuint rbo) {
    glFramebufferRenderbuffer(GL_FRAMEBUFFER, attachment, GL_RENDERBUFFER, rbo);

    AttachmentData data = {attachment, rbo};
    _attachmentMap.insert(data);

    if (isColorAttachment(attachment)) {
        _dirty = true;
    }
}

void AttachmentContainer::removeAttachment(GLenum attachment) {
    _attachmentMap.erase(attachment);
    if (isColorAttachment(attachment)) {
        _dirty = true;
    }
}

void AttachmentContainer::enableAttachments() {
    if (_dirty) {
        _activeAttachments.clear();

        AttachmentMap_in_order& inOrder = _attachmentMap.get<inorder>();
        AttachmentMap_in_order::iterator itr;
        for (itr = inOrder.begin(); itr != inOrder.end(); ++itr) {
            if (isColorAttachment(itr->attachmentType)) {
                _activeAttachments.push_back(itr->attachmentType);
            }
        }

        _dirty = false;
    }

    glDrawBuffers(_activeAttachments.size(), &_activeAttachments[0]);
}



QueryFramebuffer::QueryFramebuffer(
    int width,
    int height,
    bool doHitTest,
    bool doDepthTest) :
    _width(width), _height(height),
    _fbo(0), _textureBuffers(MAX_TEXTURE_BUFFERS+1, 0), _renderBuffers(MAX_RENDER_BUFFERS+1, 0) {

    assert(width > 0 && height > 0);
    _init(doHitTest, doDepthTest);
}

QueryFramebuffer::~QueryFramebuffer() {
    glDeleteFramebuffers(1, &_fbo);
    glDeleteTextures(MAX_TEXTURE_BUFFERS+1, &_textureBuffers[0]);
    glDeleteRenderbuffers(MAX_RENDER_BUFFERS+1, &_renderBuffers[0]);
    // std::cerr << "IN QueryFramebuffer DESTRUCTOR" << std::endl;
}

void QueryFramebuffer::_init(bool doHitTest, bool doDepthTest) {
    GLint currFramebuffer, currTexture, currRbo;
    GLuint tex, rbo;

    _doHitTest = doHitTest;
    _doDepthTest = _doDepthTest;

    glGetIntegerv(GL_FRAMEBUFFER_BINDING, &currFramebuffer);
    glGetIntegerv(GL_TEXTURE_BINDING_2D, &currTexture);
    glGetIntegerv(GL_RENDERBUFFER_BINDING, &currRbo);

    glGenFramebuffers(1, &_fbo);
    glBindFramebuffer(GL_FRAMEBUFFER, _fbo);

    // TODO: Make a texture wrapper
    tex = generateTexture2D(_width, _height, GL_RGBA, GL_RGBA, GL_UNSIGNED_BYTE);
    _attachmentManager.addTexture2dAttachment(GL_COLOR_ATTACHMENT0, tex);
    _textureBuffers[COLOR_BUFFER] = tex;

    if (_doHitTest) {
        tex = generateTexture2D(_width, _height, GL_R32UI, GL_RED_INTEGER, GL_UNSIGNED_INT);
        _attachmentManager.addTexture2dAttachment(GL_COLOR_ATTACHMENT1, tex);
        _textureBuffers[ID_BUFFER] = tex;
    }

    if (_doDepthTest) {
        rbo = generateRenderBuffer(_width, _height, GL_DEPTH_COMPONENT);
        _attachmentManager.addRenderbufferAttachment(GL_DEPTH_ATTACHMENT, rbo);
        _renderBuffers[DEPTH_BUFFER] = rbo;
    }

    _attachmentManager.enableAttachments();

    GLenum status = glCheckFramebufferStatus(GL_FRAMEBUFFER);


    if (status != GL_FRAMEBUFFER_COMPLETE) {
        // TODO - throw error?
        std::cerr << "FB error status: 0x" << status << std::endl;
    }
    assert(status == GL_FRAMEBUFFER_COMPLETE);

    glBindFramebuffer(GL_FRAMEBUFFER, currFramebuffer);
    glBindTexture(GL_TEXTURE_2D, currTexture);
    glBindRenderbuffer(GL_RENDERBUFFER, currRbo);
}

void QueryFramebuffer::resize(int width, int height) {
    if (width != _width || height != _height) {
        resizeTexture2D(_textureBuffers[COLOR_BUFFER], width, height, GL_RGBA, GL_RGBA, GL_UNSIGNED_BYTE);

        if (_doHitTest) {
            resizeTexture2D(_textureBuffers[ID_BUFFER], width, height, GL_R32UI, GL_RED_INTEGER, GL_UNSIGNED_INT);
        }

        if (_doDepthTest) {
            resizeRenderbuffer(_renderBuffers[DEPTH_BUFFER], width, height, GL_DEPTH_COMPONENT);
        }

        _width = width;
        _height = height;
    }
}

void QueryFramebuffer::bindToRenderer(BindType bindType) {
    glBindFramebuffer(static_cast<int>(bindType), _fbo);
    // _attachmentManager.enableAttachments();

    // TODO: How do we properly deal with the state machine
    if (_renderBuffers[DEPTH_BUFFER]) {
        glEnable(GL_DEPTH_TEST);
    }
}

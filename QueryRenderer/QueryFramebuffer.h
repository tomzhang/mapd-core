#ifndef QUERY_FRAMEBUFFER_H_
#define QUERY_FRAMEBUFFER_H_

#include <vector>
#include <unordered_map>
#include <GL/glew.h>

namespace MapD_Renderer {

enum TextureBuffers { COLOR_BUFFER=0, ID_BUFFER, MAX_TEXTURE_BUFFERS = ID_BUFFER };
enum RenderBuffers { DEPTH_BUFFER=0, MAX_RENDER_BUFFERS = DEPTH_BUFFER };


///////////////////////////////////////////////////////////////////////
/**
 * AttachmentContainer
 *  Class used to manage attachments of a framebuffer
 */

class AttachmentContainer {
    public:
        AttachmentContainer();
        ~AttachmentContainer();

        static bool isColorAttachment(GLenum attachment);
        void addTexture2dAttachment(GLenum attachment, GLuint tex);
        void addRenderbufferAttachment(GLenum attachment, GLuint rbo);
        void removeAttachment(GLenum attachment);
        void enableAttachments();
    private:
        bool _dirty;
        std::vector<GLenum> _activeAttachments;
        std::unordered_map<GLenum, GLuint> _attachmentMap;
        static int numColorAttachments;
};



///////////////////////////////////////////////////////////////////////
/**
 * QueryFramebuffer
 *  Class used for managing framebuffers for backend rendering of
 *  database queries.
 */

class QueryFramebuffer {
    public:
        QueryFramebuffer(int width, int height, bool doHitTest = false, bool doDepthTest = false);
        ~QueryFramebuffer();

        void resize(int width, int height);
        void bindToRenderer();

    private:
        int _width, _height;

        // framebuffer object id
        GLuint _fbo;

        // texture buffers
        std::vector<GLuint> _textureBuffers;

        // render buffers
        std::vector<GLuint> _renderBuffers;

        // attachment manager
        AttachmentContainer _attachmentManager;

        void _init(bool doHitTest, bool doDepthTest);
};

} // namespace MapD_Renderer


#endif //QUERY_FRAMEBUFFER_H_

#ifndef QUERY_FRAMEBUFFER_H_
#define QUERY_FRAMEBUFFER_H_

#include <vector>
#include <unordered_map>
#include <GL/glew.h>
#include <memory>

#include <boost/multi_index_container.hpp>
#include <boost/multi_index/hashed_index.hpp>
#include <boost/multi_index/random_access_index.hpp>
#include <boost/multi_index/member.hpp>

using namespace ::boost;
using namespace ::boost::multi_index;

namespace MapD_Renderer {

enum TextureBuffers { COLOR_BUFFER=0, ID_BUFFER, MAX_TEXTURE_BUFFERS = ID_BUFFER };
enum RenderBuffers { DEPTH_BUFFER=0, MAX_RENDER_BUFFERS = DEPTH_BUFFER };

enum class BindType { READ=GL_READ_FRAMEBUFFER, DRAW=GL_DRAW_FRAMEBUFFER, READ_AND_DRAW=GL_FRAMEBUFFER};


///////////////////////////////////////////////////////////////////////
/**
 * AttachmentContainer
 *  Class used to manage attachments of a framebuffer
 */

struct AttachmentData {
    GLenum attachmentType;
    GLuint attachedTextureId;
};

struct inorder {};

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
        typedef multi_index_container<
                    AttachmentData,
                    indexed_by<
                        // hashed on name
                        hashed_unique<member<AttachmentData,GLenum,&AttachmentData::attachmentType> >,

                        random_access<tag<inorder>>
                    >
                > AttachmentMap;
        typedef AttachmentMap::index<inorder>::type AttachmentMap_in_order;


        bool _dirty;
        std::vector<GLenum> _activeAttachments;

        AttachmentMap _attachmentMap;

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
        void bindToRenderer(BindType bindType=BindType::READ_AND_DRAW);

        int getWidth() {
            return _width;
        }
        int getHeight() {
            return _height;
        }

    private:
        int _width, _height;
        bool _doHitTest, _doDepthTest;

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

typedef std::unique_ptr<QueryFramebuffer> QueryFramebufferUqPtr;
typedef std::shared_ptr<QueryFramebuffer> QueryFramebufferShPtr;

} // namespace MapD_Renderer


#endif //QUERY_FRAMEBUFFER_H_

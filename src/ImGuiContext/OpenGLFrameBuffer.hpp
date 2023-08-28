#pragma once

namespace PD {

typedef unsigned int GLuint;

struct FrameBuffer {
    GLuint mFBO = 0;
    GLuint mTexId = 0;
    GLuint mDepthId = 0;
    int mWidth = 0;
    int mHeight = 0;

    FrameBuffer()
        : mFBO(0), mDepthId(0), mTexId(0), mWidth(0), mHeight(0) {}

    virtual void create_buffers(int width, int height) = 0;
    virtual void delete_buffers() = 0;
    virtual void bind() = 0;
    virtual void unbind() = 0;
    virtual GLuint get_texture() = 0;
};

struct OpenGLFrameBuffer : public FrameBuffer {
    void create_buffers(int width, int height) override;
    void delete_buffers() override;
    void bind() override;
    void unbind() override;
    GLuint get_texture() override;
};

}
#include "ImGuiContext/OpenGLFrameBuffer.hpp"
#include "spdlog/spdlog.h"

#include <glad/glad.h>

namespace PD {

void OpenGLFrameBuffer::create_buffers(int width, int height)
{
    // spdlog::info(">>> OpenGLFrameBuffer::create_buffers - before");
    mWidth = width;
    mHeight = height;

    if (mFBO) {
        delete_buffers();
    }

    glGenFramebuffers(1, &mFBO);
    glBindFramebuffer(GL_FRAMEBUFFER, mFBO);

    // depth texture
    glGenTextures(1, &mDepthId);
    glBindTexture(GL_TEXTURE_2D, mDepthId);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_DEPTH24_STENCIL8, mWidth, mHeight, 0, GL_DEPTH_STENCIL, GL_UNSIGNED_INT_24_8, NULL);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_R, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_DEPTH_STENCIL_ATTACHMENT, GL_TEXTURE_2D, mDepthId, 0);

    // color texture
    glGenTextures(1, &mTexId);
    glBindTexture(GL_TEXTURE_2D, mTexId);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, mWidth, mHeight, 0, GL_RGBA, GL_UNSIGNED_BYTE, NULL);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_R, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, mTexId, 0);

    GLenum buffers[4] = { GL_COLOR_ATTACHMENT0 };
    glDrawBuffers(mTexId, buffers);

    unbind();

    // spdlog::info(">>> OpenGLFrameBuffer::create_buffers - ok");
}

void OpenGLFrameBuffer::delete_buffers()
{
    if (mFBO) {
        glDeleteFramebuffers(GL_FRAMEBUFFER, &mFBO);
        glDeleteTextures(1, &mTexId);
        glDeleteTextures(1, &mDepthId);
        mTexId = 0;
        mDepthId = 0;
    }
}

void OpenGLFrameBuffer::bind()
{
    glBindFramebuffer(GL_FRAMEBUFFER, mFBO);
    glViewport(0, 0, mWidth, mHeight);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
}

void OpenGLFrameBuffer::unbind()
{
    glBindFramebuffer(GL_FRAMEBUFFER, 0);
}

GLuint OpenGLFrameBuffer::get_texture()
{
    return mTexId;
}

}
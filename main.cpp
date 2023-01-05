#include <iostream>

#include <random>
#include <cmath>

#include <glad/gl.h>
#include <GLFW/glfw3.h>

#include "cuda_ops.cuh"


#define ELEMENTS_PER_THREAD 128

struct cudaGraphicsResource *cudaResource;



void fillRandomBern(uint8_t *a, size_t N, size_t M, float p = 0.2) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::bernoulli_distribution d(p);

    for (std::size_t i = 0; i < N; i++)
        for (std::size_t j = 0; j < M; j++)
            a[i * M + j] = (i == 0 || j == 0 || i == (N - 1) || j == (M - 1)) ? 0 : d(gen);
}


class GameOfLife {
    int N;
    int M;

    uint8_t *frame = nullptr;

    uint8_t *dev_frame = nullptr;
    uint8_t *dev_buffer = nullptr;

public:

    explicit GameOfLife(int height, int width, float p = 0.2) : N(height), M(width) {
        frame = (uint8_t *) malloc(N * M * sizeof(uint8_t));
        fillRandomBern(frame, N, M, p);

        cudaCheck(cudaMalloc((void**)&dev_frame, N * M * sizeof(uint8_t)));
        cudaCheck(cudaMemcpy(dev_frame, frame, N * M * sizeof(uint8_t), cudaMemcpyHostToDevice));

        cudaCheck(cudaMalloc((void**)&dev_buffer, N * M * sizeof(uint8_t)));
    }

    ~GameOfLife() {
        cudaCheck(cudaFree(dev_frame));
        cudaCheck(cudaFree(dev_buffer));

        free(frame);
    }

    uint8_t *getDevFrame() const {
        return dev_frame;
    }

    void runStep() {
        cudaRunStep(dev_frame, dev_buffer, N, M, ELEMENTS_PER_THREAD);
        std::swap(dev_frame, dev_buffer);
    }

    int const& height() const { return N; }
    int const& width()  const { return M; }
};


void resize(GLFWwindow *window, int width, int height) {
    glViewport(0, 0, width, height);
}


void keyboard(GLFWwindow *window, int key, int scancode, int action, int mods) {
    if (key == GLFW_KEY_ESCAPE && action == GLFW_PRESS)
        glfwSetWindowShouldClose(window, true);
}


void windowSetCallbacks(GLFWwindow **window) {
    glfwSetFramebufferSizeCallback(*window, resize);
    glfwSetKeyCallback(*window, keyboard);
}


void initWindow(GLFWwindow **window, int wWidth, int wHeight) {
    glfwInit();
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_COMPAT_PROFILE);

    *window = glfwCreateWindow(wWidth, wHeight, "first", NULL, NULL);

    if (*window == NULL) {
        std::cerr << "Failed to init GLFW window!\n";
        glfwTerminate();
        exit(EXIT_FAILURE);
    }
    glfwMakeContextCurrent(*window);

    glfwSwapInterval(0);

    if (!gladLoadGL((GLADloadfunc) glfwGetProcAddress)) {
        std::cerr << "Failed to init GLAD!\n";
        glfwTerminate();
        exit(EXIT_FAILURE);
    }

    glViewport(0, 0, wWidth, wHeight);

    glShadeModel(GL_FLAT);

    glEnable(GL_POINT_SMOOTH);
    glDisable(GL_BLEND);
}


void renderFrame(GLuint *tex) {
    glClearColor(0.4, 0.5, 0.5, 1);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    glOrtho(-1.f, 1.f, -1.f, 1.f, -1.f, 1.f);

    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();

//    glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);

    glEnable(GL_DEPTH_TEST);

    glEnable(GL_TEXTURE_2D);
    glBindTexture(GL_TEXTURE_2D, *tex);

    glBegin(GL_QUADS);
      glTexCoord2f(0.f, 1.f);
      glVertex2f(-1,-1);

      glTexCoord2f(0.f,0.f);
      glVertex2f(-1,1);

      glTexCoord2f(1.f,0.f);
      glVertex2f(1.f,1.f);

      glTexCoord2f(1.f,1.f);
      glVertex2f(1.f,-1.f);
    glEnd();

    glBindTexture(GL_TEXTURE_2D, 0);
}


void createTexture(GLuint *tex, const int width, const int height) {
    glGenTextures(1, tex);

    glBindTexture(GL_TEXTURE_2D, *tex);

    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);

    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F, width, height, 0, GL_RGBA, GL_FLOAT, NULL);

    cudaCheck(cudaGraphicsGLRegisterImage(
            &cudaResource, *tex, GL_TEXTURE_2D, cudaGraphicsMapFlagsNone
    ));

    glBindTexture(GL_TEXTURE_2D, 0);
}


void processStep(GameOfLife &gol, float *dev_frame) {
    int size = gol.width() * gol.height();

    gol.runStep();

    cudaUpdateTexture(dev_frame, gol.getDevFrame(), size, ELEMENTS_PER_THREAD);

    cudaArray *tex_array;

    cudaCheck(cudaGraphicsMapResources(1, &cudaResource));
    cudaCheck(cudaGraphicsSubResourceGetMappedArray(&tex_array, cudaResource, 0, 0));

    int dsize = size * 4 * sizeof(float);

    cudaCheck(cudaMemcpyToArray(tex_array, 0, 0, dev_frame, dsize, cudaMemcpyDeviceToDevice));
    cudaCheck(cudaGraphicsUnmapResources(1, &cudaResource));
}


int main() {
    int wWidth = 1024;
    int wHeight = 1024;

    GLFWwindow *window;
    initWindow(&window, wWidth, wHeight);
    windowSetCallbacks(&window);

    const int N = 256;
    const int M = 256;

    GLuint texture;
    createTexture(&texture, M, N);

    GameOfLife gol(N, N, 0.5);

    int dsize = N * M * 4 * sizeof(float);

    float *dev_frame;
    cudaCheck(cudaMalloc((void **) &dev_frame, dsize));

    while (!glfwWindowShouldClose(window)) {
        processStep(gol, dev_frame);
        renderFrame(&texture);

        glfwSwapBuffers(window);
        glfwPollEvents();
    }

    cudaFree(dev_frame);
    glDeleteTextures(1, &texture);
    glfwTerminate();
    return 0;
}

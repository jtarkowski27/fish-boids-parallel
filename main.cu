#include <cuda_runtime.h>
#include <cuda_gl_interop.h>

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

#include "boids.cuh"
#include "boids_algorithm.cuh"

#include <helper_cuda.h>

#include <vector_types.h>
#include <helper_math.h>
#include <helper_timer.h>

#include "imgui.h"
#include "imgui_impl_glut.h"
#include "imgui_impl_opengl2.h"

#ifndef OPENGL_HEADERS
#define OPENGL_HEADERS
#include <helper_gl.h>
#if defined(__APPLE__) || defined(MACOSX)
#pragma clang diagnostic ignored "-Wdeprecated-declarations"
#include <GLUT/glut.h>
#ifndef glutCloseFunc
#define glutCloseFunc glutWMCloseFunc
#endif
#else
#include <GL/freeglut.h>
#endif
#endif

#define DT_FACTOR 0.001f

static float time_measured = 0.0f;

static bool compute_on_gpu = true;

static ImVec4 clear_color = ImVec4(0.45f, 0.55f, 0.60f, 1.00f);

#define KEY_ESC 27 /* GLUT doesn't supply this */

boids_s h_boids, d_boids;

int fullscreen = 0;
int mouseDown = 0;

float xrot = 180.0f;
float yrot = 220.0f;

float xdiff = 100.0f;
float ydiff = 100.0f;

float tra_x = 0.0f;
float tra_y = 0.0f;
float tra_z = 0.0f;

float grow_shrink = 70.0f;
float resize_f = 1.0f;

clock_t start, end;

StopWatchInterface *timer = NULL;


void init_boids()
{
    sdkCreateTimer(&timer);

    malloc_data(h_boids, BOIDS_N, cudaMemoryType::cudaMemoryTypeHost);
    malloc_data(d_boids, BOIDS_N, cudaMemoryType::cudaMemoryTypeDevice);

    randomize_data(h_boids);

    memcpy_data(d_boids, h_boids, cudaMemcpyKind::cudaMemcpyHostToDevice);

    start = clock();
    timer->start();

    h_boids.n = BOIDS_N / 2;
}

void free_boids()
{
    free_data(h_boids, cudaMemoryType::cudaMemoryTypeHost);
    free_data(d_boids, cudaMemoryType::cudaMemoryTypeDevice);

    sdkDeleteTimer(&timer);
}

void draw_indices()
{
    end = clock();

    double elapsed = 8 * double(end - start) / CLOCKS_PER_SEC;

    float predators_offset = min(h_boids.predators_count, h_boids.n) * TRIANGLES_PER_FISH * DIMENSIONS_COUNT;

    glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);

    glColor3f(0.1f, 0.1f, 0.1f);

    glEnableClientState(GL_VERTEX_ARRAY);
    glVertexPointer(3, GL_FLOAT, 0, h_boids.geometry);
    glDrawArrays(GL_TRIANGLES, predators_offset, h_boids.n * TRIANGLES_PER_FISH * DIMENSIONS_COUNT - predators_offset);

    float r = sin(elapsed);
    r *= r;
    float b = cos(elapsed);
    b *= b;

    glColor3f(r, 0.1f, b);

    glEnableClientState(GL_VERTEX_ARRAY);
    glVertexPointer(3, GL_FLOAT, 0, h_boids.geometry);
    glDrawArrays(GL_TRIANGLES, 0, predators_offset);

    glDisableClientState(GL_VERTEX_ARRAY);
}

bool debug = true;

void run_algorithm()
{
    dim3 threadsPerBlock(256, 1, 1);
    dim3 numBlocks((h_boids.n / threadsPerBlock.x) + 1, 1, 1);

    cudaEvent_t start, stop;

    memcpy_data(d_boids, h_boids, cudaMemcpyKind::cudaMemcpyHostToDevice);

    timer->stop();
    float dt = timer->getTime() * DT_FACTOR;
    timer->reset();
    timer->start();

    checkCudaErrors(cudaEventCreate(&start));
    checkCudaErrors(cudaEventCreate(&stop));
    checkCudaErrors(cudaEventRecord(start, 0));

    if (compute_on_gpu)
    {
        boids_step_gpu<<<numBlocks, threadsPerBlock>>>(d_boids, dt, debug);
        checkCudaErrors(cudaGetLastError());
        checkCudaErrors(cudaDeviceSynchronize());
    }
    else
    {
        boids_step_cpu(h_boids, dt, debug);
    }

    checkCudaErrors(cudaEventRecord(stop, 0));
    checkCudaErrors(cudaEventSynchronize(stop));
    checkCudaErrors(cudaEventElapsedTime(&time_measured, start, stop));

    debug = false;

    if (compute_on_gpu)
    {
        memcpy_data(h_boids, d_boids, cudaMemcpyKind::cudaMemcpyDeviceToHost);
    }
}

void draw_cube_borders()
{
    glColor3f(0.3f, 0.3f, 0.3f);

    glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);
    glBegin(GL_QUADS);

    glVertex3f(1.0f, 1.0f, -1.0f);
    glVertex3f(-1.0f, 1.0f, -1.0f);
    glVertex3f(-1.0f, 1.0f, 1.0f);
    glVertex3f(1.0f, 1.0f, 1.0f);

    glVertex3f(1.0f, -1.0f, 1.0f);
    glVertex3f(-1.0f, -1.0f, 1.0f);
    glVertex3f(-1.0f, -1.0f, -1.0f);
    glVertex3f(1.0f, -1.0f, -1.0f);

    glVertex3f(1.0f, 1.0f, 1.0f);
    glVertex3f(-1.0f, 1.0f, 1.0f);
    glVertex3f(-1.0f, -1.0f, 1.0f);
    glVertex3f(1.0f, -1.0f, 1.0f);

    glVertex3f(1.0f, -1.0f, -1.0f);
    glVertex3f(-1.0f, -1.0f, -1.0f);
    glVertex3f(-1.0f, 1.0f, -1.0f);
    glVertex3f(1.0f, 1.0f, -1.0f);

    glVertex3f(-1.0f, 1.0f, 1.0f);
    glVertex3f(-1.0f, 1.0f, -1.0f);
    glVertex3f(-1.0f, -1.0f, -1.0f);
    glVertex3f(-1.0f, -1.0f, 1.0f);

    glVertex3f(1.0f, 1.0f, -1.0f);
    glVertex3f(1.0f, 1.0f, 1.0f);
    glVertex3f(1.0f, -1.0f, 1.0f);
    glVertex3f(1.0f, -1.0f, -1.0f);

    glEnd(); 

    glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
}

void render_scene()
{
    glTranslatef(tra_x, tra_y, tra_z);

    glColor3f(0.1f, 0.1f, 0.1f);

    draw_cube_borders();
}

void display_controls()
{
    ImGui::Begin("Simulation Options", nullptr, ImGuiWindowFlags_AlwaysAutoResize);

    ImGui::Text("Application average %.3f ms/frame (%.1f FPS)", 1000.0f / ImGui::GetIO().Framerate, ImGui::GetIO().Framerate);
    ImGui::Text("Algorithm execution time %.3f ms \n", h_boids.controls.is_running ? time_measured : 0.0f);

    if (ImGui::Checkbox("GPU Computation", &compute_on_gpu))
        h_boids.n = !compute_on_gpu ? min(h_boids.n, 1000) : h_boids.n;
    ImGui::Checkbox("Running", &h_boids.controls.is_running);
    ImGui::SliderFloat("Animation speed", &h_boids.controls.animation_speed, 0.1f, 5.0f);
    ImGui::SliderInt("Fish count", &h_boids.n, 12, compute_on_gpu ? 20000 : 500, "%d", ImGuiSliderFlags_Logarithmic);

    ImGui::SliderFloat("Fish size", &h_boids.controls.fish_size, 0.001f, 0.05f);

    ImGui::Spacing();
    ImGui::Separator();

    ImGui::Text("Parameters");

    ImGui::SliderFloat("Alignment", &h_boids.controls.alignment, 0.01f, 25.0f, "%.3f", ImGuiSliderFlags_Logarithmic);
    ImGui::SliderFloat("Separation", &h_boids.controls.separation, 0.01f, 25.0f, "%.3f", ImGuiSliderFlags_Logarithmic);
    ImGui::SliderFloat("Cohesion", &h_boids.controls.cohesion, 0.01f, 50.0f, "%.3f", ImGuiSliderFlags_Logarithmic);

    ImGui::Text("Predators");
    ImGui::SliderFloat("Avoidance", &h_boids.controls.predator_avoidance, 0.1f, 100.0f, "%.3f", ImGuiSliderFlags_Logarithmic);
    ImGui::SliderFloat("Avoidance radius", &h_boids.controls.predator_avoidance_radius, 0.01f, 1.0f, "%.3f", ImGuiSliderFlags_Logarithmic);
    ImGui::SliderInt("Predators count", &h_boids.predators_count, 0, 1000, "%d", ImGuiSliderFlags_Logarithmic);

    ImGui::Text("Camera");

    ImGui::SliderFloat("Yaw", &xrot, 0.0f, 360.0f);
    ImGui::SliderFloat("Pitch", &yrot, 0.0f, 360.0f);

    ImGui::Spacing();

    ImGui::End();
}

void glut_display_func()
{
    run_algorithm();

    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    glLoadIdentity();

    gluLookAt(
        0.0f, 0.0f, 3.0f,
        0.0f, 0.0f, 0.0f,
        0.0f, 1.0f, 0.0f);

    glRotatef(xrot, 1.0f, 0.0f, 0.0f);
    glRotatef(yrot, 0.0f, 1.0f, 0.0f);
    render_scene();
    draw_indices();

    // Start the ImGui frame
    ImGui_ImplOpenGL2_NewFrame();
    ImGui_ImplGLUT_NewFrame();

    display_controls();

    // Rendering
    ImGui::Render();
    ImGuiIO &io = ImGui::GetIO();
    glViewport(0, 0, (GLsizei)io.DisplaySize.x, (GLsizei)io.DisplaySize.y);

    ImGui_ImplOpenGL2_RenderDrawData(ImGui::GetDrawData());

    // glFlush();
    glutSwapBuffers();
    glutPostRedisplay();
}

int init(void)
{
    glClearColor(0.93f, 0.93f, 0.93f, 0.0f);

    glEnable(GL_DEPTH_TEST);
    glDepthFunc(GL_LEQUAL);
    glClearDepth(1.0f);

    return 1;
}

void resize(int w, int h)
{
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();

    glViewport(0, 0, w, h);

    gluPerspective(grow_shrink, resize_f * w / h, resize_f, 100 * resize_f);

    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();

    ImGui_ImplGLUT_ReshapeFunc(w, h);
}

void specialKeyboard(int key, int x, int y)
{
    if (key == GLUT_KEY_F1)
    {
        fullscreen = !fullscreen;

        if (fullscreen)
            glutFullScreen();
        else
        {
            glutReshapeWindow(500, 500);
            glutPositionWindow(50, 50);
        }
    }

    ImGui_ImplGLUT_SpecialFunc(key, x, y);
}

int main(int argc, char **argv)
{
    init_boids();

    // Create GLUT window
    glutInit(&argc, argv);
#ifdef __FREEGLUT_EXT_H__
    glutSetOption(GLUT_ACTION_ON_WINDOW_CLOSE, GLUT_ACTION_GLUTMAINLOOP_RETURNS);
#endif
    glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE | GLUT_MULTISAMPLE);
    glutInitWindowSize(1280, 720);
    glutCreateWindow("Parallel fish boids");

    glutDisplayFunc(glut_display_func);

    // Setup ImGui context
    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGuiIO &io = ImGui::GetIO();
    (void)io;
    io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard; // Enable Keyboard Controls

    // Setup ImGui style
    ImGui::StyleColorsDark();

    // Setup Platform/Renderer backends
    ImGui_ImplGLUT_Init();

    glutMotionFunc(ImGui_ImplGLUT_MotionFunc);
    glutPassiveMotionFunc(ImGui_ImplGLUT_MotionFunc);
    glutMouseFunc(ImGui_ImplGLUT_MouseFunc);
#ifdef __FREEGLUT_EXT_H__
    glutMouseWheelFunc(ImGui_ImplGLUT_MouseWheelFunc);
#endif
    glutKeyboardFunc(ImGui_ImplGLUT_KeyboardFunc);
    glutKeyboardUpFunc(ImGui_ImplGLUT_KeyboardUpFunc);
    glutSpecialUpFunc(ImGui_ImplGLUT_SpecialUpFunc);

    glutSpecialFunc(specialKeyboard);
    glutReshapeFunc(resize);

    ImGui_ImplOpenGL2_Init();

    if (!init())
        return 1;

    glutMainLoop();

    // Cleanup
    ImGui_ImplOpenGL2_Shutdown();
    ImGui_ImplGLUT_Shutdown();
    ImGui::DestroyContext();

    free_boids();

    return 0;
}

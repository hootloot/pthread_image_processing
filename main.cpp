#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image.h"
#include "stb_image_write.h"
#include "filters.h"

#include <pthread.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <queue>
#include <vector>
#include <chrono>
#include <sys/stat.h>

#define TILE_SIZE 256

typedef void (*FilterFunc)(unsigned char*, int, int, int);

FilterFunc getFilter(const std::string& name) {
    if (name == "grayscale") return filterGrayscale;
    if (name == "threshold") return filterThreshold;
    if (name == "sharpen")   return filterSharpen;
    if (name == "edge")      return filterEdgeDetect;
    if (name == "blur")      return filterGaussianBlur;
    if (name == "none")      return nullptr;
    return nullptr;
}

struct TileTask {
    unsigned char* src;
    int imgW, imgH, channels;
    int tileX, tileY;
    int tileW, tileH;
    std::string outPath;
    FilterFunc filter;
};

class WorkQueue {
    std::queue<TileTask> q;
    pthread_mutex_t mtx;
    pthread_cond_t cv;
    bool done;

public:
    WorkQueue() : done(false) {
        pthread_mutex_init(&mtx, nullptr);
        pthread_cond_init(&cv, nullptr);
    }

    ~WorkQueue() {
        pthread_mutex_destroy(&mtx);
        pthread_cond_destroy(&cv);
    }

    void push(const TileTask& t) {
        pthread_mutex_lock(&mtx);
        q.push(t);
        pthread_cond_signal(&cv);
        pthread_mutex_unlock(&mtx);
    }

    bool pop(TileTask& t) {
        pthread_mutex_lock(&mtx);
        while (q.empty() && !done)
            pthread_cond_wait(&cv, &mtx);

        if (q.empty()) {
            pthread_mutex_unlock(&mtx);
            return false;
        }
        t = q.front();
        q.pop();
        pthread_mutex_unlock(&mtx);
        return true;
    }

    void shutdown() {
        pthread_mutex_lock(&mtx);
        done = true;
        pthread_cond_broadcast(&cv);
        pthread_mutex_unlock(&mtx);
    }
};

static WorkQueue* workQueue = nullptr;

unsigned char* extractTile(const TileTask& t) {
    unsigned char* buf = new unsigned char[t.tileW * t.tileH * t.channels];

    int x0 = t.tileX * TILE_SIZE;
    int y0 = t.tileY * TILE_SIZE;

    for (int row = 0; row < t.tileH; row++) {
        int srcOff = ((y0 + row) * t.imgW + x0) * t.channels;
        int dstOff = row * t.tileW * t.channels;
        memcpy(buf + dstOff, t.src + srcOff, t.tileW * t.channels);
    }
    return buf;
}

void* workerFunc(void* arg) {
    int id = *((int*)arg);
    delete (int*)arg;

    TileTask task;
    while (workQueue->pop(task)) {
        unsigned char* data = extractTile(task);

        if (task.filter)
            task.filter(data, task.tileW, task.tileH, task.channels);

        if (stbi_write_png(task.outPath.c_str(),
                           task.tileW, task.tileH,
                           task.channels, data,
                           task.tileW * task.channels)) {
            printf("[thread %d] wrote %s (%dx%d)\n",
                   id, task.outPath.c_str(), task.tileW, task.tileH);
        } else {
            fprintf(stderr, "[thread %d] FAILED to write %s\n",
                    id, task.outPath.c_str());
        }

        delete[] data;
    }

    printf("[thread %d] done\n", id);
    return nullptr;
}

class ThreadPool {
    std::vector<pthread_t> threads;
    int count;
public:
    ThreadPool(int n) : count(n), threads(n) {
        for (int i = 0; i < n; i++) {
            int* id = new int(i);
            if (pthread_create(&threads[i], nullptr, workerFunc, id)) {
                fprintf(stderr, "couldn't create thread %d\n", i);
                delete id;
            }
        }
        printf("spun up %d threads\n", n);
    }

    void join() {
        for (int i = 0; i < count; i++)
            pthread_join(threads[i], nullptr);
        printf("all threads joined\n");
    }
};

static void mkdirSafe(const std::string& path) {
#ifdef _WIN32
    _mkdir(path.c_str());
#else
    mkdir(path.c_str(), 0755);
#endif
}

int main(int argc, char* argv[]) {
    if (argc < 5) {
        fprintf(stderr, "usage: %s <input> <outdir> <threads> <filter>\n", argv[0]);
        return 1;
    }

    const char* inputPath  = argv[1];
    std::string outDir     = argv[2];
    int nThreads           = atoi(argv[3]);
    std::string filterName = argv[4];

    if (nThreads < 1) {
        fprintf(stderr, "need at least 1 thread\n");
        return 1;
    }

    FilterFunc filter = getFilter(filterName);
    if (!filter && filterName != "none") {
        fprintf(stderr, "unknown filter '%s'\n", filterName.c_str());
        return 1;
    }

    if (!outDir.empty() && outDir.back() != '/' && outDir.back() != '\\')
        outDir += '/';
    mkdirSafe(outDir);

    auto t0 = std::chrono::high_resolution_clock::now();

    printf("loading %s...\n", inputPath);
    int w, h, ch;
    unsigned char* img = stbi_load(inputPath, &w, &h, &ch, 0);
    if (!img) {
        fprintf(stderr, "can't load image: %s (%s)\n",
                inputPath, stbi_failure_reason());
        return 1;
    }
    printf("loaded: %dx%d, %d ch\n", w, h, ch);
    printf("filter: %s\n", filterName.c_str());

    int nx = (w + TILE_SIZE - 1) / TILE_SIZE;
    int ny = (h + TILE_SIZE - 1) / TILE_SIZE;
    int total = nx * ny;
    printf("grid: %dx%d = %d tiles\n", nx, ny, total);

    workQueue = new WorkQueue();
    ThreadPool pool(nThreads);

    for (int ty = 0; ty < ny; ty++) {
        for (int tx = 0; tx < nx; tx++) {
            int sx = tx * TILE_SIZE;
            int sy = ty * TILE_SIZE;

            TileTask t;
            t.src      = img;
            t.imgW     = w;
            t.imgH     = h;
            t.channels = ch;
            t.tileX    = tx;
            t.tileY    = ty;
            t.tileW    = std::min(TILE_SIZE, w - sx);
            t.tileH    = std::min(TILE_SIZE, h - sy);
            t.filter   = filter;

            char fname[256];
            snprintf(fname, sizeof(fname), "%stile_%02d_%02d.png",
                     outDir.c_str(), ty, tx);
            t.outPath = fname;

            workQueue->push(t);
        }
    }
    printf("queued %d tasks\n", total);

    workQueue->shutdown();
    pool.join();

    stbi_image_free(img);
    delete workQueue;

    auto t1 = std::chrono::high_resolution_clock::now();
    long long ms = std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count();

    printf("\ndone\n");
    printf("filter: %s | tiles: %d | threads: %d | time: %lld ms\n",
           filterName.c_str(), total, nThreads, ms);

    return 0;
}

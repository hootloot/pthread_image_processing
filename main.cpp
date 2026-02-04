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

constexpr int TILE_SIZE = 256;

using FilterFunc = void (*)(unsigned char*, int, int, int);

// filters
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
    unsigned char* srcImage;
    int imgWidth;
    int imgHeight;
    int channels;
    int tileX;
    int tileY;
    int tileW;
    int tileH;
    std::string outputPath;
    FilterFunc filter;  
};


class WorkQueue {
private:
    std::queue<TileTask> tasks;
    pthread_mutex_t mutex;
    pthread_cond_t condVar;
    bool shutdown;

public:
    WorkQueue() : shutdown(false) {
        pthread_mutex_init(&mutex, nullptr);
        pthread_cond_init(&condVar, nullptr);
    }

    ~WorkQueue() {
        pthread_mutex_destroy(&mutex);
        pthread_cond_destroy(&condVar);
    }

    void push(const TileTask& task) {
        pthread_mutex_lock(&mutex);
        tasks.push(task);
        pthread_cond_signal(&condVar);
        pthread_mutex_unlock(&mutex);
    }

    bool pop(TileTask& task) {
        pthread_mutex_lock(&mutex);
        while (tasks.empty() && !shutdown) {
            pthread_cond_wait(&condVar, &mutex);
        }
        if (tasks.empty()) {
            pthread_mutex_unlock(&mutex);
            return false;
        }
        task = tasks.front();
        tasks.pop();
        pthread_mutex_unlock(&mutex);
        return true;
    }

    void signalShutdown() {
        pthread_mutex_lock(&mutex);
        shutdown = true;
        pthread_cond_broadcast(&condVar);
        pthread_mutex_unlock(&mutex);
    }
};


WorkQueue* g_workQueue = nullptr;


unsigned char* extractTile(const TileTask& task) {
    int tileBytes = task.tileW * task.tileH * task.channels;
    unsigned char* tile = new unsigned char[tileBytes];
    
    int startX = task.tileX * TILE_SIZE;
    int startY = task.tileY * TILE_SIZE;
    
    for (int y = 0; y < task.tileH; ++y) {
        int srcY = startY + y;
        int srcIdx = (srcY * task.imgWidth + startX) * task.channels;
        int dstIdx = y * task.tileW * task.channels;
        memcpy(tile + dstIdx, task.srcImage + srcIdx, task.tileW * task.channels);
    }
    
    return tile;
}

void* workerThread(void* arg) {
    int threadId = *static_cast<int*>(arg);
    delete static_cast<int*>(arg);
    
    TileTask task;
    while (g_workQueue->pop(task)) {
        unsigned char* tileData = extractTile(task);
        
        if (task.filter) {
            task.filter(tileData, task.tileW, task.tileH, task.channels);
        }
        
        int success = stbi_write_png(
            task.outputPath.c_str(),
            task.tileW,
            task.tileH,
            task.channels,
            tileData,
            task.tileW * task.channels
        );
        
        if (success) {
            printf("[Thread %d] Saved: %s (%dx%d)\n", 
                   threadId, task.outputPath.c_str(), task.tileW, task.tileH);
        } else {
            fprintf(stderr, "[Thread %d] Failed to save: %s\n", 
                    threadId, task.outputPath.c_str());
        }
        
        delete[] tileData;
    }
    
    printf("[Thread %d] Shutting down\n", threadId);
    return nullptr;
}

class ThreadPool {
private:
    std::vector<pthread_t> threads;
    int numThreads;

public:
    ThreadPool(int n) : numThreads(n) {
        threads.resize(n);
        for (int i = 0; i < n; ++i) {
            int* threadId = new int(i);
            if (pthread_create(&threads[i], nullptr, workerThread, threadId) != 0) {
                fprintf(stderr, "Failed to create thread %d\n", i);
                delete threadId;
            }
        }
        printf("Created %d worker threads\n", n);
    }

    void waitAll() {
        for (int i = 0; i < numThreads; ++i) {
            pthread_join(threads[i], nullptr);
        }
        printf("All threads joined\n");
    }
};

void ensureDirectory(const std::string& path) {
    #ifdef _WIN32
    _mkdir(path.c_str());
    #else
    mkdir(path.c_str(), 0755);
    #endif
}








int main(int argc, char* argv[]) {


    const char* inputPath = argv[1];
    std::string outputDir = argv[2];
    int threadCount = atoi(argv[3]);
    std::string filterName = argv[4];

    if (threadCount < 1) {
        fprintf(stderr, "Thread count must be at least 1\n");
        return 1;
    }

    FilterFunc filter = getFilter(filterName);
    if (!filter && filterName != "none") {
        fprintf(stderr, "Unknown filter: %s\n", filterName.c_str());
        return 1;
    }

    if (!outputDir.empty() && outputDir.back() != '/' && outputDir.back() != '\\') {
        outputDir += '/';
    }
    ensureDirectory(outputDir);

    auto startTime = std::chrono::high_resolution_clock::now();

    printf("Loading image: %s\n", inputPath);
    int width, height, channels;
    unsigned char* image = stbi_load(inputPath, &width, &height, &channels, 0);
    
    if (!image) {
        fprintf(stderr, "Failed to load image: %s\n", inputPath);
        fprintf(stderr, "Reason: %s\n", stbi_failure_reason());
        return 1;
    }
    printf("Image loaded: %dx%d, %d channels\n", width, height, channels);
    printf("Applying filter: %s\n", filterName.c_str());

    int tilesX = (width + TILE_SIZE - 1) / TILE_SIZE;
    int tilesY = (height + TILE_SIZE - 1) / TILE_SIZE;
    int totalTiles = tilesX * tilesY;
    printf("Splitting into %dx%d grid (%d tiles)\n", tilesX, tilesY, totalTiles);

    g_workQueue = new WorkQueue();
    ThreadPool pool(threadCount);

    for (int ty = 0; ty < tilesY; ++ty) {
        for (int tx = 0; tx < tilesX; ++tx) {
            TileTask task;
            task.srcImage = image;
            task.imgWidth = width;
            task.imgHeight = height;
            task.channels = channels;
            task.tileX = tx;
            task.tileY = ty;
            task.filter = filter; 
            
            int startX = tx * TILE_SIZE;
            int startY = ty * TILE_SIZE;
            task.tileW = std::min(TILE_SIZE, width - startX);
            task.tileH = std::min(TILE_SIZE, height - startY);
            
            char filename[256];
            snprintf(filename, sizeof(filename), "%stile_%02d_%02d.png", 
                     outputDir.c_str(), ty, tx);
            task.outputPath = filename;
            
            g_workQueue->push(task);
        }
    }

    printf("Queued %d tile tasks\n", totalTiles);

    g_workQueue->signalShutdown();
    pool.waitAll();

    stbi_image_free(image);
    delete g_workQueue;

    auto endTime = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(
        endTime - startTime
    ).count();

    printf("Processing complete!\n");
    printf("Filter: %s\n", filterName.c_str());
    printf("Total tiles: %d\n", totalTiles);
    printf("Threads used: %d\n", threadCount);
    printf("Elapsed time: %lld ms\n", duration);

    return 0;
}
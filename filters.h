#pragma once
#include <cstring>
#include <cmath>
#include <algorithm>

// Grayscale
void filterGrayscale(unsigned char* data, int w, int h, int channels) {
    for (int i = 0; i < w * h; ++i) {
        int idx = i * channels;
        unsigned char r = data[idx];
        unsigned char g = (channels > 1) ? data[idx + 1] : r;
        unsigned char b = (channels > 2) ? data[idx + 2] : r;
        
        unsigned char gray = static_cast<unsigned char>(
            0.299 * r + 0.587 * g + 0.114 * b + 0.5
        );
        
        data[idx] = gray;
        if (channels > 1) data[idx + 1] = gray;
        if (channels > 2) data[idx + 2] = gray;
    }
}

// Threshold 
void filterThreshold(unsigned char* data, int w, int h, int channels) {
    int cutoff = 128;
    for (int i = 0; i < w * h; ++i) {
        int idx = i * channels;
        unsigned char r = data[idx];
        unsigned char g = (channels > 1) ? data[idx + 1] : r;
        unsigned char b = (channels > 2) ? data[idx + 2] : r;
        
        unsigned char gray = static_cast<unsigned char>(
            0.299 * r + 0.587 * g + 0.114 * b + 0.5
        );
        unsigned char val = (gray >= cutoff) ? 255 : 0;
        
        data[idx] = val;
        if (channels > 1) data[idx + 1] = val;
        if (channels > 2) data[idx + 2] = val;
    }
}


// Gaussian Blur 
void filterGaussianBlur(unsigned char* data, int w, int h, int channels) {
    int passes = 3;
    
    int kernel[5][5] = {
        {1,  4,  6,  4, 1},
        {4, 16, 24, 16, 4},
        {6, 24, 36, 24, 6},
        {4, 16, 24, 16, 4},
        {1,  4,  6,  4, 1}
    };
    int kernelSum = 256;
    int radius = 2;
    
    for (int p = 0; p < passes; ++p) {
        unsigned char* copy = new unsigned char[w * h * channels];
        memcpy(copy, data, w * h * channels);
        
        for (int y = radius; y < h - radius; ++y) {
            for (int x = radius; x < w - radius; ++x) {
                for (int c = 0; c < std::min(channels, 3); ++c) {
                    int sum = 0;
                    for (int ky = -radius; ky <= radius; ++ky) {
                        for (int kx = -radius; kx <= radius; ++kx) {
                            int idx = ((y + ky) * w + (x + kx)) * channels + c;
                            sum += copy[idx] * kernel[ky + radius][kx + radius];
                        }
                    }
                    int idx = (y * w + x) * channels + c;
                    data[idx] = static_cast<unsigned char>(sum / kernelSum);
                }
            }
        }
        
        delete[] copy;
    }
}

// Sharpen
void filterSharpen(unsigned char* data, int w, int h, int channels) {
    unsigned char* copy = new unsigned char[w * h * channels];
    memcpy(copy, data, w * h * channels);
    
    // Sharpening kernel
    int kernel[3][3] = {
        { 0, -1,  0},
        {-1,  5, -1},
        { 0, -1,  0}
    };
    
    for (int y = 1; y < h - 1; ++y) {
        for (int x = 1; x < w - 1; ++x) {
            for (int c = 0; c < std::min(channels, 3); ++c) {
                int sum = 0;
                for (int ky = -1; ky <= 1; ++ky) {
                    for (int kx = -1; kx <= 1; ++kx) {
                        int idx = ((y + ky) * w + (x + kx)) * channels + c;
                        sum += copy[idx] * kernel[ky + 1][kx + 1];
                    }
                }
                int idx = (y * w + x) * channels + c;
                data[idx] = static_cast<unsigned char>(std::clamp(sum, 0, 255));
            }
        }
    }
    
    delete[] copy;
}

// Edge Detection 
void filterEdgeDetect(unsigned char* data, int w, int h, int channels) {
    unsigned char* copy = new unsigned char[w * h * channels];
    memcpy(copy, data, w * h * channels);
    
    // Convert to grayscale first
    filterGrayscale(copy, w, h, channels);
    
    int sobelX[3][3] = {
        {-1, 0, 1},
        {-2, 0, 2},
        {-1, 0, 1}
    };
    
    int sobelY[3][3] = {
        {-1, -2, -1},
        { 0,  0,  0},
        { 1,  2,  1}
    };
    
    for (int y = 1; y < h - 1; ++y) {
        for (int x = 1; x < w - 1; ++x) {
            int gx = 0, gy = 0;
            
            for (int ky = -1; ky <= 1; ++ky) {
                for (int kx = -1; kx <= 1; ++kx) {
                    int idx = ((y + ky) * w + (x + kx)) * channels;
                    int pixel = copy[idx];
                    gx += pixel * sobelX[ky + 1][kx + 1];
                    gy += pixel * sobelY[ky + 1][kx + 1];
                }
            }
            
            int magnitude = static_cast<int>(std::sqrt(gx * gx + gy * gy));
            unsigned char edge = static_cast<unsigned char>(std::clamp(magnitude, 0, 255));
            
            int idx = (y * w + x) * channels;
            data[idx] = edge;
            if (channels > 1) data[idx + 1] = edge;
            if (channels > 2) data[idx + 2] = edge;
        }
    }
    
    delete[] copy;
}
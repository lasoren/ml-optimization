#include <math.h>
#include <stdio.h>

typedef float data_t;
// Compute the euclidean distance between two multi-dimensional vectors.
// Note: x and y must be the same length.
float euclid_distance(const data_t* x, const data_t* y, const int length);

int main() {
    data_t x[] = {0, 0};
    data_t y[] = {1, 1};
    printf("%f\n", euclid_distance(x, y, 2));
}

float euclid_distance(const data_t* x, const data_t* y, const int length) {
    float dist = 0;
    for (int i = 0; i < length; i++) {
        dist += pow(fabs(x[i] - y[i]), 2);
    } 
    return sqrt(dist);
}

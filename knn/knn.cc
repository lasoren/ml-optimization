#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <vector>
#include <algorithm>
#include <unordered_map>
#include <map>
#include <iostream>
#include <utility>
#include <functional>
#include <numeric>

#include "utils.h"

typedef float data_t;

using namespace std;

// Perform K nearest neighbor classification on all x data points.
void perform_knn(const int k,
         const data_t* x,
         const data_t* labeled,
         const int* labels,
         const int dim,
         const int x_length,
         const int labeled_length,
         int* x_pred);

// Perform K nearest neighbor classification on all one data point at index i.
void knn(const int k,
         const data_t* x,
         const int i,
         const data_t* labeled,
         const int* labels,
         const int dim,
         const int labeled_length,
         int* x_pred);

// Compute the euclidean distance between two multi-dimensional vectors.
// Note: x and y must be the same length.
float euclid_distance(const data_t* x, const data_t* y, const int length);

// Computes the hamming distance between any multi-dimensional vectors where
// where the difference between each dimension determines the hamming distance.
data_t hamming_distance(const data_t* x, const data_t* y, const int length);

// Given the distances from this test point to all of the labeled points
// and the number of nearest neighbors to consider, predict this point's
// label.
int predict_class(const float* distances,
                  const int* labels,
                  const int length,
                  const int k);

// Compute the mode of the nearest neighbor labels.
int mode_labels(const vector< pair<int, float> >& labels);

// Used for sorting pairs of integer and float values.
bool compare_labels(const pair<int, float>&i, const pair<int, float>&j);

// Get fields from a line and fill an array with the values.
const char* getfields(char* line, int num);

int main() {
    const int x_length = 500;
    const int x_dim = 2;
    int i, j;

    const int labeled_length = 200;

    data_t x[x_length*x_dim];
    data_t labeled[labeled_length*x_dim];
    int x_labels[x_length];
    int x_pred[x_length];
    int labels[labeled_length];

    int line_counter = 0;

    FILE* stream = fopen("train2d.csv", "r");
    char line[1024];
    while (fgets(line, 1024, stream))
    {
        char* tmp = strdup(line);
        int idx = line_counter*x_dim;
        labeled[idx] = strtod(getfield(tmp, 1), NULL);
        tmp = strdup(line);
        labeled[idx + 1] = strtod(getfield(tmp, 2), NULL);
        tmp = strdup(line);
        labels[line_counter] = strtod(getfield(tmp, 3), NULL);
        free(tmp);
        line_counter++;
    }

    stream = fopen("test2d.csv", "r");
    line_counter = 0;
    while (fgets(line, 1024, stream))
    {
        char* tmp = strdup(line);
        int idx = line_counter*x_dim;
        x[idx] = strtod(getfield(tmp, 1), NULL);
        tmp = strdup(line);
        x[idx + 1] = strtod(getfield(tmp, 2), NULL);
        tmp = strdup(line);
        x_labels[line_counter] = strtod(getfield(tmp, 3), NULL);
        free(tmp);
        line_counter++;
    }
/*
    for (i = 0; i < x_length; i++) {
        for (j = 0; j < x_dim; j++) {
            printf("%f,", x[i*x_dim+j]);
        }
        printf("%d",x_labels[i]);
        printf("\n");
    }
*/
    perform_knn(6, x, labeled, labels, x_dim, x_length, labeled_length, x_pred);
    int count_right = 0;    
    for (i = 0; i < x_length; i++) {
        if (x_pred[i] == x_labels[i]) {
            count_right++;
        }
    } 
    cout << "Percentage correctly classified: " <<
        100*count_right / (float) x_length << endl;
    return 0;
}

void perform_knn(const int k,
         const data_t* x,
         const data_t* labeled,
         const int* labels,
         const int dim,
         const int x_length,
         const int labeled_length,
         int* x_pred) {
    // Loop through data points and classify each based on nearest labeled
    // neighbors.
    for (int i = 0; i < x_length; i++) {
        knn(k, x, i, labeled, labels, dim, labeled_length, x_pred);
    }
}

void knn(const int k,
         const data_t* x,
         const int i,
         const data_t* labeled,
         const int* labels,
         const int dim,
         const int labeled_length,
         int* x_pred) {
    float distances[labeled_length];
    // Compute the euclidean distances between x and the labeled data.
    for (int j = 0; j < labeled_length; j++) {
        distances[j] = euclid_distance(x + i*dim, labeled + j*dim, dim);
    }
    // Predict the class for this data point.
    x_pred[i] = predict_class(
            distances,
            labels,
            labeled_length,
            k);
}

float euclid_distance(const data_t* x, const data_t* y, const int length) {
    float dist = 0;
    for (int i = 0; i < length; i++) {
        dist += pow(fabs(x[i] - y[i]), 2);
    } 
    return sqrt(dist);
}

data_t hamming_distance(const data_t* x, const data_t* y, const int length) {
    data_t dist = 0;
    for (int i = 0; i < length; i++) {
        dist += (data_t) fabs(x[i] - y[i]);
    }
    return dist;
}

int predict_class(const float* distances,
                  const int* labels,
                  const int length,
                  const int k) {
    vector< pair<int, float> > min_dist_labels;
    // Compute the labels of the k minimum distances.
    for (int i = 0; i < length; i++) {
        if (min_dist_labels.size() < k) {
            min_dist_labels.push_back(
                    pair<int, float>(labels[i], distances[i]));
        } else if (distances[i] < min_dist_labels.back().second) {
            // Delete the last element, the largest.
            min_dist_labels.pop_back();
            // Insert new smaller distance element.
            min_dist_labels.push_back(
                    pair<int, float>(labels[i], distances[i]));
        }
        // Keep our list sorted so that we can compare new distances to
        // the last element.
        sort(min_dist_labels.begin(), min_dist_labels.end(), compare_labels);
    }
    // Compute the mode of the k labels.
    return mode_labels(min_dist_labels);
}

int mode_labels(const vector< pair<int, float> >& labels) {
    std::unordered_map<int, size_t> counts;
    for (auto i : labels) {
        ++counts[i.first];
    }

    std::multimap<size_t, int, std::greater<size_t> > inv;
    for (auto p : counts)
        inv.insert(std::make_pair(p.second, p.first));

    auto e = inv.upper_bound(inv.begin()->first);

    double sum = std::accumulate(inv.begin(),
            e,
            0.0,
            [](double a, std::pair<size_t, int> const &b)
                { return a + b.second; });

    return sum / std::distance(inv.begin(), e);
}

bool compare_labels(const pair<int, float>&i, const pair<int, float>&j) {
    return i.second < j.second;
}

const char* getfield(char* line, int num) {
    const char* tok;
    for (tok = strtok(line, ",");
            tok && *tok;
            tok = strtok(NULL, ",\n"))
    {
        if (!--num)
            return tok;
    }
    return NULL;
}

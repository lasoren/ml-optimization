#include <math.h>
#include <stdio.h>
#include <vector>
#include <algorithm>
#include <unordered_map>
#include <map>
#include <iostream>
#include <utility>
#include <functional>
#include <numeric>

typedef float data_t;

using namespace std;

// Compute the euclidean distance between two multi-dimensional vectors.
// Note: x and y must be the same length.
float euclid_distance(const data_t* x, const data_t* y, const int length);

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

int main() {
    float distances[] = {0.4, 0.2, 0.1, 0.1, 0.05};
    int labels[] = {1, 2, 3, 1, 3};
    cout << predict_class(distances, labels, 5, 4) << endl;
}

float euclid_distance(const data_t* x, const data_t* y, const int length) {
    float dist = 0;
    for (int i = 0; i < length; i++) {
        dist += pow(fabs(x[i] - y[i]), 2);
    } 
    return sqrt(dist);
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

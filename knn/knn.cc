#include "knn.h"

#include <math.h>
#include <unordered_map>
#include <map>
#include <utility>
#include <functional>
#include <numeric>
#include <iostream>

using namespace std;

void perform_knn(const int k,
         const data_t* x,
         const data_t* labeled,
         const data_t* labels,
         const int dim,
         const int x_length,
         const int labeled_length,
         data_t* x_pred) {
    // Loop through data points and classify each based on nearest labeled
    // neighbors.
    for (int i = 0; i < x_length; i++) {
        if (i%100 == 0) {
            cout << "Done classifying: " << i << endl;
        }
        knn(k, x, i, labeled, labels, dim, labeled_length, x_pred);
    }
}

void knn(const int k,
         const data_t* x,
         const int i,
         const data_t* labeled,
         const data_t* labels,
         const int dim,
         const int labeled_length,
         data_t* x_pred) {
    float distances[labeled_length];
    // Compute the hamming distances between x and the labeled data.
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
                  const data_t* labels,
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
    vector<int> counts;
    for (int i = 0; i < labels.size(); i++) {
        counts.push_back(labels[i].first);
    }

    sort(counts.begin(), counts.end());

    int number = counts[0];
    int mode = number;
    int count = 1;
    int countMode = 1;

    for (int i=1; i< counts.size(); i++)
    {
        if (counts[i] == number)
        { // count occurrences of the current number
            countMode++;
        }
        else
        { // now this is a different number
            if (count > countMode)
            {
                countMode = count; // mode is the biggest ocurrences
                mode = number;
            }
            count = 1; // reset count for the new number
            number = counts[i];
        }
    }

    return mode;
}

bool compare_labels(const pair<int, float>&i, const pair<int, float>&j) {
    return i.second < j.second;
}


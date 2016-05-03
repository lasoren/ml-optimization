// g++ $(pkg-config --cflags --libs opencv) -o knn.o mnist_main.cc knn.cc 

#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <iostream>

#include "opencv2/highgui/highgui.hpp"

#include "knn.h"

using namespace cv;
using namespace std;

// Get fields from a line and fill an array with the values.
void get_fields(char* line, data_t* data, int max_width);

Mat* construct_image(data_t* data, int height, int width);

const char kWindowName[] = "Example Data Point";

int main() {
    const int x_length = 2163;
    const int labeled_length = 13007;
    const int x_dim = 784;
    const int im_height = 28;
    const int im_width = 28;
    int i, j;

    data_t x[x_length*x_dim];
    data_t* labeled = (data_t*) malloc(labeled_length*x_dim*sizeof(data_t));
    data_t x_labels[x_length];
    data_t x_pred[x_length];
    data_t labels[labeled_length];

    int line_counter = 0;
    FILE* stream = fopen("Xtrain.csv", "r");
    char line[10240];
    while (fgets(line, 10240, stream))
    {
        char* tmp = strdup(line);
        int idx = line_counter*x_dim;
        get_fields(tmp, labeled+idx, x_dim);
        free(tmp);
        line_counter++;
        if (line_counter == labeled_length) {
            break;
        }
    }

    stream = fopen("ytrain.csv", "r");
    line_counter = 0;
    while (fgets(line, 1024, stream))
    {
        char* tmp = strdup(line);
        get_fields(tmp, labels+line_counter, 1);
        free(tmp);
        line_counter++;
        if (line_counter == labeled_length) {
            break;
        }
    }
    
    stream = fopen("Xtest.csv", "r");
    line_counter = 0;
    while (fgets(line, 1024, stream))
    {
        char* tmp = strdup(line);
        int idx = line_counter*x_dim;
        get_fields(tmp, x+idx, x_dim);
        free(tmp);
        line_counter++;
        if (line_counter == x_length) {
            break;
        }
    }

    stream = fopen("ytest.csv", "r");
    line_counter = 0;
    while (fgets(line, 1024, stream))
    {
        char* tmp = strdup(line);
        get_fields(tmp, x_labels+line_counter, 1);
        free(tmp);
        line_counter++;
        if (line_counter == x_length) {
            break;
        }
    }
    
    namedWindow(kWindowName, CV_WINDOW_AUTOSIZE);

    perform_knn(6, x, labeled, labels, x_dim, x_length, labeled_length, x_pred);
    int count_right = 0;    
    for (i = 0; i < x_length; i++) {
        if (x_pred[i] == x_labels[i]) {
            count_right++;
        }
        if (i % (x_length / 20) == 0) {  // Display sample image.
            unique_ptr<Mat> mat_ptr(
                    construct_image(x + i*x_dim, im_height, im_width));
            imshow(kWindowName, *mat_ptr);
            cout << "Prediction: " << (int) x_pred[i] << " Real label: " << (int) x_labels[i] << endl;
            waitKey(0);
        }
    } 
    cout << "Percentage correctly classified: " <<
        100*count_right / (float) x_length << endl;
    free(labeled);
    return 0;
}

void get_fields(char* line, data_t* data, int max_width) {
    int count = 0; 
    const char* tok;
    for (tok = strtok(line, ",");
            tok && *tok;
            tok = strtok(NULL, ",\n"))
    {
        data[count] = (data_t) strtod(tok, NULL);
        count++;
        if (count == max_width) {
            break;
        }
    }
}

Mat* construct_image(data_t* data, int height, int width) {
    unique_ptr<Mat> mat_ptr(new Mat(width, height, CV_8U));
    for (int i = 0; i < height; i++) {
        for (int j = 0; j < width; j++) {
            if (data[i*width + j]) {
                mat_ptr->at<unsigned char>(i,j) = 255;
            }
        }
    }
    return mat_ptr.release();
}


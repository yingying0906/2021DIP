#include <iostream>
#include <vector>
#include <math.h>
#include <opencv2/opencv.hpp>
#include <cstdlib>

using namespace cv;
using namespace std;

void shift(Mat input, Mat output, int tx, int ty)
{
    for (int i = 0; i < output.rows; i++)
    {
        for (int j = 0; j < output.cols; j++)
        {
            int x = j;
            int y = i;

            x -= tx;
            y -= ty;

            if (x >= 0 && x < input.cols && y >= 0 && y < input.rows)
            {
                output.at<uchar>(i, j) = input.at<uchar>(y, x);
            }
        }
    }
}

void scaling(Mat input, Mat output, float sx, float sy)
{
    for (int i = 0; i < output.rows; i++)
    {
        for (int j = 0; j < output.cols; j++)
        {
            float x = j;
            float y = i;

            x /= sx;
            y /= sy;

            if (x >= 0 && x < input.cols && y >= 0 && y < input.rows)
            {
                output.at<uchar>(i, j) = input.at<uchar>((int)y, (int)x);
            }
        }
    }
}

void rotating(Mat input, Mat output, float theta, int centerx, int centery)
{
    float rad = theta / (180.0 / 3.14159);
    for (int i = 0; i < output.rows; i++)
    {
        for (int j = 0; j < output.cols; j++)
        {
            int x_old = j;
            int y_old = i;

            float x = j;
            float y = i;

            float x_new = j;
            float y_new = i;

            x -= centerx;
            y -= centery;

            x_new = (x)*cos(rad) + (y)*sin(rad);
            y_new = (x)*sin(rad) * -1 + (y)*cos(rad);

            x = x_new + centerx;
            y = y_new + centery;

            if (x >= 0 && x < input.cols && y >= 0 && y < input.rows)
            {
                output.at<uchar>(i, j) = input.at<uchar>((int)y, (int)x);
            }
        }
    }
}

void s_rotate(Mat input, Mat output, float theta, int centerx, int centery, int thickx, int thicky)
{
    float rad = theta / (180.0 / 3.14159);
    for (int i = 0 + thicky; i < output.rows - thicky; i++)
    {
        for (int j = 0 + thickx; j < output.cols - thickx; j++)
        {
            int x_old = j;
            int y_old = i;

            float x = j;
            float y = i;
            float x_new = j;
            float y_new = i;

            x -= centerx;
            y -= centery;

            x_new = (x)*cos(rad) + (y)*sin(rad);
            y_new = (x)*sin(rad) * -1 + (y)*cos(rad);

            x = x_new + centerx;
            y = y_new + centery;

            if (x >= 0 && x < input.cols && y >= 0 && y < input.rows)
            {
                output.at<uchar>(i, j) = input.at<uchar>((int)y, (int)x);
            }
        }
    }
}

void s1_rotate(Mat input, Mat output, float theta, int centerx, int centery)
{
    float rad;
    for (int i = 0; i < output.rows; i++)
    {
        for (int j = 0; j < output.cols; j++)
        {
            int x_old = j;
            int y_old = i;

            float x = j;
            float y = i;
            float x_new = j;
            float y_new = i;

            x -= centerx;
            y -= centery;

            //angle
            float x_diff = abs(j - centerx);
            float y_diff = abs(i - centery);
            float max_num = sqrt(512 * 512 + 640 * 640);
            float angle = 60 - ((sqrt(y_diff * y_diff + x_diff * x_diff) / max_num * 60));

            rad = ((int)angle * -1 % 360) / (180.0 / 3.14159);

            //angle

            x_new = (x)*cos(rad) + (y)*sin(rad);
            y_new = (x)*sin(rad) * -1 + (y)*cos(rad);

            x = x_new + centerx;
            y = y_new + centery;

            if (x >= 0 && x < input.cols && y >= 0 && y < input.rows)
            {
                output.at<uchar>(i, j) = input.at<uchar>((int)y, (int)x);
            }
        }
    }
}

int main()
{
    // (-) input image
    Mat sample1 = imread("sample1.jpg", CV_8UC1);
    Mat sample2 = imread("sample2.jpg", CV_8UC1);
    Mat sample3 = imread("sample3.jpg", CV_8UC1);
    Mat sample5 = imread("sample5.jpg", CV_8UC1);

    // (a)
    Mat result6_input(sample3.rows, sample3.cols, CV_8UC1, Scalar(0));
    Mat result6_output(sample3.rows, sample3.cols, CV_8UC1, Scalar(0));
    Mat black(sample3.rows, sample3.cols, CV_8UC1, Scalar(0));

    //shifer to the center
    result6_input = sample3.clone();
    shift(result6_input, result6_output, -200, 0);
    imwrite("case/result6_1.jpg", result6_output);

    //rotate by center
    result6_input = result6_output.clone();
    result6_output = black.clone();
    rotating(result6_input, result6_output, -83, result6_input.cols / 2, result6_input.rows / 2);
    imwrite("case/result6_2.jpg", result6_output);

    //shift
    result6_input = result6_output.clone();
    result6_output = black.clone();
    shift(result6_input, result6_output, -350, -250);
    imwrite("case/result6_3.jpg", result6_output);

    //scaling
    result6_input = result6_output.clone();
    result6_output = black.clone();
    scaling(result6_input, result6_output, 1.5, 1.5);
    imwrite("result6.jpg", result6_output);

    //(b)
    int thickx = 0;
    int thicky = 0;
    int add = 27;
    float angle = 1;
    Mat result7_input(sample5.rows, sample5.cols, CV_8UC1, Scalar(0));
    Mat result7_output(sample5.rows, sample5.cols, CV_8UC1, Scalar(0));
    result7_input = sample5.clone();

    s_rotate(result7_input, result7_output, 1, result6_input.cols / 2, result6_input.rows / 2, thickx, thicky);

    while (1)
    {
        thickx += add * 1.25;
        thicky += add;
        add -= 0.5;
        angle -= 0.4;
        if (add <= 0)
        {
            add = 1;
        }

        result7_input = result7_output.clone();
        s_rotate(result7_input, result7_output, -1, result6_input.cols / 2, result6_input.rows / 2, thickx, thicky);

        if (thickx > 640 && thicky > 512)
        {
            break;
        }
    }

    imwrite("result7.jpg", result7_output);

    // method2
    Mat result7_input_1(sample5.rows, sample5.cols, CV_8UC1, Scalar(0));
    Mat result7_output_1(sample5.rows, sample5.cols, CV_8UC1, Scalar(0));
    result7_input_1 = sample5.clone();

    s1_rotate(result7_input_1, result7_output_1, -1, result6_input.cols / 2, result6_input.rows / 2);

    imwrite("result7_1.jpg", result7_output_1);

    return 0;
}
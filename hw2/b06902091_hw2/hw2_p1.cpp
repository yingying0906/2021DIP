#include <iostream>
#include <vector>
#include <math.h>
#include <opencv2/opencv.hpp>
#include <cstdlib>

using namespace cv;
using namespace std;

int direction[1800][1800];

// 1st order
void roberts(Mat input, Mat output, int threshold)
{
    for (int i = 0; i < input.rows - 1; i++)
    {
        for (int j = 0; j < input.cols - 1; j++)
        {
            int g1 = input.at<uchar>(i, j) - input.at<uchar>(i + 1, j + 1);
            int g2 = input.at<uchar>(i, j + 1) - input.at<uchar>(i + 1, j);
            int gradient = sqrt(g1 * g1 + g2 * g2);
            if (gradient >= threshold)
            {
                output.at<uchar>(i, j) = 255;
            }
            else
            {
                output.at<uchar>(i, j) = 0;
            }
        }
    }
}

void prewitt(Mat input, Mat output, int threshold)
{
    for (int i = 0; i < input.rows - 1; i++)
    {
        for (int j = 0; j < input.cols - 1; j++)
        {
            int array[3][3];
            if (i == 0 && j == 0)
            {
                array[0][0] = input.at<uchar>(i, j);
                array[0][1] = input.at<uchar>(i, j);
                array[1][0] = input.at<uchar>(i, j);
                array[1][1] = input.at<uchar>(i, j);

                array[1][2] = input.at<uchar>(i, j + 1);
                array[2][1] = input.at<uchar>(i + 1, j);
                array[2][2] = input.at<uchar>(i + 1, j + 1);

                array[0][2] = input.at<uchar>(i, j + 1);
                array[2][0] = input.at<uchar>(i + 1, j);
            }
            else if (i == 0 && j != 0)
            {
                array[1][0] = input.at<uchar>(i, j - 1);
                array[1][1] = input.at<uchar>(i, j);
                array[1][2] = input.at<uchar>(i, j + 1);

                array[2][0] = input.at<uchar>(i + 1, j - 1);
                array[2][1] = input.at<uchar>(i + 1, j);
                array[2][2] = input.at<uchar>(i + 1, j + 1);

                array[0][0] = input.at<uchar>(i, j - 1);
                array[0][1] = input.at<uchar>(i, j);
                array[0][2] = input.at<uchar>(i, j + 1);
            }
            else if (j == 0 && i != 0)
            {
                array[0][1] = input.at<uchar>(i - 1, j);
                array[1][1] = input.at<uchar>(i, j);
                array[2][1] = input.at<uchar>(i + 1, j);

                array[0][2] = input.at<uchar>(i - 1, j + 1);
                array[1][2] = input.at<uchar>(i, j + 1);
                array[2][2] = input.at<uchar>(i + 1, j + 1);

                array[0][0] = input.at<uchar>(i - 1, j);
                array[1][0] = input.at<uchar>(i, j);
                array[2][0] = input.at<uchar>(i + 1, j);
            }
            else
            {
                array[0][0] = input.at<uchar>(i - 1, j - 1);
                array[0][1] = input.at<uchar>(i - 1, j);
                array[0][2] = input.at<uchar>(i - 1, j + 1);
                array[1][0] = input.at<uchar>(i, j - 1);
                array[1][1] = input.at<uchar>(i, j);
                array[1][2] = input.at<uchar>(i, j + 1);
                array[2][0] = input.at<uchar>(i + 1, j - 1);
                array[2][1] = input.at<uchar>(i + 1, j);
                array[2][2] = input.at<uchar>(i + 1, j + 1);
            }

            int gr = array[0][2] + array[1][2] + array[2][2] - array[0][0] - array[1][0] - array[2][0];
            int gc = array[0][0] + array[0][1] + array[0][2] - array[2][0] - array[2][1] - array[2][2];

            int gradient = sqrt(gr * gr + gc * gc);
            if (gradient >= threshold)
            {
                output.at<uchar>(i, j) = 255;
            }
            else
            {
                output.at<uchar>(i, j) = 0;
            }
        }
    }
}

void sobel(Mat input, Mat output, int threshold)
{
    for (int i = 0; i < input.rows - 1; i++)
    {
        for (int j = 0; j < input.cols - 1; j++)
        {
            int array[3][3];
            if (i == 0 && j == 0)
            {
                array[0][0] = input.at<uchar>(i, j);
                array[0][1] = input.at<uchar>(i, j);
                array[1][0] = input.at<uchar>(i, j);
                array[1][1] = input.at<uchar>(i, j);

                array[1][2] = input.at<uchar>(i, j + 1);
                array[2][1] = input.at<uchar>(i + 1, j);
                array[2][2] = input.at<uchar>(i + 1, j + 1);

                array[0][2] = input.at<uchar>(i, j + 1);
                array[2][0] = input.at<uchar>(i + 1, j);
            }
            else if (i == 0 && j != 0)
            {
                array[1][0] = input.at<uchar>(i, j - 1);
                array[1][1] = input.at<uchar>(i, j);
                array[1][2] = input.at<uchar>(i, j + 1);

                array[2][0] = input.at<uchar>(i + 1, j - 1);
                array[2][1] = input.at<uchar>(i + 1, j);
                array[2][2] = input.at<uchar>(i + 1, j + 1);

                array[0][0] = input.at<uchar>(i, j - 1);
                array[0][1] = input.at<uchar>(i, j);
                array[0][2] = input.at<uchar>(i, j + 1);
            }
            else if (j == 0 && i != 0)
            {
                array[0][1] = input.at<uchar>(i - 1, j);
                array[1][1] = input.at<uchar>(i, j);
                array[2][1] = input.at<uchar>(i + 1, j);

                array[0][2] = input.at<uchar>(i - 1, j + 1);
                array[1][2] = input.at<uchar>(i, j + 1);
                array[2][2] = input.at<uchar>(i + 1, j + 1);

                array[0][0] = input.at<uchar>(i - 1, j);
                array[1][0] = input.at<uchar>(i, j);
                array[2][0] = input.at<uchar>(i + 1, j);
            }
            else
            {
                array[0][0] = input.at<uchar>(i - 1, j - 1);
                array[0][1] = input.at<uchar>(i - 1, j);
                array[0][2] = input.at<uchar>(i - 1, j + 1);
                array[1][0] = input.at<uchar>(i, j - 1);
                array[1][1] = input.at<uchar>(i, j);
                array[1][2] = input.at<uchar>(i, j + 1);
                array[2][0] = input.at<uchar>(i + 1, j - 1);
                array[2][1] = input.at<uchar>(i + 1, j);
                array[2][2] = input.at<uchar>(i + 1, j + 1);
            }
            int gr = array[0][2] + array[1][2] * 2 + array[2][2] - array[0][0] - array[1][0] * 2 - array[2][0];
            int gc = array[0][0] + array[0][1] * 2 + array[0][2] - array[2][0] - array[2][1] * 2 - array[2][2];
            int gradient = sqrt(gr * gr + gc * gc);
            if (gradient >= threshold)
            {
                output.at<uchar>(i, j) = 255;
            }
            else
            {
                output.at<uchar>(i, j) = 0;
            }
        }
    }
}

// 2nd order
void low_pass_filter(Mat input, Mat output)
{
    int kernel[3][3] = {1, 2, 1, 2, 4, 2, 1, 2, 1};
    for (int i = 0; i < input.rows; i++)
    {
        for (int j = 0; j < input.cols; j++)
        {
            int sum = 0;
            for (int a = -3 / 2; a <= 3 / 2; a++)
            {
                for (int b = -3 / 2; b <= 3 / 2; b++)
                {
                    int x = i + a;
                    int y = j + b;
                    if (x < 0)
                    {
                        x = 0;
                    }
                    if (x >= input.rows)
                    {
                        x = input.rows - 1;
                    }
                    if (y < 0)
                    {
                        y = 0;
                    }
                    if (y >= input.cols)
                    {
                        y = input.cols - 1;
                    }

                    sum += output.at<uchar>(x, y) * kernel[a + 1][b + 1];
                }
            }
            output.at<uchar>(i, j) = sum / 16;
        }
    }
}

void low_pass_filter_size(Mat input, Mat output, int size)
{
    for (int i = 0; i < input.rows; i++)
    {
        for (int j = 0; j < input.cols; j++)
        {
            int sum = 0;
            int count = 0;
            for (int a = -size / 2; a <= size / 2; a++)
            {
                for (int b = -size / 2; b <= size / 2; b++)
                {
                    int x = i + a;
                    int y = j + b;
                    if (x < 0)
                    {
                        x = 0;
                    }
                    if (x >= input.rows)
                    {
                        x = input.rows - 1;
                    }
                    if (y < 0)
                    {
                        y = 0;
                    }
                    if (y >= input.cols)
                    {
                        y = input.cols - 1;
                    }

                    sum += output.at<uchar>(x, y);
                    count++;
                }
            }
            output.at<uchar>(i, j) = sum / count;
        }
    }
}

void laplacian(Mat input, Mat output, int threshold, int constant, int kernel[3][3])
{
    //laplacian output
    Mat tmp(input.rows, input.cols, CV_8UC1, Scalar(255));
    for (int i = 0; i < input.rows - 1; i++)
    {
        for (int j = 0; j < input.cols - 1; j++)
        {
            int array[3][3];
            if (i == 0 && j == 0)
            {
                array[0][0] = input.at<uchar>(i, j);
                array[0][1] = input.at<uchar>(i, j);
                array[1][0] = input.at<uchar>(i, j);
                array[1][1] = input.at<uchar>(i, j);

                array[1][2] = input.at<uchar>(i, j + 1);
                array[2][1] = input.at<uchar>(i + 1, j);
                array[2][2] = input.at<uchar>(i + 1, j + 1);

                array[0][2] = input.at<uchar>(i, j + 1);
                array[2][0] = input.at<uchar>(i + 1, j);
            }
            else if (i == 0 && j != 0)
            {
                array[1][0] = input.at<uchar>(i, j - 1);
                array[1][1] = input.at<uchar>(i, j);
                array[1][2] = input.at<uchar>(i, j + 1);

                array[2][0] = input.at<uchar>(i + 1, j - 1);
                array[2][1] = input.at<uchar>(i + 1, j);
                array[2][2] = input.at<uchar>(i + 1, j + 1);

                array[0][0] = input.at<uchar>(i, j - 1);
                array[0][1] = input.at<uchar>(i, j);
                array[0][2] = input.at<uchar>(i, j + 1);
            }
            else if (j == 0 && i != 0)
            {
                array[0][1] = input.at<uchar>(i - 1, j);
                array[1][1] = input.at<uchar>(i, j);
                array[2][1] = input.at<uchar>(i + 1, j);

                array[0][2] = input.at<uchar>(i - 1, j + 1);
                array[1][2] = input.at<uchar>(i, j + 1);
                array[2][2] = input.at<uchar>(i + 1, j + 1);

                array[0][0] = input.at<uchar>(i - 1, j);
                array[1][0] = input.at<uchar>(i, j);
                array[2][0] = input.at<uchar>(i + 1, j);
            }
            else
            {
                array[0][0] = input.at<uchar>(i - 1, j - 1);
                array[0][1] = input.at<uchar>(i - 1, j);
                array[0][2] = input.at<uchar>(i - 1, j + 1);
                array[1][0] = input.at<uchar>(i, j - 1);
                array[1][1] = input.at<uchar>(i, j);
                array[1][2] = input.at<uchar>(i, j + 1);
                array[2][0] = input.at<uchar>(i + 1, j - 1);
                array[2][1] = input.at<uchar>(i + 1, j);
                array[2][2] = input.at<uchar>(i + 1, j + 1);
            }

            int result = 0;
            for (int a = 0; a < 3; a++)
            {
                for (int b = 0; b < 3; b++)
                {
                    result += (float)array[a][b] * (float)kernel[a][b];
                }
            }
            result = result / constant;

            if (result >= threshold)
            {
                tmp.at<uchar>(i, j) = 1;
            }
            else if (result <= (threshold * -1))
            {
                tmp.at<uchar>(i, j) = 100; //-1 -> 100
            }
            else
            {
                tmp.at<uchar>(i, j) = 0;
            }
        }
    }

    //zero crossing
    for (int i = 0; i < input.rows - 1; i++)
    {
        for (int j = 0; j < input.cols - 1; j++)
        {
            if (tmp.at<uchar>(i, j) == 100 || tmp.at<uchar>(i, j) == 0)
            {
                output.at<uchar>(i, j) = 0;
                continue;
            }
            int array[3][3];
            if (i == 0 && j == 0)
            {
                array[0][0] = tmp.at<uchar>(i, j);
                array[0][1] = tmp.at<uchar>(i, j);
                array[1][0] = tmp.at<uchar>(i, j);
                array[1][1] = tmp.at<uchar>(i, j);

                array[1][2] = tmp.at<uchar>(i, j + 1);
                array[2][1] = tmp.at<uchar>(i + 1, j);
                array[2][2] = tmp.at<uchar>(i + 1, j + 1);

                array[0][2] = tmp.at<uchar>(i, j + 1);
                array[2][0] = tmp.at<uchar>(i + 1, j);
            }
            else if (i == 0 && j != 0)
            {
                array[1][0] = tmp.at<uchar>(i, j - 1);
                array[1][1] = tmp.at<uchar>(i, j);
                array[1][2] = tmp.at<uchar>(i, j + 1);

                array[2][0] = tmp.at<uchar>(i + 1, j - 1);
                array[2][1] = tmp.at<uchar>(i + 1, j);
                array[2][2] = tmp.at<uchar>(i + 1, j + 1);

                array[0][0] = tmp.at<uchar>(i, j - 1);
                array[0][1] = tmp.at<uchar>(i, j);
                array[0][2] = tmp.at<uchar>(i, j + 1);
            }
            else if (j == 0 && i != 0)
            {
                array[0][1] = tmp.at<uchar>(i - 1, j);
                array[1][1] = tmp.at<uchar>(i, j);
                array[2][1] = tmp.at<uchar>(i + 1, j);

                array[0][2] = tmp.at<uchar>(i - 1, j + 1);
                array[1][2] = tmp.at<uchar>(i, j + 1);
                array[2][2] = tmp.at<uchar>(i + 1, j + 1);

                array[0][0] = tmp.at<uchar>(i - 1, j);
                array[1][0] = tmp.at<uchar>(i, j);
                array[2][0] = tmp.at<uchar>(i + 1, j);
            }
            else
            {
                array[0][0] = tmp.at<uchar>(i - 1, j - 1);
                array[0][1] = tmp.at<uchar>(i - 1, j);
                array[0][2] = tmp.at<uchar>(i - 1, j + 1);
                array[1][0] = tmp.at<uchar>(i, j - 1);
                array[1][1] = tmp.at<uchar>(i, j);
                array[1][2] = tmp.at<uchar>(i, j + 1);
                array[2][0] = tmp.at<uchar>(i + 1, j - 1);
                array[2][1] = tmp.at<uchar>(i + 1, j);
                array[2][2] = tmp.at<uchar>(i + 1, j + 1);
            }
            int flag = 0;
            for (int a = 0; a < 3; a++)
            {
                for (int b = 0; b < 3; b++)
                {
                    if (a == 1 && b == 1)
                    {
                        continue;
                    }
                    if (array[a][b] == 100)
                    {
                        flag = 1;
                    }
                }
            }
            if (flag == 1)
            {
                output.at<uchar>(i, j) = 255;
            }
            else
            {
                output.at<uchar>(i, j) = 0;
            }
        }
    }
}

// canny
void sobel_canny(Mat input, Mat output)
{
    for (int i = 0; i < input.rows - 1; i++)
    {
        for (int j = 0; j < input.cols - 1; j++)
        {
            int array[3][3];
            if (i == 0 && j == 0)
            {
                array[0][0] = input.at<uchar>(i, j);
                array[0][1] = input.at<uchar>(i, j);
                array[1][0] = input.at<uchar>(i, j);
                array[1][1] = input.at<uchar>(i, j);

                array[1][2] = input.at<uchar>(i, j + 1);
                array[2][1] = input.at<uchar>(i + 1, j);
                array[2][2] = input.at<uchar>(i + 1, j + 1);

                array[0][2] = input.at<uchar>(i, j + 1);
                array[2][0] = input.at<uchar>(i + 1, j);
            }
            else if (i == 0 && j != 0)
            {
                array[1][0] = input.at<uchar>(i, j - 1);
                array[1][1] = input.at<uchar>(i, j);
                array[1][2] = input.at<uchar>(i, j + 1);

                array[2][0] = input.at<uchar>(i + 1, j - 1);
                array[2][1] = input.at<uchar>(i + 1, j);
                array[2][2] = input.at<uchar>(i + 1, j + 1);

                array[0][0] = input.at<uchar>(i, j - 1);
                array[0][1] = input.at<uchar>(i, j);
                array[0][2] = input.at<uchar>(i, j + 1);
            }
            else if (j == 0 && i != 0)
            {
                array[0][1] = input.at<uchar>(i - 1, j);
                array[1][1] = input.at<uchar>(i, j);
                array[2][1] = input.at<uchar>(i + 1, j);

                array[0][2] = input.at<uchar>(i - 1, j + 1);
                array[1][2] = input.at<uchar>(i, j + 1);
                array[2][2] = input.at<uchar>(i + 1, j + 1);

                array[0][0] = input.at<uchar>(i - 1, j);
                array[1][0] = input.at<uchar>(i, j);
                array[2][0] = input.at<uchar>(i + 1, j);
            }
            else
            {
                array[0][0] = input.at<uchar>(i - 1, j - 1);
                array[0][1] = input.at<uchar>(i - 1, j);
                array[0][2] = input.at<uchar>(i - 1, j + 1);
                array[1][0] = input.at<uchar>(i, j - 1);
                array[1][1] = input.at<uchar>(i, j);
                array[1][2] = input.at<uchar>(i, j + 1);
                array[2][0] = input.at<uchar>(i + 1, j - 1);
                array[2][1] = input.at<uchar>(i + 1, j);
                array[2][2] = input.at<uchar>(i + 1, j + 1);
            }
            int gr = array[0][2] + array[1][2] * 2 + array[2][2] - array[0][0] - array[1][0] * 2 - array[2][0];
            int gc = array[0][0] + array[0][1] * 2 + array[0][2] - array[2][0] - array[2][1] * 2 - array[2][2];
            int gradient = sqrt(gr * gr + gc * gc);
            output.at<uchar>(i, j) = gradient;
            float dir;
            if (gr != 0)
            {
                dir = atan(gc / gr);
                dir = dir * 180 / 3.14159;
                if (dir < 0)
                {
                    dir += 180;
                }
                if (dir >= 0 && dir < 22.5)
                {
                    direction[i][j] = 2; //horizontal
                }
                else if (dir >= 157.5 && dir <= 180)
                {
                    direction[i][j] = 2; //horizontal
                }
                else if (dir >= 22.5 && dir < 67.5)
                {
                    direction[i][j] = 3; //top right
                }
                else if (dir >= 67.5 && dir < 112.5)
                {
                    direction[i][j] = 1; //vertical
                }
                else if (dir >= 112.5 && dir < 157.5)
                {
                    direction[i][j] = 4; //topleft
                }
            }
            else
            {
                direction[i][j] = 1; //vertical
            }
        }
    }
}

void non_max_canny(Mat input, Mat output)
{
    for (int i = 0; i < input.rows - 1; i++)
    {
        for (int j = 0; j < input.cols - 1; j++)
        {
            if (input.at<uchar>(i, j) == 0)
            {
                continue;
            }
            int array[3][3];
            if (i == 0 && j == 0)
            {
                array[0][0] = input.at<uchar>(i, j);
                array[0][1] = input.at<uchar>(i, j);
                array[1][0] = input.at<uchar>(i, j);
                array[1][1] = input.at<uchar>(i, j);

                array[1][2] = input.at<uchar>(i, j + 1);
                array[2][1] = input.at<uchar>(i + 1, j);
                array[2][2] = input.at<uchar>(i + 1, j + 1);

                array[0][2] = input.at<uchar>(i, j + 1);
                array[2][0] = input.at<uchar>(i + 1, j);
            }
            else if (i == 0 && j != 0)
            {
                array[1][0] = input.at<uchar>(i, j - 1);
                array[1][1] = input.at<uchar>(i, j);
                array[1][2] = input.at<uchar>(i, j + 1);

                array[2][0] = input.at<uchar>(i + 1, j - 1);
                array[2][1] = input.at<uchar>(i + 1, j);
                array[2][2] = input.at<uchar>(i + 1, j + 1);

                array[0][0] = input.at<uchar>(i, j - 1);
                array[0][1] = input.at<uchar>(i, j);
                array[0][2] = input.at<uchar>(i, j + 1);
            }
            else if (j == 0 && i != 0)
            {
                array[0][1] = input.at<uchar>(i - 1, j);
                array[1][1] = input.at<uchar>(i, j);
                array[2][1] = input.at<uchar>(i + 1, j);

                array[0][2] = input.at<uchar>(i - 1, j + 1);
                array[1][2] = input.at<uchar>(i, j + 1);
                array[2][2] = input.at<uchar>(i + 1, j + 1);

                array[0][0] = input.at<uchar>(i - 1, j);
                array[1][0] = input.at<uchar>(i, j);
                array[2][0] = input.at<uchar>(i + 1, j);
            }
            else
            {
                array[0][0] = input.at<uchar>(i - 1, j - 1);
                array[0][1] = input.at<uchar>(i - 1, j);
                array[0][2] = input.at<uchar>(i - 1, j + 1);
                array[1][0] = input.at<uchar>(i, j - 1);
                array[1][1] = input.at<uchar>(i, j);
                array[1][2] = input.at<uchar>(i, j + 1);
                array[2][0] = input.at<uchar>(i + 1, j - 1);
                array[2][1] = input.at<uchar>(i + 1, j);
                array[2][2] = input.at<uchar>(i + 1, j + 1);
            }

            int max = input.at<uchar>(i, j);
            if (direction[i][j] == 1)
            { //vertical
                if (max > array[0][1] && max > array[2][1])
                {
                    output.at<uchar>(i, j) = input.at<uchar>(i, j);
                }
                else
                {
                    output.at<uchar>(i, j) = 0;
                }
            }
            else if (direction[i][j] == 2)
            { //horiz
                if (max > array[1][2] && max > array[1][0])
                {
                    output.at<uchar>(i, j) = input.at<uchar>(i, j);
                }
                else
                {
                    output.at<uchar>(i, j) = 0;
                }
            }
            else if (direction[i][j] == 3)
            { //upper right
                if (max > array[0][0] && max > array[2][2])
                {
                    output.at<uchar>(i, j) = input.at<uchar>(i, j);
                }
                else
                {
                    output.at<uchar>(i, j) = 0;
                }
            }
            else
            {
                if (max > array[0][2] && max > array[2][0])
                {
                    output.at<uchar>(i, j) = input.at<uchar>(i, j);
                }
                else
                {
                    output.at<uchar>(i, j) = 0;
                }
            }
            input.at<uchar>(i, j) = output.at<uchar>(i, j);
        }
    }
}

void thresholding_canny(Mat input, Mat output, int high_t, int low_t)
{
    for (int i = 0; i < input.rows - 1; i++)
    {
        for (int j = 0; j < input.cols - 1; j++)
        {
            if (input.at<uchar>(i, j) >= high_t)
            {
                output.at<uchar>(i, j) = 255;
            }
            else if (input.at<uchar>(i, j) >= low_t && input.at<uchar>(i, j) < high_t)
            {
                output.at<uchar>(i, j) = 70;
            }
            else
            {
                output.at<uchar>(i, j) = 0;
            }
        }
    }
}

void connect(Mat input, Mat output)
{
    for (int i = 0; i < input.rows - 1; i++)
    {
        for (int j = 0; j < input.cols - 1; j++)
        {
            if (input.at<uchar>(i, j) == 255)
            {
                output.at<uchar>(i, j) = 255;
                continue;
            }
            if (input.at<uchar>(i, j) == 0)
            {
                output.at<uchar>(i, j) = 0;
                continue;
            }
            int array[3][3];
            if (i == 0 && j == 0)
            {
                array[0][0] = input.at<uchar>(i, j);
                array[0][1] = input.at<uchar>(i, j);
                array[1][0] = input.at<uchar>(i, j);
                array[1][1] = input.at<uchar>(i, j);

                array[1][2] = input.at<uchar>(i, j + 1);
                array[2][1] = input.at<uchar>(i + 1, j);
                array[2][2] = input.at<uchar>(i + 1, j + 1);

                array[0][2] = input.at<uchar>(i, j + 1);
                array[2][0] = input.at<uchar>(i + 1, j);
            }
            else if (i == 0 && j != 0)
            {
                array[1][0] = input.at<uchar>(i, j - 1);
                array[1][1] = input.at<uchar>(i, j);
                array[1][2] = input.at<uchar>(i, j + 1);

                array[2][0] = input.at<uchar>(i + 1, j - 1);
                array[2][1] = input.at<uchar>(i + 1, j);
                array[2][2] = input.at<uchar>(i + 1, j + 1);

                array[0][0] = input.at<uchar>(i, j - 1);
                array[0][1] = input.at<uchar>(i, j);
                array[0][2] = input.at<uchar>(i, j + 1);
            }
            else if (j == 0 && i != 0)
            {
                array[0][1] = input.at<uchar>(i - 1, j);
                array[1][1] = input.at<uchar>(i, j);
                array[2][1] = input.at<uchar>(i + 1, j);

                array[0][2] = input.at<uchar>(i - 1, j + 1);
                array[1][2] = input.at<uchar>(i, j + 1);
                array[2][2] = input.at<uchar>(i + 1, j + 1);

                array[0][0] = input.at<uchar>(i - 1, j);
                array[1][0] = input.at<uchar>(i, j);
                array[2][0] = input.at<uchar>(i + 1, j);
            }
            else
            {
                array[0][0] = input.at<uchar>(i - 1, j - 1);
                array[0][1] = input.at<uchar>(i - 1, j);
                array[0][2] = input.at<uchar>(i - 1, j + 1);
                array[1][0] = input.at<uchar>(i, j - 1);
                array[1][1] = input.at<uchar>(i, j);
                array[1][2] = input.at<uchar>(i, j + 1);
                array[2][0] = input.at<uchar>(i + 1, j - 1);
                array[2][1] = input.at<uchar>(i + 1, j);
                array[2][2] = input.at<uchar>(i + 1, j + 1);
            }
            output.at<uchar>(i, j) = 0;
            for (int a = 0; a < 3; a++)
            {
                for (int b = 0; b < 3; b++)
                {
                    if (array[a][b] == 255)
                    {
                        output.at<uchar>(i, j) = 255;
                        break;
                    }
                }
            }
            input.at<uchar>(i, j) = output.at<uchar>(i, j);
        }
    }
}

void canny(Mat input, Mat output, int high, int low, int pass_filter_size)
{
    Mat canny_noise = input.clone();
    Mat canny_sobel = input.clone();
    Mat canny_nonmax(input.rows, input.cols, CV_8UC1, Scalar(0));
    Mat canny_threshold(input.rows, input.cols, CV_8UC1, Scalar(0));

    low_pass_filter_size(input, canny_noise, pass_filter_size);
    sobel_canny(canny_noise, canny_sobel);

    non_max_canny(canny_sobel, canny_nonmax);
    thresholding_canny(canny_nonmax, canny_threshold, high, low);

    connect(canny_threshold, output);
}

//crispening
void crispening(Mat input, Mat output, int ker)
{
    int kernel[2][3][3] = {{0, -1, 0, -1, 5, -1, 0, -1, 0},
                           {1, -1, -1, -1, 9, -1, -1, -1, -1}};
    for (int i = 0; i < input.rows; i++)
    {
        for (int j = 0; j < input.cols; j++)
        {
            int sum = 0;
            for (int a = -3 / 2; a <= 3 / 2; a++)
            {
                for (int b = -3 / 2; b <= 3 / 2; b++)
                {
                    int x = i + a;
                    int y = j + b;
                    if (x < 0)
                    {
                        x = 0;
                    }
                    if (x >= input.rows)
                    {
                        x = input.rows - 1;
                    }
                    if (y < 0)
                    {
                        y = 0;
                    }
                    if (y >= input.cols)
                    {
                        y = input.cols - 1;
                    }
                    sum += input.at<uchar>(x, y) * kernel[ker][a + 1][b + 1];
                }
            }
            output.at<uchar>(i, j) = sum;
        }
    }
}

void unsharp(Mat lowpass, Mat allpass, Mat output, int c)
{
    for (int i = 0; i < lowpass.rows; i++)
    {
        for (int j = 0; j < lowpass.cols; j++)
        {
            output.at<uchar>(i, j) = (c / (2 * c - 1)) * output.at<uchar>(i, j) - ((1 - c) / (2 * c - 1)) * output.at<uchar>(i, j);
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
    // (1) 1st order
    Mat result1 = sample1.clone();
    Mat sample1_roberts = sample1.clone();
    Mat sample1_prewitt = sample1.clone();
    Mat sample1_sobel = sample1.clone();

    roberts(sample1, sample1_roberts, 40);
    imwrite("case/result1_roberts40.jpg", sample1_roberts);
    roberts(sample1, sample1_roberts, 70);
    imwrite("case/result1_roberts70.jpg", sample1_roberts);
    prewitt(sample1, sample1_prewitt, 40);
    imwrite("case/result1_prewitt40.jpg", sample1_prewitt);
    prewitt(sample1, sample1_prewitt, 70);
    imwrite("case/result1_prewitt70.jpg", sample1_prewitt);
    sobel(sample1, sample1_sobel, 40);
    imwrite("case/result1_sobel40.jpg", sample1_sobel);
    sobel(sample1, sample1_sobel, 70);
    imwrite("case/result1_sobel70.jpg", sample1_sobel);

    roberts(sample1, result1, 40);
    imwrite("result1.jpg", result1);

    // (2) 2nd order
    Mat sample1_tmp = sample1.clone();
    Mat sample1_2nd_ker1_t2 = sample1.clone();
    Mat sample1_2nd_ker2_t2 = sample1.clone();
    Mat sample1_2nd_ker1_t4 = sample1.clone();
    Mat sample1_2nd_ker2_t4 = sample1.clone();
    int kernel1[3][3] = {0, 1, 0, 1, -4, 1, 0, 1, 0};
    int kernel2[3][3] = {1, 1, 1, 1, -8, 1, 1, 1, 1};

    low_pass_filter(sample1, sample1_tmp);
    laplacian(sample1_tmp, sample1_2nd_ker1_t2, 2, 4, kernel1);
    laplacian(sample1_tmp, sample1_2nd_ker2_t2, 2, 8, kernel2);
    laplacian(sample1_tmp, sample1_2nd_ker1_t4, 4, 4, kernel1);
    laplacian(sample1_tmp, sample1_2nd_ker2_t4, 4, 8, kernel2);
    imwrite("case/result2_ker1_t2.jpg", sample1_2nd_ker1_t2);
    imwrite("case/result2_ker2_t2.jpg", sample1_2nd_ker2_t2);
    imwrite("case/result2_ker1_t4.jpg", sample1_2nd_ker1_t4);
    imwrite("case/result2_ker2_t4.jpg", sample1_2nd_ker2_t4);
    imwrite("result2.jpg", sample1_2nd_ker2_t2);

    // (3) canny
    Mat sample1_canny_noise = sample1.clone();
    Mat sample1_canny_sobel = sample1.clone();
    Mat sample1_canny_nonmax(sample1.rows, sample1.cols, CV_8UC1, Scalar(0));
    Mat sample1_canny_threshold(sample1.rows, sample1.cols, CV_8UC1, Scalar(0));
    Mat sample1_canny_result(sample1.rows, sample1.cols, CV_8UC1, Scalar(0));

    low_pass_filter_size(sample1, sample1_canny_noise, 7);
    imwrite("case/result3_1.jpg", sample1_canny_noise);
    sobel_canny(sample1_canny_noise, sample1_canny_sobel);
    imwrite("case/result3_2.jpg", sample1_canny_sobel);
    non_max_canny(sample1_canny_sobel, sample1_canny_nonmax);
    imwrite("case/result3_3.jpg", sample1_canny_nonmax);
    thresholding_canny(sample1_canny_nonmax, sample1_canny_threshold, 110, 55);
    imwrite("case/result3_4.jpg", sample1_canny_threshold);
    connect(sample1_canny_threshold, sample1_canny_result);
    imwrite("case/result3_5.jpg", sample1_canny_result);
    
    // other case
    Mat sample1_canny_hight = sample1.clone();
    Mat sample1_canny_lowt = sample1.clone();
    Mat sample1_canny_highfilter = sample1.clone();
    Mat sample1_canny_lowfilter = sample1.clone();
    canny(sample1, sample1_canny_hight, 70, 55, 7); //high t change
    canny(sample1, sample1_canny_lowt, 110, 22, 7); //lowt change -> more edge
    canny(sample1, sample1_canny_highfilter, 110, 55, 10);  // high filter size
    canny(sample1, sample1_canny_lowfilter, 110, 55, 3);  // low filter size

    imwrite("case/result3_higher_t.jpg", sample1_canny_hight);
    imwrite("case/result3_lower_t.jpg", sample1_canny_lowt);
    imwrite("case/result3_higher_filter.jpg", sample1_canny_highfilter);
    imwrite("case/result3_lower_filter.jpg", sample1_canny_lowfilter);

    imwrite("result3.jpg", sample1_canny_lowt);

    // (4) edge crispening
    Mat sample1_crispening1 = sample1.clone();
    Mat sample1_crispening2 = sample1.clone();
    crispening(sample1, sample1_crispening1, 0);
    crispening(sample1, sample1_crispening2, 1);

    Mat result4 = sample1.clone();
    Mat sample1_lowpass = sample1.clone();
    low_pass_filter_size(sample1, sample1_lowpass, 7);
    unsharp(sample1_lowpass, sample1_crispening2, result4, 5 / 6);

    imwrite("case/result4_1.jpg", sample1_crispening1);
    imwrite("case/result4_2.jpg", sample1_crispening2);
    result4 = sample1_crispening1.clone();
    imwrite("result4.jpg", result4);

    Mat result5 = sample1.clone();
    roberts(result4, result5, 70);
    imwrite("result5.jpg", result5);

    // (5) sample2.jpg
    // cripsening
    Mat sample2_crispening2 = sample2.clone();
    crispening(sample2, sample2_crispening2, 0);

    Mat sample2_lowpass = sample2.clone();
    low_pass_filter_size(sample2, sample2_lowpass, 3);

    Mat sample2_sharp = sample2.clone();
    unsharp(sample2_lowpass, sample2_crispening2, sample2_sharp, 5 / 6);
    sample2 = sample2_sharp.clone();

    // 1st 比較好調
    Mat sample2_roberts = sample2.clone();
    Mat sample2_prewitt = sample2.clone();
    Mat sample2_sobel = sample2.clone();

    roberts(sample2, sample2_roberts, 3);
    imwrite("case/sample2_roberts1.jpg", sample2_roberts);
    roberts(sample2, sample2_roberts, 4.5);
    imwrite("case/sample2_roberts2.jpg", sample2_roberts);
    prewitt(sample2, sample2_prewitt, 7);
    imwrite("case/sample2_prewitt1.jpg", sample2_prewitt);
    prewitt(sample2, sample2_prewitt, 11);
    imwrite("case/sample2_prewitt2.jpg", sample2_prewitt);
    sobel(sample2, sample2_sobel, 7);
    imwrite("case/sample2_sobel1.jpg", sample2_sobel);
    sobel(sample2, sample2_sobel, 14);
    imwrite("case/sample2_sobel2.jpg", sample2_sobel);

    // 2nd
    Mat sample2_tmp = sample2.clone();
    Mat sample2_2nd_ker1_1 = sample2.clone();
    Mat sample2_2nd_ker2_1 = sample2.clone();
    Mat sample2_2nd_ker1_2 = sample2.clone();
    Mat sample2_2nd_ker2_2 = sample2.clone();

    low_pass_filter(sample2, sample2_tmp);
    laplacian(sample2_tmp, sample2_2nd_ker1_1, 0.5, 4, kernel1);
    laplacian(sample2_tmp, sample2_2nd_ker2_1, 0.5, 8, kernel2);
    laplacian(sample2_tmp, sample2_2nd_ker1_2, 0.99999, 4, kernel1);
    laplacian(sample2_tmp, sample2_2nd_ker2_2, 0.99999, 8, kernel2);
    imwrite("case/sample2_ker1_1.jpg", sample2_2nd_ker1_1);
    imwrite("case/sample2_ker2_1.jpg", sample2_2nd_ker2_1);
    imwrite("case/sample2_ker1_2.jpg", sample2_2nd_ker1_2);
    imwrite("case/sample2_ker2_2.jpg", sample2_2nd_ker2_2);

    // canny
    Mat sample2_canny = sample2.clone();
    canny(sample2, sample2_canny, 10, 5, 7);
    imwrite("case/sample2_canny.jpg", sample2_canny);

    // result of sample2
    imwrite("case/sample2_result.jpg", sample2_prewitt);

    return 0;
}
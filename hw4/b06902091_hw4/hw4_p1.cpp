#include <iostream>
#include <vector>
#include <math.h>
#include <opencv2/opencv.hpp>
#include <cstdlib>

using namespace cv;
using namespace std;

double psnr_func(Mat original, Mat target)
{
    double MSE = 0;
    for (int i = 0; i < original.rows; i++)
    {
        for (int j = 0; j < original.cols; j++)
        {
            MSE += (original.at<uchar>(i, j) - target.at<uchar>(i, j)) * (original.at<uchar>(i, j) - target.at<uchar>(i, j));
        }
    }

    MSE /= (double)(original.rows * original.cols);

    double PSNR = 10 * log10(255 * 255 / MSE);
    return PSNR;
}

void threshold_matrix(int input[][256], int output[][256], int size)
{
    for (int i = 0; i < size; i++)
    {
        for (int j = 0; j < size; j++)
        {
            output[i][j] = 255 * (input[i][j] + 0.5) / (size * size);
        }
    }
}

void halftoning(Mat input, Mat output, int threshold_mat[][256], int mat_size)
{
    for (int i = 0; i < input.rows; i += mat_size)
    {
        for (int j = 0; j < input.cols; j += mat_size)
        {
            for (int a = 0; a < mat_size; a++)
            {
                for (int b = 0; b < mat_size; b++)
                {
                    int x = i + a;
                    int y = j + b;
                    if (x >= input.rows || y >= input.cols)
                    {
                        continue;
                    }
                    if (input.at<uchar>(x, y) >= threshold_mat[a][b])
                    {
                        output.at<uchar>(x, y) = 255;
                    }
                    else
                    {
                        output.at<uchar>(x, y) = 0;
                    }
                }
            }
        }
    }
}

void expand(int input[][256], int output[256][256], int size, int final_size)
{
    int temp[256][256] = {0};
    for (int i = 0; i < size; i++)
    {
        for (int j = 0; j < size; j++)
        {
            temp[i][j] = input[i][j];
        }
    }

    while (size < final_size)
    {
        for (int i = 0; i < size; i++)
        {
            for (int j = 0; j < size; j++)
            {
                output[i][j] = temp[i][j] * 4 + 1;
            }
        }
        for (int i = size; i < size * 2; i++)
        {
            for (int j = 0; j < size; j++)
            {
                output[i][j] = temp[i - size][j] * 4 + 3;
            }
        }
        for (int i = 0; i < size; i++)
        {
            for (int j = size; j < size * 2; j++)
            {
                output[i][j] = temp[i][j - size] * 4 + 2;
            }
        }
        for (int i = size; i < size * 2; i++)
        {
            for (int j = size; j < size * 2; j++)
            {
                output[i][j] = temp[i - size][j - size] * 4 + 0;
            }
        }
        size *= 2;

        for (int i = 0; i < size; i++)
        {
            for (int j = 0; j < size; j++)
            {

                temp[i][j] = output[i][j];
            }
        }
    }
}

void floyd_error_diffusion(Mat input, Mat output)
{
    float diff[2][3] = {{0, 0, 7},
                        {3, 5, 1}};
    float total_error[input.rows][input.cols] = {0};
    float error;
    for (int i = 0; i < input.rows; i++)
    {
        for (int j = 0; j < input.cols; j++)
        {
            //diffused
            input.at<uchar>(i, j) += total_error[i][j];

            //thresholding
            if (input.at<uchar>(i, j) >= 128)
            {
                error = input.at<uchar>(i, j) - 255;
                output.at<uchar>(i, j) = 255;
            }
            else
            {
                error = input.at<uchar>(i, j) - 0;
                output.at<uchar>(i, j) = 0;
            }

            //new error diffusion
            for (int a = 0; a < 2; a++)
            {
                for (int b = 0; b < 3; b++)
                {
                    int x = i + a;
                    int y = j - 1 + b;
                    if (x >= input.rows || y >= output.cols)
                    {
                        continue;
                    }
                    total_error[x][y] += (float)error * diff[a][b] / 16.0;
                }
            }
        }
    }
}

void Jarvis_error_diffusion(Mat input, Mat output)
{
    float diff[3][5] = {{0, 0, 0, 7, 5}, {3, 5, 7, 5, 3}, {1, 3, 5, 3, 1}};
    float total_error[input.rows][input.cols] = {0};
    float error;
    for (int i = 0; i < input.rows; i++)
    {
        for (int j = 0; j < input.cols; j++)
        {
            //diffused
            input.at<uchar>(i, j) += total_error[i][j];

            //thresholding
            if (input.at<uchar>(i, j) >= 128)
            {
                error = input.at<uchar>(i, j) - 255;
                output.at<uchar>(i, j) = 255;
            }
            else
            {
                error = input.at<uchar>(i, j) - 0;
                output.at<uchar>(i, j) = 0;
            }

            //new error diffusion
            for (int a = 0; a < 3; a++)
            {
                for (int b = 0; b < 5; b++)
                {
                    int x = i + a;
                    int y = j - 2 + b;
                    if (x >= input.rows || y >= output.cols)
                    {
                        continue;
                    }
                    total_error[x][y] += (float)error * diff[a][b] / 48.0;
                }
            }
        }
    }
}

void dot_mean(Mat input, Mat output, float max_intensity)
{
    int size = 8;
    for (int i = 0; i < input.rows; i += 8)
    {
        for (int j = 0; j < input.cols; j += 8)
        {
            float count = 0;
            for (int a = 0; a < 8; a++)
            {
                for (int b = 0; b < 8; b++)
                {
                    count += input.at<uchar>(i + a, j + b);
                }
            }
            count = (float)count / 64.0 / max_intensity * 6.0;

            circle(output, Point(j + size / 2, i + size / 2), count, Scalar(255), FILLED);
        }
    }
}
void dot_median(Mat input, Mat output, float max_intensity)
{
    int size = 8;
    float count;
    for (int i = 0; i < input.rows; i += 8)
    {
        for (int j = 0; j < input.cols; j += 8)
        {
            vector<int> v;
            for (int a = 0; a < 8; a++)
            {
                for (int b = 0; b < 8; b++)
                {
                    v.push_back(input.at<uchar>(i + a, j + b));
                }
            }
            sort(v.begin(), v.end());
            count = v[31] / max_intensity * 6.0;
            v.clear();
            circle(output, Point(j + size / 2, i + size / 2), count, Scalar(255), FILLED);
        }
    }
}
int main()
{
    // input image
    Mat sample1 = imread("sample1.png", CV_8UC1);
    Mat sample2 = imread("sample2.png", CV_8UC1);
    Mat sample3 = imread("sample3.png", CV_8UC1);

    // (1a) I_2 halftoning
    int I_2[256][256] = {{1, 2}, {3, 0}};
    I_2[0][0] = 1;
    I_2[0][1] = 2;
    I_2[1][0] = 3;
    I_2[1][1] = 0;
    int T_2[256][256] = {0};
    threshold_matrix(I_2, T_2, 2);

    Mat result(sample1.rows, sample1.cols, CV_8UC1, Scalar(0));
    halftoning(sample1, result, T_2, 2);
    imwrite("result1.png", result);

    // (1b) Expand to I_256 and halftoning
    int I_256[256][256] = {0};
    int T_256[256][256] = {0};
    expand(I_2, I_256, 2, 256);
    threshold_matrix(I_256, T_256, 256);
    halftoning(sample1, result, T_256, 256);
    imwrite("result2.png", result);

    //(1c) error diffusion
    Mat sample1_clone = sample1.clone();
    floyd_error_diffusion(sample1, result);
    imwrite("result3.png", result);
    cout << "Floyd: " << psnr_func(sample1_clone, result)<<endl; 

    //(1c) error diffusion2
    sample1 = sample1_clone.clone();
    Jarvis_error_diffusion(sample1, result);
    imwrite("result4.png", result);
    cout << "Jarvis : " << psnr_func(sample1_clone, result)<<endl; 

    //(1d) dot
    sample1 = sample1_clone.clone();
    float max_intensity = 0;
    for (int i = 0; i < sample1.rows; i += 8)
    {
        for (int j = 0; j < sample1.cols; j += 8)
        {
            if (sample1.at<uchar>(i, j) > max_intensity)
            {
                max_intensity = sample1.at<uchar>(i, j);
            }
        }
    }

    Mat mean(sample1.rows, sample1.cols, CV_8UC1, Scalar(0));
    dot_mean(sample1, mean, max_intensity);
    imwrite("sample1 dotted_mean.png", mean);

    Mat median(sample1.rows, sample1.cols, CV_8UC1, Scalar(0));
    dot_median(sample1, median, max_intensity);
    imwrite("sample1 dotted_median.png", median);

    return 0;
}
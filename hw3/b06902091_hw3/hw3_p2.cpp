#include <iostream>
#include <vector>
#include <math.h>
#include <opencv2/opencv.hpp>
#include <limits.h>
#include <cstdlib>

using namespace cv;
using namespace std;

float conv[9][700][700];
float en[9][700][700];
float minimum_conv[9];
float maximum_conv[9];
float minimum_en[9];
float maximum_en[9];

void normalize_conv(Mat input, Mat output, int which)
{
    for (int i = 0; i < input.rows; i++)
    {
        for (int j = 0; j < input.cols; j++)
        {
            float ans = (conv[which][i][j] - minimum_conv[which]);
            ans = ans / (maximum_conv[which] - minimum_conv[which]);
            ans *= 255;
            output.at<uchar>(i, j) = (int)ans;
        }
    }
}
void normalize_en(Mat input, Mat output, int which)
{
    for (int i = 0; i < input.rows; i++)
    {
        for (int j = 0; j < input.cols; j++)
        {
            float ans = (en[which][i][j] - minimum_en[which]);
            ans = ans / (maximum_en[which] - minimum_en[which]);
            ans *= 255;
            output.at<uchar>(i, j) = (int)ans;
        }
    }
}
void convolution(Mat input, float result[9][700][700], int kernel[9][3][3], int divide[9])
{
    for (int z = 0; z < 9; z++)
    {
        float min = FLT_MAX;
        float max = FLT_MIN;
        for (int i = 0; i < input.rows; i++)
        {
            for (int j = 0; j < input.cols; j++)
            {
                int total = 0;
                for (int a = -1; a <= 1; a++)
                {
                    for (int b = -1; b <= 1; b++)
                    {
                        int x = i + a;
                        int y = j + b;
                        if (x < 0 || x >= input.rows)
                        {
                            x = i;
                        }
                        if (y < 0 || y >= input.cols)
                        {
                            y = j;
                        }
                        total += input.at<uchar>(x, y) * kernel[z][a + 1][b + 1];
                    }
                }
                result[z][i][j] = (float)total / (float)divide[z];
                if (result[z][i][j] < min)
                {
                    min = result[z][i][j];
                }
                if (result[z][i][j] > max)
                {
                    max = result[z][i][j];
                }
            }
        }
        minimum_conv[z] = min;
        maximum_conv[z] = max;
    }
}
void energy(Mat input, int size)
{
    for (int z = 0; z < 9; z++)
    {
        float min = FLT_MAX;
        float max = FLT_MIN;
        for (int i = 0; i < input.rows; i++)
        {
            for (int j = 0; j < input.cols; j++)
            {
                float total = 0;
                for (int a = -size / 2; a <= size / 2; a++)
                {
                    for (int b = -size / 2; b <= size / 2; b++)
                    {
                        int x = i + a;
                        int y = j + b;
                        if (x < 0 || x >= input.rows)
                        {
                            x = i;
                        }
                        if (y < 0 || y >= input.cols)
                        {
                            y = j;
                        }
                        total += conv[z][x][y] * conv[z][x][y];
                    }
                }
                en[z][i][j] = total;
                if (en[z][i][j] < min)
                {
                    min = en[z][i][j];
                }
                if (en[z][i][j] > max)
                {
                    max = en[z][i][j];
                }
            }
        }
        minimum_en[z] = min;
        maximum_en[z] = max;
    }
}

void flood_fill(int i, int j, int color, Mat input)
{
    input.at<uchar>(i, j) = color;
    if (i - 1 >= 0 && input.at<uchar>(i - 1, j) != color)
    {
        flood_fill(i - 1, j, color, input);
    }
    if (j - 1 >= 0 && input.at<uchar>(i, j - 1) != color)
    {
        flood_fill(i, j - 1, color, input);
    }
    if (i + 1 < input.rows && input.at<uchar>(i + 1, j) != color)
    {
        flood_fill(i + 1, j, color, input);
    }
    if (j + 1 < input.cols && input.at<uchar>(i, j + 1) != color)
    {
        flood_fill(i, j + 1, color, input);
    }

    //8 neighbor
    if (i - 1 >= 0 && j - 1 >= 0 && input.at<uchar>(i - 1, j - 1) != color)
    {
        flood_fill(i - 1, j - 1, color, input);
    }
    if (i + 1 < input.rows && j - 1 >= 0 && input.at<uchar>(i + 1, j - 1) != color)
    {
        flood_fill(i + 1, j - 1, color, input);
    }
    if (i - 1 >= 0 && j + 1 < input.cols && input.at<uchar>(i - 1, j + 1) != color)
    {
        flood_fill(i - 1, j + 1, color, input);
    }
    if (i + 1 < input.rows && j + 1 < input.cols && input.at<uchar>(i + 1, j + 1) != color)
    {
        flood_fill(i + 1, j + 1, color, input);
    }
}

void replace(Mat input, Mat output, Mat pattern, int color)
{
    bool truth[700][700] = {false};
    for (int i = 0; i < input.rows; i++)
    {
        for (int j = 0; j < input.cols; j += pattern.cols)
        {
            if (input.at<uchar>(i, j) != color)
            {
                continue;
            }
            if (truth[i][j] == true)
            {
                continue;
            }

            for (int a = 0; a < pattern.rows; a++)
            {
                for (int b = 0; b < pattern.cols; b++)
                {
                    if (i + a < input.rows && j + b < input.cols)
                    {
                        if (truth[i + a][j + b] == false)
                        {
                            output.at<uchar>(i + a, j + b) = pattern.at<uchar>(a, b);
                            truth[i + a][j + b] = true;
                        }
                    }
                }
            }
        }
    }
}
int main()
{
    // (-) input image
    Mat sample2 = imread("sample2.png", CV_8UC1);
    Mat sample3 = imread("sample3.png", CV_8UC1);

    // (2a)
    // #1 convolution
    int ker[9][3][3] = {{1, 2, 1, 2, 4, 2, 1, 2, 1},
                        {1, 0, -1, 2, 0, -2, 1, 0, -1},
                        {-1, 2, -1, -2, 4, -2, -1, 2, -1},
                        {-1, -2, -1, 0, 0, 0, 1, 2, 1},
                        {1, 0, -1, 0, 0, 0, -1, 0, 1},
                        {-1, 2, -1, 0, 0, 0, 1, -2, 1},
                        {-1, -2, -1, 2, 4, 2, -1, -2, -1},
                        {-1, 0, 1, 2, 0, -2, -1, 0, 1},
                        {1, -2, 1, -2, 4, -2, 1, -2, 1}};

    int div[9] = {36, 12, 12, 12, 4, 4, 12, 4, 4};
    convolution(sample2, conv, ker, div);

    // normalize convolution output
    Mat temp_out(sample2.rows, sample2.cols, CV_8UC1, Scalar(0));
    for (int z = 0; z < 9; z++)
    {
        normalize_conv(sample2, temp_out, z);
        string name = "case/sample2_law_" + to_string(z + 1);
        name = name + ".png";
        imwrite(name, temp_out);
    }

    // #2 energy
    energy(sample2, 31);

    // normalize convolution output
    for (int z = 0; z < 9; z++)
    {
        normalize_en(sample2, temp_out, z);
        string name = "case/sample2_energy_" + to_string(z + 1);
        name = name + ".png";
        imwrite(name, temp_out);
    }

    // #3 kmeans
    Mat temp = sample2.clone();

    int tag[700][700] = {0};
    int k_max = 4;
    float mean[9][4] = {0};

    for (int z = 0; z < 9; z++)
    {
        mean[z][0] = en[z][19][220];
        mean[z][1] = en[z][205][205];
        mean[z][2] = en[z][333][333];
        mean[z][3] = en[z][134][294];
    }

    // iteration
    int flag = 1;
    int iteration_count = 0;
    while (1)
    {
        iteration_count++;
        flag = 0;
        for (int i = 0; i < sample2.rows; i++)
        {
            for (int j = 0; j < sample2.cols; j++)
            {
                float dist[3] = {0};
                for (int z = 0; z < 9; z++)
                {
                    for (int k = 0; k < k_max; k++)
                    {
                        dist[k] += pow(en[z][i][j] - mean[z][k], 2);
                    }
                }

                float min = 10000000;
                int min_loc = -1;
                for (int k = 0; k < k_max; k++)
                {
                    dist[k] = sqrt(dist[k]);
                    if (dist[k] < min)
                    {
                        min = dist[k];
                        min_loc = k;
                    }
                }

                if (min_loc == 0 && tag[i][j] != 1)
                {
                    tag[i][j] = 1;
                    flag = 1;
                }
                else if (min_loc == 1 && tag[i][j] != 2)
                {
                    tag[i][j] = 2;
                    flag = 1;
                }
                else if (min_loc == 2 && tag[i][j] != 3)
                {
                    tag[i][j] = 3;
                    flag = 1;
                }
                else if (min_loc == 3 && tag[i][j] != 4)
                {
                    tag[i][j] = 4;
                    flag = 1;
                }
            }
        }
        if (flag == 0)
        {
            break;
        }

        //new mean
        float tag_total[9][4] = {0};
        float tag_count[4] = {0};
        for (int i = 0; i < sample2.rows; i++)
        {
            for (int j = 0; j < sample2.cols; j++)
            {
                for (int z = 0; z < 9; z++)
                {
                    tag_total[z][tag[i][j] - 1] += en[z][i][j];
                }
                tag_count[tag[i][j] - 1]++;
            }
        }
        for (int z = 0; z < 9; z++)
        {
            for (int k = 0; k < k_max; k++)
            {
                mean[z][k] = tag_total[z][k] / tag_count[k];
            }
        }
    }

    //output segmentation result
    Mat result3(sample2.rows, sample2.cols, CV_8UC1, Scalar(0));
    for (int i = 0; i < sample2.rows; i++)
    {
        for (int j = 0; j < sample2.cols; j++)
        {
            result3.at<uchar>(i, j) = (tag[i][j] - 1) * 50;
        }
    }
    cout << "k-mean itertaion: " << iteration_count << endl;
    imwrite("result3.png", result3);

    // (c) better result
    // hole filling
    flood_fill(398, 251, 100, result3);
    flood_fill(399, 322, 100, result3);
    flood_fill(391, 451, 100, result3);
    flood_fill(398, 251, 100, result3);
    imwrite("result4.png", result3);

    // (bounus)
    Mat result4 = sample2.clone();
    replace(result3, result4, sample3, 100);
    imwrite("result5.png", result4);
    return 0;
}
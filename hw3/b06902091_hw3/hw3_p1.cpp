#include <iostream>
#include <vector>
#include <math.h>
#include <opencv2/opencv.hpp>
#include <cstdlib>

using namespace cv;
using namespace std;

int label[500][500];
int equ[5000];

void boundary_extract(Mat input, Mat output, int size)
{
    Mat temp(input.rows, input.cols, CV_8UC1, Scalar(0));
    for (int i = size / 2; i <= input.rows - 1 - size / 2; i++)
    {
        for (int j = size / 2; j <= input.cols - 1 - size / 2; j++)
        {
            int count = 0;
            for (int a = i - size / 2; a <= i + size / 2; a++)
            {
                for (int b = j - size / 2; b <= j + size / 2; b++)
                {

                    if (input.at<uchar>(a, b) == 255)
                    {
                        count++;
                    }
                }
            }
            if (count == size * size)
            {
                temp.at<uchar>(i, j) = 255;
            }
            else
            {
                temp.at<uchar>(i, j) = 0;
            }
            output.at<uchar>(i, j) = input.at<uchar>(i, j) - temp.at<uchar>(i, j);
        }
    }
}

void filling(Mat input, Mat output, Mat pos)
{
    while (1)
    {
        int change = 0;
        for (int i = 1; i < input.rows - 1; i++)
        {
            for (int j = 1; j < input.cols - 1; j++)
            {
                //touch point
                if (pos.at<uchar>(i, j) == 255)
                {
                    if (input.at<uchar>(i - 1, j - 1) == 0 && pos.at<uchar>(i - 1, j - 1) != 255)
                    {
                        pos.at<uchar>(i - 1, j - 1) = 255;
                        change = 1;
                    }
                    if (input.at<uchar>(i - 1, j) == 0 && pos.at<uchar>(i - 1, j) != 255)
                    {
                        pos.at<uchar>(i - 1, j) = 255;
                        change = 1;
                    }
                    if (input.at<uchar>(i - 1, j + 1) == 0 && pos.at<uchar>(i - 1, j + 1) != 255)
                    {
                        pos.at<uchar>(i - 1, j + 1) = 255;
                        change = 1;
                    }
                    if (input.at<uchar>(i, j - 1) == 0 && pos.at<uchar>(i, j - 1) != 255)
                    {
                        pos.at<uchar>(i, j - 1) = 255;
                        change = 1;
                    }
                    if (input.at<uchar>(i, j + 1) == 0 && pos.at<uchar>(i, j + 1) != 255)
                    {
                        pos.at<uchar>(i, j + 1) = 255;
                        change = 1;
                    }
                    if (input.at<uchar>(i + 1, j - 1) == 0 && pos.at<uchar>(i + 1, j - 1) != 255)
                    {
                        pos.at<uchar>(i + 1, j - 1) = 255;
                        change = 1;
                    }
                    if (input.at<uchar>(i + 1, j) == 0 && pos.at<uchar>(i + 1, j) != 255)
                    {
                        pos.at<uchar>(i + 1, j) = 255;
                        change = 1;
                    }
                    if (input.at<uchar>(i + 1, j + 1) == 0 && pos.at<uchar>(i + 1, j + 1) != 255)
                    {
                        pos.at<uchar>(i + 1, j + 1) = 255;
                        change = 1;
                    }
                }
            }
        }
        if (change == 0)
        {
            break;
        }
    }

    //apply
    for (int i = 0; i < input.rows; i++)
    {
        for (int j = 0; j < input.cols; j++)
        {
            output.at<uchar>(i, j) += pos.at<uchar>(i, j);
        }
    }
}

void connecting(Mat input, Mat output)
{
    //initial
    for (int z = 0; z < 50; z++)
    {
        int tag = 1;
        for (int i = 1; i < input.rows - 1; i++)
        {
            for (int j = 1; j < input.cols - 1; j++)
            {
                if (input.at<uchar>(i, j) == 0)
                {
                    label[i][j] = 0;
                    continue;
                }

                int min = 10000;
                int min_which = -1;
                int new_label[8] = {0, 0, 0, 0, 0, 0, 0, 0};
                if (label[i - 1][j] != 0)
                {
                    new_label[0] = label[i - 1][j];
                    if (new_label[0] < min)
                    {
                        min = new_label[0];
                        min_which = 0;
                    }
                }
                if (label[i][j - 1] != 0)
                {
                    new_label[1] = label[i][j - 1];
                    if (new_label[1] < min)
                    {
                        min = new_label[1];
                        min_which = 1;
                    }
                }
                if (label[i - 1][j - 1] != 0)
                {
                    new_label[2] = label[i - 1][j - 1];
                    if (new_label[2] < min)
                    {
                        min = new_label[2];
                        min_which = 2;
                    }
                }
                if (label[i - 1][j + 1] != 0)
                {
                    new_label[3] = label[i - 1][j + 1];
                    if (new_label[3] < min)
                    {
                        min = new_label[3];
                        min_which = 3;
                    }
                }
                if (label[i][j + 1] != 0)
                {
                    new_label[4] = label[i][j + 1];
                    if (new_label[4] < min)
                    {
                        min = new_label[4];
                        min_which = 4;
                    }
                }
                if (label[i + 1][j] != 0)
                {
                    new_label[5] = label[i + 1][j];
                    if (new_label[5] < min)
                    {
                        min = new_label[5];
                        min_which = 5;
                    }
                }
                if (label[i + 1][j + 1] != 0)
                {
                    new_label[6] = label[i + 1][j + 1];
                    if (new_label[6] < min)
                    {
                        min = new_label[6];
                        min_which = 6;
                    }
                }
                if (label[i + 1][j - 1] != 0)
                {
                    new_label[7] = label[i + 1][j - 1];
                    if (new_label[7] < min)
                    {
                        min = new_label[7];
                        min_which = 7;
                    }
                }

                if (min == 10000)
                {
                    label[i][j] = tag;
                    tag += 1;
                    equ[tag] = tag;
                }
                else
                {
                    label[i][j] = min;
                    for (int x = 0; x < 8; x++)
                    {
                        if (new_label[x] == 0)
                        {
                            continue;
                        }
                        else
                        {
                            equ[new_label[x]] = min;
                        }
                    }
                }
            }
        }
    }

    //apply equvalence table
    int truth[255] = {0};
    for (int i = 0; i < input.rows; i++)
    {
        for (int j = 0; j < input.cols; j++)
        {
            if (input.at<uchar>(i, j) == 0)
            {
                output.at<uchar>(i, j) = 255;
                continue;
            }
            label[i][j] = equ[label[i][j]];
            output.at<uchar>(i, j) = label[i][j];
            truth[label[i][j]] = 1;
        }
    }

    //count no. of labels
    int num = 0;
    for (int i = 1; i < 255; i++)
    {
        if (truth[i] == 1)
        {
            num++;
        }
    }
    cout << "count: " << num << endl;
}

int main()
{
    // (-) input image
    Mat sample1 = imread("sample1.png", CV_8UC1);

    // (1a)
    Mat result1 = sample1.clone();
    boundary_extract(sample1, result1, 3);
    imwrite("result1.png", result1);

    Mat result1_2 = sample1.clone();
    boundary_extract(sample1, result1_2, 9);
    imwrite("case/result1_2.png", result1_2);

    Mat result1_3 = sample1.clone();
    boundary_extract(sample1, result1_3, 17);
    imwrite("case/result1_3.png", result1_3);

    // (1b)
    Mat result2 = sample1.clone();
    Mat position(sample1.rows, sample1.cols, CV_8UC1, Scalar(0));

    position.at<uchar>(59, 142) = 255;
    position.at<uchar>(188, 50) = 255;
    position.at<uchar>(171, 86) = 255;
    position.at<uchar>(232, 74) = 255;
    position.at<uchar>(302, 166) = 255;
    position.at<uchar>(196, 293) = 255;

    imwrite("case/result2_pos.png", position);
    filling(result1, result2, position);
    imwrite("result2.png", result2);

    // (1c)
    Mat result3 = result2.clone();
    connecting(result2, result3);
    imwrite("case/result2_count.png", result3);

    return 0;
}
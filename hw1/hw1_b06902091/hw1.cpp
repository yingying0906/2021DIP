#include <iostream>
#include <fstream>
#include <vector>
#include <math.h>
#include <opencv2/opencv.hpp>
#include <cstdlib>
using namespace cv;
using namespace std;

Mat histogram(Mat img, int bin[])
{
    int max = 0;

    //pixel value count
    for (int i = 0; i < img.rows; i++)
    {
        for (int j = 0; j < img.cols; j++)
        {
            bin[img.at<uchar>(i, j)]++;
        }
    }

    //highest intensity
    for (int i = 0; i < 256; i++)
    {
        if (bin[i] >= max)
        {
            max = bin[i];
        }
    }

    //create graph
    Mat graph(768, 768, CV_8UC1, Scalar(255));
    float ratio = max / 768.0;
    for (int x = 0; x < 256; x++)
    {
        for (int y = 0; y < bin[x]; y++)
        {
            graph.at<uchar>(767 - floor(y / ratio), x * 3) = 0;
            graph.at<uchar>(767 - floor(y / ratio), x * 3 + 1) = 0;
            graph.at<uchar>(767 - floor(y / ratio), x * 3 + 2) = 0;
        }
    }

    return graph;
}

void global_eq(Mat input, Mat output, int bin[])
{
    //cdf
    int cdf[256] = { 0 };
    cdf[0] = bin[0];
    for (int i = 1; i < 256; i++)
    {
        cdf[i] = cdf[i - 1] + bin[i];
    }
    for (int i = 0; i < 256; i++)
    {
        cdf[i] /= (input.rows * input.cols / 255);
    }

    //apply new intensity
    for (int i = 0; i < input.rows; i++)
    {
        for (int j = 0; j < input.cols; j++)
        {
            output.at<uchar>(i, j) = cdf[input.at<uchar>(i, j)];
        }
    }
}

void local_eq(Mat input, Mat output, int kernel_size)
{
    for (int i = 0; i < input.rows; i++)
    {
        for (int j = 0; j < input.cols; j++)
        {
            //cdf + histo
            int bin_temp[256] = { 0 };
            float cdf_temp[256] = { 0 };
            float total = 0;

            for (int a = -kernel_size / 2; a <= kernel_size / 2; a++)
            {
                for (int b = -kernel_size / 2; b <= kernel_size / 2; b++)
                {
                    int x = i + a;
                    int y = j + b;
                    if (x < 0)
                    {
                        x = 0;
                    }
                    if (x >= input.rows) {
                        x = input.rows - 1;
                    }
                    if (y < 0) {
                        y = 0;
                    }
                    if (y >= input.cols) {
                        y = input.cols - 1;
                    }

                    bin_temp[output.at<uchar>(x, y)]++;
                    total += 1;
                }
            }
            cdf_temp[0] = bin_temp[0];
            for (int k = 1; k < 256; k++)
            {
                cdf_temp[k] = cdf_temp[k - 1] + bin_temp[k];
            }
            for (int k = 0; k < 256; k++)
            {
                cdf_temp[k] /= (total / 255);
            }
            output.at<uchar>(i, j) = (int)cdf_temp[output.at<uchar>(i, j)];
        }
    }
}

void low_pass_filter(Mat input, Mat output, int size, int ker)
{
    int kernel[3][3][3] = { {1,1,1,1,1,1,1,1,1},{1,1,1,1,2,1,1,1,1},{1,2,1,2,4,2,1,2,1} };
    int divide[3] = { 9,10,16 };
    for (int i = 0; i < input.rows; i++)
    {
        for (int j = 0; j < input.cols; j++)
        {
            int sum = 0;
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
                    if (x >= input.rows) {
                        x = input.rows - 1;
                    }
                    if (y < 0) {
                        y = 0;
                    }
                    if (y >= input.cols) {
                        y = input.cols - 1;
                    }

                    if (size == 5) {
                        sum += output.at<uchar>(x, y);
                    }
                    else {
                        sum += output.at<uchar>(x, y) * kernel[ker][a + 1][b + 1];
                    }
                    
                }
            }

            if (size == 5) {
                output.at<uchar>(i, j) = sum / 25;
            }
            else {
                output.at<uchar>(i, j) = sum / divide[ker];
            }
            
        }
    }
}

void median_filter(Mat input, Mat output, int size)
{
    for (int i = 0; i < input.rows; i++)
    {
        for (int j = 0; j < input.cols; j++)
        {
            vector<int> temp;
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
                    if (x >= input.rows) {
                        x = input.rows - 1;
                    }
                    if (y < 0) {
                        y = 0;
                    }
                    if (y >= input.cols) {
                        y = input.cols - 1;
                    }
                        temp.push_back(output.at<uchar>(x, y));
                        count++;
                }
            }
            int mid = count / 2;
            sort(temp.begin(), temp.end());
            output.at<uchar>(i, j) = temp.at(mid);
        }
    }
}

void outlier(Mat input, Mat output, int size, int threshold)
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
                    if (a == 0 && b == 0)
                    {
                        continue;
                    }

                    int x = i + a;
                    int y = j + b;
                    if (x < 0)
                    {
                        x = 0;
                    }
                    if (x >= input.rows) {
                        x = input.rows - 1;
                    }
                    if (y < 0) {
                        y = 0;
                    }
                    if (y >= input.cols) {
                        y = input.cols - 1;
                    }

                        sum += output.at<uchar>(x, y);
                        count++;

                }
            }
            float mean = sum / count;
            float ans = abs(output.at<uchar>(i, j) - mean);

            if (ans > threshold)
            {
                output.at<uchar>(i, j) = mean;
            }
        }
    }
}

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

int main()
{
    // input image
    Mat sample1 = imread("sample1.jpg");
    Mat sample2 = imread("sample2.jpg", CV_8UC1);
    Mat sample3 = imread("sample3.jpg", CV_8UC1);
    Mat sample4 = imread("sample4.jpg", CV_8UC1);
    Mat sample5 = imread("sample5.jpg", CV_8UC1);
    Mat sample6 = imread("sample6.jpg", CV_8UC1);
    Mat sample7 = imread("sample7.jpg", CV_8UC1);

    // Problem1
    // (a) grayscale
    Mat sample1_gray(sample1.rows, sample1.cols, CV_8UC1, Scalar(0));

    for (int i = 0; i < sample1.rows; i++)
    {
        for (int j = 0; j < sample1.cols; j++)
        {
            sample1_gray.at<uchar>(i, j) = sample1.at<Vec3b>(i, j)[2] * 0.299 + sample1.at<Vec3b>(i, j)[1] * 0.587 + sample1.at<Vec3b>(i, j)[0] * 0.114;
        }
    }

    // (b) horizontal flipping
    Mat sample1_flip(sample1.rows, sample1.cols, CV_8UC3, Scalar(0));

    for (int i = 0; i < sample1.rows; i++)
    {
        for (int j = 0; j < sample1.cols; j++)
        { //BGR RGB
            sample1_flip.at<Vec3b>(i, j) = sample1.at<Vec3b>(i, sample1.cols - j - 1);
        }
    }

    // Problem2
    // (a) divided by 5
    Mat sample2_divide = sample2.clone();

    for (int i = 0; i < sample2.rows; i++)
    {
        for (int j = 0; j < sample2.cols; j++)
        { //BGR RGB
            sample2_divide.at<uchar>(i, j) /= 5;
        }
    }

    // (b) multiply by 5
    Mat sample2_multi = sample2_divide.clone();

    for (int i = 0; i < sample2.rows; i++)
    {
        for (int j = 0; j < sample2.cols; j++)
        {
            sample2_multi.at<uchar>(i, j) *= 5;
        }
    }

    // (c) histogram
    int bin_2[256] = { 0 };
    int bin_2_divide[256] = { 0 };
    int bin_2_multi[256] = { 0 };

    Mat hist_sample2 = histogram(sample2, bin_2);
    Mat hist_sample2_divide = histogram(sample2_divide, bin_2_divide);
    Mat hist_sample2_multi = histogram(sample2_multi, bin_2_multi);

    imwrite("hist_sample2.jpg", hist_sample2);
    imwrite("hist_sample2_divide.jpg", hist_sample2_divide);
    imwrite("hist_sample2_multi.jpg", hist_sample2_multi);

    // (d) global histogram equalization
    int bin_3[256] = { 0 };
    Mat hist_sample3 = histogram(sample3, bin_3);
    imwrite("hist_sample3.jpg", hist_sample3);

    Mat sample3_global = sample3.clone();
    global_eq(sample3, sample3_global, bin_3);

    // (e) local histogram equalization
    Mat sample3_local = sample3.clone();
    local_eq(sample3, sample3_local, 60);

    // (f) histogram result_5_6
    int bin_3_global[256] = { 0 };
    int bin_3_local[256] = { 0 };

    Mat hist_sample3_global = histogram(sample3_global, bin_3_global);
    Mat hist_sample3_local = histogram(sample3_local, bin_3_local);

    imwrite("hist_sample3_global.jpg", hist_sample3_global);
    imwrite("hist_sample3_local.jpg", hist_sample3_local);

    // (g) enhancement
    // global
    int bin_4[256] = { 0 };
    Mat hist_sample4 = histogram(sample4, bin_4);
    imwrite("hist_sample4.jpg", hist_sample4);

    Mat sample4_global = sample4.clone();
    global_eq(sample4, sample4_global, bin_4);

    int bin_4_global[256] = { 0 };
    Mat hist_sample4_global = histogram(sample4_global, bin_4_global);
    imwrite("hist_sample4_global.jpg", hist_sample4_global);

    // power law
    Mat sample4_power = sample4.clone();
    for (int i = 0; i < sample4.rows; i++)
    {
        for (int j = 0; j < sample4.cols; j++)
        {
            float temp = (float)sample4_power.at<uchar>(i, j);
            temp = temp / 256.0;
            temp = pow(temp, 0.7);
            temp *= 256.0;
            sample4_power.at<uchar>(i, j) = temp;
        }
    }
    int bin_4_power[256] = { 0 };
    Mat hist_sample4_power = histogram(sample4_power, bin_4_power);
    imwrite("hist_sample4_power.jpg", hist_sample4_power);

    // Problem 3
    //sample 6 lowpass
    Mat sample6_lowpass_3 = sample6.clone();
    Mat sample6_lowpass_5 = sample6.clone();
    low_pass_filter(sample6, sample6_lowpass_3, 3,0);
    low_pass_filter(sample6, sample6_lowpass_5, 5,0);

    Mat sample6_lowpass_3_1 = sample6.clone();
    Mat sample6_lowpass_3_2 = sample6.clone();
    low_pass_filter(sample6, sample6_lowpass_3_1, 3, 1);
    low_pass_filter(sample6, sample6_lowpass_3_2, 3, 2);

    //sample7 median
    Mat sample7_median_3 = sample7.clone();
    Mat sample7_median_5 = sample7.clone();
    median_filter(sample7, sample7_median_3, 3);
    median_filter(sample7, sample7_median_5, 5);

    //sample7 outlier
    Mat sample7_outlier_3_50 = sample7.clone();
    Mat sample7_outlier_5_50 = sample7.clone();
    outlier(sample7, sample7_outlier_3_50, 3, 50);
    outlier(sample7, sample7_outlier_5_50, 5, 50);

    Mat sample7_outlier_3_65 = sample7.clone();
    Mat sample7_outlier_5_65 = sample7.clone();
    outlier(sample7, sample7_outlier_3_65, 3, 65);
    outlier(sample7, sample7_outlier_5_65, 5, 65);

    //output result
    imwrite("1_result.jpg", sample1_gray);
    imwrite("2_result.jpg", sample1_flip);

    imwrite("3_result.jpg", sample2_divide);
    imwrite("4_result.jpg", sample2_multi);
    imwrite("5_result.jpg", sample3_global);
    imwrite("6_result.jpg", sample3_local);

    imwrite("7_result.jpg", sample4_power);
    imwrite("7_result_global.jpg", sample4_global);
    imwrite("7_result_power.jpg", sample4_power);

    imwrite("8_result.jpg", sample6_lowpass_3_2);
    imwrite("8_result_lowpass_3.jpg", sample6_lowpass_3);
    imwrite("8_result_lowpass_5.jpg", sample6_lowpass_5);
    imwrite("8_result_lowpass_3_1.jpg", sample6_lowpass_3_1);
    imwrite("8_result_lowpass_3_2.jpg", sample6_lowpass_3_2);

    imwrite("9_result.jpg", sample7_median_3);
    imwrite("9_result_median_3.jpg", sample7_median_3);
    imwrite("9_result_median_5.jpg", sample7_median_5);
    imwrite("9_result_outlier_3_50.jpg", sample7_outlier_3_50);
    imwrite("9_result_outlier_5_50.jpg", sample7_outlier_5_50);
    imwrite("9_result_outlier_3_65.jpg", sample7_outlier_3_65);
    imwrite("9_result_outlier_5_65.jpg", sample7_outlier_5_65);

    //PSNR
    cout << "sample6 = " << psnr_func(sample5, sample6) << endl;
    cout << "8_result_lowpass_3x3_kernel_0 = " << psnr_func(sample5, sample6_lowpass_3) << endl;
    cout << "8_result_lowpass_5x5_kernel_0 = " << psnr_func(sample5, sample6_lowpass_5) << endl;
    cout << "8_result_lowpass_3x3_kernel_1 = " << psnr_func(sample5, sample6_lowpass_3_1) << endl;
    cout << "8_result_lowpass_3x3_kernel_2 = " << psnr_func(sample5, sample6_lowpass_3_2) << endl;

    cout << endl;
    cout << "sample7 = " << psnr_func(sample5, sample7) << endl;
    cout << "9_result_median_3x3 = " << psnr_func(sample5, sample7_median_3) << endl;
    cout << "9_result_median_5x5 = " << psnr_func(sample5, sample7_median_5) << endl;
    cout << "9_result_outlier_3x3_t_50 = " << psnr_func(sample5, sample7_outlier_3_50) << endl;
    cout << "9_result_outlier_5x5_t_50 = " << psnr_func(sample5, sample7_outlier_5_50) << endl;
    cout << "9_result_outlier_3x3_t_65 = " << psnr_func(sample5, sample7_outlier_3_65) << endl;
    cout << "9_result_outlier_5x5_t_65 = " << psnr_func(sample5, sample7_outlier_5_65) << endl;

    return 0;
}

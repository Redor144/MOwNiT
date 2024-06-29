#include <stdio.h>
#include <math.h>

float f1_f(float x)
{
    return powf(x, 8) - 8 * powf(x, 7) + 28 * powf(x, 6) - 56 * powf(x, 5) + 70 * powf(x, 4) - 56 * powf(x, 3) + 28 * powf(x, 2) - 8 * x + 1;
}

double f1_d(double x)
{
    return pow(x, 8) - 8 * pow(x, 7) + 28 * pow(x, 6) - 56 * pow(x, 5) + 70 * pow(x, 4) - 56 * pow(x, 3) + 28 * pow(x, 2) - 8 * x + 1;
}

long double f1_ld(long double x)
{
    return powl(x, 8) - 8 * powl(x, 7) + 28 * powl(x, 6) - 56 * powl(x, 5) + 70 * powl(x, 4) - 56 * powl(x, 3) + 28 * powl(x, 2) - 8 * x + 1;
}

float f2_f(float x)
{
    return (((((((x - 8) * x + 28) * x - 56) * x + 70) * x - 56) * x + 28) * x - 8) * x + 1;
}

double f2_d(double x)
{
    return (((((((x - 8) * x + 28) * x - 56) * x + 70) * x - 56) * x + 28) * x - 8) * x + 1;
}

long double f2_ld(long double x)
{
    return (((((((x - 8) * x + 28) * x - 56) * x + 70) * x - 56) * x + 28) * x - 8) * x + 1;
}

float f3_f(float x)
{
    return powf((x - 1), 8);
}

double f3_d(double x)
{
    return pow((x - 1), 8);
}

long double f3_ld(long double x)
{
    return powl((x - 1), 8);
}

float f4_f(float x)
{
    if (x != 1)
    {
        return expf(8 * logf(fabsf(x - 1)));
    }
    else
    {
        return NAN;
    }
}

double f4_d(double x)
{
    if (x != 1)
    {
        return exp(8 * log(fabs(x - 1)));
    }
    else
    {
        return NAN;
    }
}

long double f4_ld(long double x)
{
    if (x != 1)
    {
        return expl(8 * logl(fabsl(x - 1)));
    }
    else
    {
        return NAN;
    }
}

void exportToCSV_float(float array[], int size, const char *filename)
{
    FILE *file = fopen(filename, "w");

    if (file != NULL)
    {
        for (int i = 0; i < size; i++)
        {
            fprintf(file, "%.30f", array[i]);
            if (i < size - 1)
            {
                fprintf(file, "\n");
            }
        }
        fclose(file);
        printf("Array exported to %s\n", filename);
    }
    else
    {
        printf("Error opening file: %s\n", filename);
    }
}

void exportToCSV_double(double array[], int size, const char *filename)
{
    FILE *file = fopen(filename, "w");

    if (file != NULL)
    {
        for (int i = 0; i < size; i++)
        {
            fprintf(file, "%.30lf", array[i]);
            if (i < size - 1)
            {
                fprintf(file, "\n");
            }
        }
        fclose(file);
        printf("Array exported to %s\n", filename);
    }
    else
    {
        printf("Error opening file: %s\n", filename);
    }
}

void exportToCSV_long_double(long double array[], int size, const char *filename)
{
    FILE *file = fopen(filename, "w");

    if (file != NULL)
    {
        for (int i = 0; i < size; i++)
        {
            fprintf(file, "%.30Lf", array[i]);
            if (i < size - 1)
            {
                fprintf(file, "\n");
            }
        }
        fclose(file);
        printf("Array exported to %s\n", filename);
    }
    else
    {
        printf("Error opening file: %s\n", filename);
    }
}

int main()
{
    int num_elem = 101;

    float small[num_elem];
    double medium[num_elem];
    long double large[num_elem];

    float start_s = 0.99f;
    float end_s = 1.01f;
    float step_s = (end_s - start_s) / (num_elem - 1);

    double start_m = 0.99;
    double end_m = 1.01;
    double step_m = (end_m - start_m) / (num_elem - 1);

    long double start_l = 0.99L;
    long double end_l = 1.01L;
    long double step_l = (end_l - start_l) / (num_elem - 1);

    for (int i = 0; i < num_elem; i++)
    {
        small[i] = start_s + i * step_s;
        medium[i] = start_m + i * step_m;
        large[i] = start_l + i * step_l;
    }

    float small_f1[num_elem];
    float small_f2[num_elem];
    float small_f3[num_elem];
    float small_f4[num_elem];

    int small_size = sizeof(small) / sizeof(small[0]);

    double medium_f1[num_elem];
    double medium_f2[num_elem];
    double medium_f3[num_elem];
    double medium_f4[num_elem];

    int medium_size = sizeof(medium) / sizeof(medium[0]);

    long double large_f1[num_elem];
    long double large_f2[num_elem];
    long double large_f3[num_elem];
    long double large_f4[num_elem];

    int large_size = sizeof(large) / sizeof(large[0]);

    for (int i = 0; i < num_elem; i++)
    {
        small_f1[i] = f1_f(small[i]);
        small_f2[i] = f2_f(small[i]);
        small_f3[i] = f3_f(small[i]);
        small_f4[i] = f4_f(small[i]);

        medium_f1[i] = f1_d(medium[i]);
        medium_f2[i] = f2_d(medium[i]);
        medium_f3[i] = f3_d(medium[i]);
        medium_f4[i] = f4_d(medium[i]);

        large_f1[i] = f1_ld(large[i]);
        large_f2[i] = f2_ld(large[i]);
        large_f3[i] = f3_ld(large[i]);
        large_f4[i] = f4_ld(large[i]);
    }

    const char *filename10 = "output_f.csv";
    const char *filename11 = "output_f_f1.csv";
    const char *filename12 = "output_f_f2.csv";
    const char *filename13 = "output_f_f3.csv";
    const char *filename14 = "output_f_f4.csv";
    exportToCSV_float(small, small_size, filename10);
    exportToCSV_float(small_f1, small_size, filename11);
    exportToCSV_float(small_f2, small_size, filename12);
    exportToCSV_float(small_f3, small_size, filename13);
    exportToCSV_float(small_f4, small_size, filename14);

    const char *filename20 = "output_d.csv";
    const char *filename21 = "output_d_f1.csv";
    const char *filename22 = "output_d_f2.csv";
    const char *filename23 = "output_d_f3.csv";
    const char *filename24 = "output_d_f4.csv";
    exportToCSV_double(medium, medium_size, filename20);
    exportToCSV_double(medium_f1, medium_size, filename21);
    exportToCSV_double(medium_f2, medium_size, filename22);
    exportToCSV_double(medium_f3, medium_size, filename23);
    exportToCSV_double(medium_f4, medium_size, filename24);

    const char *filename30 = "output_ld.csv";
    const char *filename31 = "output_ld_f1.csv";
    const char *filename32 = "output_ld_f2.csv";
    const char *filename33 = "output_ld_f3.csv";
    const char *filename34 = "output_ld_f4.csv";
    exportToCSV_long_double(large, large_size, filename30);
    exportToCSV_long_double(large_f1, large_size, filename31);
    exportToCSV_long_double(large_f2, large_size, filename32);
    exportToCSV_long_double(large_f3, large_size, filename33);
    exportToCSV_long_double(large_f4, large_size, filename34);

    return 0;
}
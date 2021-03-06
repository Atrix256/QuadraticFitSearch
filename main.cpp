#include "stdio.h"
#include <vector>
#include <random>
#include <thread>
#include <atomic>
#include <string>
#include <chrono>
#include <array>
#include <assert.h>

// TODO: c_maxNumValues should be 1000, c_numRunsPerTest should be 100

static const size_t c_maxValue = 2000;           // the sorted arrays will have values between 0 and this number in them (inclusive)
static const size_t c_maxNumValues = 100;       // the graphs will graph between 1 and this many values in a sorted array
static const size_t c_numRunsPerTest = 10;      // how many times does it do the same test to gather min, max, average?
static const size_t c_perfTestNumSearches = 100000; // how many searches are going to be done per list type, to come up with timing for a search type.

#define VERIFY_RESULT() 1 // verifies that the search functions got the right answer. prints out a message if they didn't.
#define MAKE_CSVS() 1 // the main test
#define DO_PERF_TESTS() 0

struct TestResults
{
    bool found;
    size_t index;
    size_t guesses;
};

typedef std::vector<std::string> TRow;
typedef std::vector<TRow> TSheet;

using MakeListFn = void(*)(std::vector<size_t>& values, size_t count);
using TestListFn = TestResults(*)(const std::vector<size_t>& values, size_t searchValue);
using InitialFitFn = void(*)(TSheet& csv, const std::vector<size_t>& values);

using Vec2u = std::array<size_t, 2>;
using Vec3u = std::array<size_t, 3>;
using Vec3 = std::array<float, 3>;
using Vec3d = std::array<double, 3>;

struct MakeListInfo
{
    const char* name;
    MakeListFn fn;
};

struct TestListInfo
{
    const char* name;
    TestListFn fn;
    InitialFitFn initialFitFn;
};

#define countof(array) (sizeof(array) / sizeof(array[0]))

template <typename T>
T Clamp(T min, T max, T value)
{
    if (value < min)
        return min;
    else if (value > max)
        return max;
    else
        return value;
}

float Lerp(float a, float b, float t)
{
    return (1.0f - t) * a + t * b;
}

void Validate(float f)
{
    // TODO: make this an assert / printf before calling it good?
    //assert(isfinite(f));
    if (!isfinite(f))
        int ijkl = 0;
}

void Validate(double f)
{
    // TODO: make this an assert / printf before calling it good?
    //assert(isfinite(f));
    if (!isfinite(f))
        int ijkl = 0;
}

void Validate(const Vec3& v)
{
    Validate(v[0]);
    Validate(v[1]);
    Validate(v[2]);
}

void Validate(const Vec3d& v)
{
    Validate(v[0]);
    Validate(v[1]);
    Validate(v[2]);
}

// ------------------------ MAKE LIST FUNCTIONS ------------------------

void MakeList_Random(std::vector<size_t>& values, size_t count)
{
    std::uniform_int_distribution<size_t> dist(0, c_maxValue);

    static std::random_device rd("dev/random");
    static std::seed_seq fullSeed{ rd(), rd(), rd(), rd(), rd(), rd(), rd(), rd() };
    static std::mt19937 rng(fullSeed);

    values.resize(count);
    for (size_t& v : values)
        v = dist(rng);

    std::sort(values.begin(), values.end());
}

void MakeList_Linear(std::vector<size_t>& values, size_t count)
{
    values.resize(count);
    for (size_t index = 0; index < count; ++index)
    {
        float x = float(index) / (count > 1 ? float(count - 1) : 1);
        float y = x;
        y *= c_maxValue;
        values[index] = size_t(y);
    }

    std::sort(values.begin(), values.end());
}

void MakeList_Linear_Outlier(std::vector<size_t>& values, size_t count)
{
    MakeList_Linear(values, count);
    *values.rbegin() = c_maxValue * 100;
}

void MakeList_Quadratic(std::vector<size_t>& values, size_t count)
{
    values.resize(count);
    for (size_t index = 0; index < count; ++index)
    {
        float x = float(index) / (count > 1 ? float(count - 1) : 1);
        float y = x * x;
        y *= c_maxValue;
        values[index] = size_t(y);
    }

    std::sort(values.begin(), values.end());
}

void MakeList_Cubic(std::vector<size_t>& values, size_t count)
{
    values.resize(count);
    for (size_t index = 0; index < count; ++index)
    {
        float x = float(index) / (count > 1 ? float(count-1) : 1);
        float y = x * x * x;
        y *= c_maxValue;
        values[index] = size_t(y);
    }

    std::sort(values.begin(), values.end());
}

void MakeList_Log(std::vector<size_t>& values, size_t count)
{
    values.resize(count);

    float maxValue = log(float(count));

    for (size_t index = 0; index < count; ++index)
    {
        float x = float(index + 1);
        float y = log1p(x) / maxValue;
        y *= c_maxValue;
        values[index] = size_t(y);
    }

    std::sort(values.begin(), values.end());
}

void MakeList_Normal(std::vector<size_t>& values, size_t count)
{
    std::normal_distribution<> dist{ c_maxValue / 2.0f, c_maxValue / 8.0f };

    static std::random_device rd("dev/random");
    static std::seed_seq fullSeed{ rd(), rd(), rd(), rd(), rd(), rd(), rd(), rd() };
    static std::mt19937 rng(fullSeed);

    values.resize(count);
    for (size_t& v : values)
        v = size_t(Clamp(0.0, double(c_maxValue), dist(rng)));

    std::sort(values.begin(), values.end());
}

// ------------------------ TEST LIST FUNCTIONS ------------------------

TestResults TestList_LinearSearch(const std::vector<size_t>& values, size_t searchValue)
{
    TestResults ret;
    ret.found = false;
    ret.guesses = 0;
    ret.index = 0;

    while (1)
    {
        if (ret.index >= values.size())
            break;
        ret.guesses++;

        size_t value = values[ret.index];
        if (value == searchValue)
        {
            ret.found = true;
            break;
        }
        if (value > searchValue)
            break;

        ret.index++;
    }

    return ret;
}

TestResults TestList_LineFit(const std::vector<size_t>& values, size_t searchValue)
{
    // The idea of this test is that we keep a fit of a line y=mx+b
    // of the left and right side known data points, and use that
    // info to make a guess as to where the value will be.
    //
    // When a guess is wrong, it becomes the new left or right of the line
    // depending on if it was too low (left) or too high (right).
    //
    // This function returns how many steps it took to find the value
    // but doesn't include the min and max reads at the beginning because
    // those could reasonably be done in advance.

    // get the starting min and max value.
    size_t minIndex = 0;
    size_t maxIndex = values.size() - 1;
    size_t min = values[minIndex];
    size_t max = values[maxIndex];

    TestResults ret;
    ret.found = true;
    ret.guesses = 0;

    // if we've already found the value, we are done
    if (searchValue < min)
    {
        ret.index = minIndex;
        ret.found = false;
        return ret;
    }
    if (searchValue > max)
    {
        ret.index = maxIndex;
        ret.found = false;
        return ret;
    }
    if (searchValue == min)
    {
        ret.index = minIndex;
        return ret;
    }
    if (searchValue == max)
    {
        ret.index = maxIndex;
        return ret;
    }

    // fit a line to the end points
    // y = mx + b
    // m = rise / run
    // b = y - mx
    float m = (float(max) - float(min)) / float(maxIndex - minIndex);
    float b = float(min) - m * float(minIndex);

    while (1)
    {
        // make a guess based on our line fit
        ret.guesses++;
        size_t guessIndex = size_t(0.5f + (float(searchValue) - b) / m);
        guessIndex = Clamp(minIndex + 1, maxIndex - 1, guessIndex);
        size_t guess = values[guessIndex];

        // if we found it, return success
        if (guess == searchValue)
        {
            ret.index = guessIndex;
            return ret;
        }

        // if we were too low, this is our new minimum
        if (guess < searchValue)
        {
            minIndex = guessIndex;
            min = guess;
        }
        // else we were too high, this is our new maximum
        else
        {
            maxIndex = guessIndex;
            max = guess;
        }

        // if we run out of places to look, we didn't find it
        if (minIndex + 1 >= maxIndex)
        {
            ret.index = minIndex;
            ret.found = false;
            return ret;
        }

        // fit a new line
        m = (float(max) - float(min)) / float(maxIndex - minIndex);
        b = float(min) - m * float(minIndex);
    }

    return ret;
}

TestResults TestList_LineFitHybridSearch(const std::vector<size_t>& values, size_t searchValue)
{
    // On even iterations, this does a line fit step.
    // On odd iterations, this does a binary search step.
    // Line fit can do better than binary search, but it can also get trapped in situations that it does poorly.
    // The binary search step is there to help it break out of those situations.

    // get the starting min and max value.
    size_t minIndex = 0;
    size_t maxIndex = values.size() - 1;
    size_t min = values[minIndex];
    size_t max = values[maxIndex];

    TestResults ret;
    ret.found = true;
    ret.guesses = 0;

    // if we've already found the value, we are done
    if (searchValue < min)
    {
        ret.index = minIndex;
        ret.found = false;
        return ret;
    }
    if (searchValue > max)
    {
        ret.index = maxIndex;
        ret.found = false;
        return ret;
    }
    if (searchValue == min)
    {
        ret.index = minIndex;
        return ret;
    }
    if (searchValue == max)
    {
        ret.index = maxIndex;
        return ret;
    }

    // fit a line to the end points
    // y = mx + b
    // m = rise / run
    // b = y - mx
    float m = (float(max) - float(min)) / float(maxIndex - minIndex);
    float b = float(min) - m * float(minIndex);

    bool doBinaryStep = false;
    while (1)
    {
        // make a guess based on our line fit, or by binary search, depending on the value of doBinaryStep
        ret.guesses++;
        size_t guessIndex = doBinaryStep ? (minIndex + maxIndex) / 2 : size_t(0.5f + (float(searchValue) - b) / m);
        guessIndex = Clamp(minIndex + 1, maxIndex - 1, guessIndex);
        size_t guess = values[guessIndex];

        // if we found it, return success
        if (guess == searchValue)
        {
            ret.index = guessIndex;
            return ret;
        }

        // if we were too low, this is our new minimum
        if (guess < searchValue)
        {
            minIndex = guessIndex;
            min = guess;
        }
        // else we were too high, this is our new maximum
        else
        {
            maxIndex = guessIndex;
            max = guess;
        }

        // if we run out of places to look, we didn't find it
        if (minIndex + 1 >= maxIndex)
        {
            ret.index = minIndex;
            ret.found = false;
            return ret;
        }

        // fit a new line
        m = (float(max) - float(min)) / float(maxIndex - minIndex);
        b = float(min) - m * float(minIndex);

        // toggle what search mode we are using
        doBinaryStep = !doBinaryStep;
    }

    return ret;
}

float EvaluateQuadratic(const Vec3& coefficients, float x)
{
    return coefficients[0] * x * x + coefficients[1] * x + coefficients[2];
}

float CalculateMeanSquaredError(const Vec2u data[3], const Vec3& coefficients)
{
    float mse = 0.0f;
    for (int i = 0; i < 3; ++i)
    {
        float error = EvaluateQuadratic(coefficients, float(data[i][0])) - float(data[i][1]);
        mse = Lerp(mse, error*error, 1.0f / float(i + 1));
    }
    return mse;
}

// TODO: only for debugging
Vec3 CalculateErrorSquaredGradientNumeric(const Vec2u data[3], const Vec3& coefficients)
{
    Vec3 errorGradient = { 0.0f, 0.0f, 0.0f };
    static const float c_epsilon = 0.001f;
    for (int i = 0; i < 3; ++i)
    {
        Vec3 c1 = coefficients;
        Vec3 c2 = coefficients;
        c1[i] -= c_epsilon;
        c2[i] += c_epsilon;
        errorGradient[i] = (CalculateMeanSquaredError(data, c2) - CalculateMeanSquaredError(data, c1)) / (c_epsilon * 2.0f);
    }
    return errorGradient;
}

Vec3 CalculateErrorSquaredGradient(const Vec2u data[3], const Vec3& coefficients)
{
#if 1  // Calculate error squared gradient analytically

    // Function = Ax^2+Bx+C
    //
    // Error = Ax^2+Bx+C-y
    //
    // Error^2 = (Ax^2+Bx+C-y)^2
    //
    // A,B,C are the coefficients.
    // x is the data point's x axis value
    // y is the data point's y axis value
    //
    // y is a known constant so you can include it in the C constant to get this:
    //
    // Error^2 = (Ax^2+Bx+C)^2
    //
    // We are calculating / minimizing "Error Squared" so that we treat negative
    // and positive error the same, and try to reach zero error.
    //
    // dError^2 / dC = 2Ax^2 + 2Bx   + 2C
    // dError^2 / dB = 2Ax^3 + 2Bx^2 + 2Cx
    // dError^2 / dA = 2Ax^4 + 2Bx^3 + 2Cx^2
    //
    // Observation: start with dEerror^2 / dC
    //              multiply by x to get dEerror^2 / dB
    //              multiply by x to get dEerror^2 / dA
    //

    // how much the error squared for a single point changes as C changes
    auto dErrorSquared_dC_SinglePoint = [](const Vec3& coefficients, const Vec2u& point) -> double
    {
        double x = double(point[0]);
        double y = double(point[1]);

        double A = coefficients[0];
        double B = coefficients[1];
        double C = coefficients[2] - y;

        // 2Ax^2 + 2Bx + 2C
        return 2.0*A*x*x +
               2.0*B*x +
               2.0*C;
    };

    // calculate the combined error squared gradient for all three points
    Vec3d errorGradient = { 0.0, 0.0, 0.0 };
    for (int i = 0; i < 3; ++i)
    {
        double dErrorSquared_dC = dErrorSquared_dC_SinglePoint(coefficients, data[i]);
        double x = double(data[i][0]);

        Validate(dErrorSquared_dC);

        errorGradient[2] += dErrorSquared_dC;
        errorGradient[1] += dErrorSquared_dC * x;
        errorGradient[0] += dErrorSquared_dC * x * x;
        Validate(errorGradient);
    }
    errorGradient[0] /= 3.0;
    errorGradient[1] /= 3.0;
    errorGradient[2] /= 3.0;

    Validate(errorGradient);

    Vec3 ret = Vec3{ float(errorGradient[0]), float(errorGradient[1]) , float(errorGradient[2]) };
    Validate(ret);
    return ret;

#else  // Calculate error squared gradient numerically using central differences.
    Vec3 errorGradient = { 0.0f, 0.0f, 0.0f };
    static const float c_epsilon = 0.001f;
    for (int i = 0; i < 3; ++i)
    {
        Vec3 c1 = coefficients;
        Vec3 c2 = coefficients;
        c1[i] -= c_epsilon;
        c2[i] += c_epsilon;
        errorGradient[i] = (CalculateMeanSquaredError(data, c2) - CalculateMeanSquaredError(data, c1)) / (c_epsilon * 2.0f);
    }
    return errorGradient;
#endif
}

void MakeQuadraticMonotonic_ProjectiveGradientDescent_ProjectValid(const Vec2u data[3], Vec3& coefficients)
{
    // TODO: maybe this is the problem?
    return;

    float deriveMin = 2.0f * coefficients[0] * float(data[0][0]) + coefficients[1];
    float deriveMax = 2.0f * coefficients[0] * float(data[2][0]) + coefficients[1];

    // This function makes sure that the derivatives of the quadratic curve is non negative over the range the data is defined in
    //
    // f(x) = Ax^2 + Bx + C
    // f'(x) = 2Ax+B
    //
    // if x in [a,b] then we want f'(a) >= 0 and f'(b) >= 0
    // assumes a > b!
    //
    // 2Aa >= -B
    // 2Ab >= -B
    //
    // Strategy: test each. Figure out how much adjustment needs to happen on each side of the equation to fix it. Apply
    // half of the adjustment on each side.
    //
    // if A is greater than or equal to zero, that means increasing the derivative at a can't decrease the derivative at b.
    // So, we test/adjust the derivative at a, and then the derivative at b.
    // Else, we test/adjust the derivative at b, and then the derivative at a.
    size_t firstTestIndex = (coefficients[0] >= 0.0f) ? 0 : 2;
    size_t secondTestIndex = (coefficients[0] >= 0.0f) ? 2 : 0;

    // first test
    float derivativeFirst = 2.0f * coefficients[0] * float(data[firstTestIndex][0]) + coefficients[1];
    if (derivativeFirst < 0.0f)
    {
        // split the derivative adjustment across the two terms in Ax + B if we can
        if (data[firstTestIndex][0] > 0)
        {
            float adjustAmount = (-derivativeFirst) / 2.0f;
            coefficients[0] += adjustAmount / (2.0f * float(data[firstTestIndex][0]));

            // do a set, instead of the addition below, due to numerical precision issues
            //coefficients[1] += adjustAmount;
            coefficients[1] = -2.0f * coefficients[0] * float(data[firstTestIndex][0]);

            if (2.0f * coefficients[0] * float(data[firstTestIndex][0]) + coefficients[1] < 0.0f)
                int ijkl = 0;
        }
        // else, x is zero so we can only adjust the B term
        else
        {
            coefficients[1] = -2.0f * coefficients[0] * float(data[firstTestIndex][0]);

            if (2.0f * coefficients[0] * float(data[firstTestIndex][0]) + coefficients[1] < 0.0f)
                int ijkl = 0;
        }
    }

    // TODO give treatment to the second test like we did for the first, for setting the coefficient directly

    // second test
    float derivativeSecond = 2.0f * coefficients[0] * float(data[secondTestIndex][0]) + coefficients[1];
    if (derivativeSecond < 0.0f)
    {
        // split the derivative adjustment across the two terms in Ax + B if we can
        if (data[secondTestIndex][0] > 0)
        {
            float adjustAmount = (-derivativeSecond) / 2.0f;
            coefficients[0] += adjustAmount / (2.0f * float(data[secondTestIndex][0]));
            coefficients[1] += adjustAmount;

            if (2.0f * coefficients[0] * float(data[secondTestIndex][0]) + coefficients[1] < 0.0f)
                int ijkl = 0;
        }
        // else, x is zero so we can only adjust the B term
        else
        {
            coefficients[1] += -derivativeSecond;

            if (2.0f * coefficients[0] * float(data[secondTestIndex][0]) + coefficients[1] < 0.0f)
                int ijkl = 0;
        }
    }

    // TODO: there is some sort of problem here, dig into it.  maybe could store off initial values and move it back up to the top and step through it again?

    // make sure we were successful
    float derivativeStart = 2.0f * coefficients[0] * float(data[0][0]) + coefficients[1];
    float derivativeEnd = 2.0f * coefficients[0] * float(data[2][0]) + coefficients[1];
    if (derivativeStart < 0.0f || derivativeEnd < 0.0f)
    {
        printf("ERROR! MakeMonotonicSingleStep() failed!!\n");
    }
}

void MakeQuadraticMonotonic_ProjectiveGradientDescent(const Vec2u data[3], Vec3& coefficients)
{
    // This function searches for the best fit of a quadratic function to the data set, where
    // the curve is monotonic over the range that the data is defined in.
    //
    // It uses projective gradient descent to find this, which means it makes the data valid
    // after each gradient descent step (projects it back into valid space).

    static const int c_numIterations = 100;
    static const float c_stepSize = 0.01f;

    Validate(coefficients);

    float val0 = EvaluateQuadratic(coefficients, float(data[0][0]));
    float val1 = EvaluateQuadratic(coefficients, float(data[1][0]));
    float val2 = EvaluateQuadratic(coefficients, float(data[2][0]));

    MakeQuadraticMonotonic_ProjectiveGradientDescent_ProjectValid(data, coefficients);

    Validate(coefficients);

    // TODO: once you fix the growing error problem, maybe keep the lowest error found.

    // TODO: remove when things are sorted
    float startingMSE = CalculateMeanSquaredError(data, coefficients);
    Validate(startingMSE);

    //printf("\n\n--------------------\nStarting Error = %f\n", CalculateMeanSquaredError(data, coefficients));
    //printf("data = (%zu,%zu), (%zu,%zu), (%zu,%zu)\n", data[0][0], data[0][1], data[1][0], data[1][1], data[2][0], data[2][1]);
    //printf("coefficients = (%f, %f, %f)\n", coefficients[0], coefficients[1], coefficients[2]);

    // TODO: momentum! https://distill.pub/2017/momentum/

    static const float c_momentumAlpha = 0.001f;
    static const float c_momentumBeta = 0.99f;

    Vec3 errorGradientWithMomentum = Vec3{ 0.0f, 0.0f, 0.0f };

    for (int i = 0; i < c_numIterations; ++i)
    {
        // TODO: remove when done
        float MSEBefore = CalculateMeanSquaredError(data, coefficients);
        Validate(MSEBefore);

        Vec3 oldCoefficients = coefficients;
        static bool doThis = false;
        if (doThis)
        {
            coefficients = oldCoefficients;
        }

        Vec3 errorGradient = CalculateErrorSquaredGradient(data, coefficients);
        Vec3 errorGradientNumeric = CalculateErrorSquaredGradientNumeric(data, coefficients);

        // momentum
        for (int i = 0; i < 3; ++i)
            errorGradientWithMomentum[i] = c_momentumBeta * errorGradientWithMomentum[i] + errorGradient[i];

        const float c_lineBackTrackC = 0.5f;
        const float c_lineBackTrackTau = 0.5f;

        // line backtracking
        // TODO: can factor some stuff out of the loop when it's working
        Vec3 proposedCoefficients;
        float t = 1.0f;
        while (1)
        {           
            for (int i = 0; i < 3; ++i)
                proposedCoefficients[i] = coefficients[i] - c_momentumAlpha * t * errorGradientWithMomentum[i];

            float errorStepped = CalculateMeanSquaredError(data, proposedCoefficients);
            float errorCurrent = CalculateMeanSquaredError(data, coefficients);


            float errorGrad_dot_errorGrad = 0.0f;
            for (int i = 0; i < 3; ++i)
                errorGrad_dot_errorGrad += errorGradient[i] * errorGradient[i];

            float rhs = (c_momentumAlpha * t) * c_lineBackTrackC * errorGrad_dot_errorGrad;

            if (errorStepped <= errorCurrent - rhs)
                break;

            t = t * c_lineBackTrackTau;
        }

        // TODO: use proposed coefficients to descend!
        coefficients = proposedCoefficients;

        // descend
        //for (int i = 0; i < 3; ++i)
            //coefficients[i] -= c_momentumAlpha * t * errorGradientWithMomentum[i];

        //coefficients[0] -= errorGradient[0] * c_stepSize;
        //coefficients[1] -= errorGradient[1] * c_stepSize;
        //coefficients[2] -= errorGradient[2] * c_stepSize;

        float MSEAfter = CalculateMeanSquaredError(data, coefficients);
        Validate(MSEAfter);

        if (MSEAfter > MSEBefore * 2 && MSEBefore > 1.0f)
            int ijkl = 0;

        Validate(coefficients);

        MakeQuadraticMonotonic_ProjectiveGradientDescent_ProjectValid(data, coefficients);

        Validate(coefficients);

        float gradientLength = sqrtf(errorGradient[0] * errorGradient[0] + errorGradient[1] * errorGradient[1] + errorGradient[2] * errorGradient[2]);
        float gradientWithMomentumLength = sqrtf(errorGradientWithMomentum[0] * errorGradientWithMomentum[0] + errorGradientWithMomentum[1] * errorGradientWithMomentum[1] + errorGradientWithMomentum[2] * errorGradientWithMomentum[2]);

        //printf("\n[%i] Error = %f (grad length = %f / %f)\n", i, CalculateMeanSquaredError(data, coefficients), gradientLength, gradientWithMomentumLength);
        //printf("Grad = %f, %f, %f\n", errorGradient[0], errorGradient[1], errorGradient[2]);
        //printf("GradNum = %f, %f, %f\n", errorGradientNumeric[0], errorGradientNumeric[1], errorGradientNumeric[2]);
        //printf("Grad M = %f, %f, %f\n", errorGradientWithMomentum[0], errorGradientWithMomentum[1], errorGradientWithMomentum[2]);
        //printf("coeff = %f, %f, %f\n", coefficients[0], coefficients[1], coefficients[2]);
    }

    //printf("Ending Error = %f\n", CalculateMeanSquaredError(data, coefficients));
}

void QuadraticFit(const Vec2u data[3], Vec3& coefficients)
{
    // This function calculates the terms for a quadratic function passing through the points
    // passed in. Derived from Lagrange interpolation.
    coefficients = { 0.0f, 0.0f, 0.0f };

    for (size_t index = 0; index < 3; ++index)
    {
        size_t index0 = index;
        size_t index1 = (index == 0) ? 1 : 0;
        size_t index2 = (index == 2) ? 1 : 2;

        // "A" coefficient is 1, always
        float termA = 1.0f;

        // "B" coefficient is -index1 + -index2
        float termB = -float(data[index1][0]) + -float(data[index2][0]);

        // "C" coefficient is -index1 * -index2
        float termC = -float(data[index1][0]) * -float(data[index2][0]);

        // denominator is (i - index1) * (i - index2)
        float denominator = (float(data[index0][0]) - float(data[index1][0])) * (float(data[index0][0]) - float(data[index2][0]));

        // add these terms into the overall A,B,C terms of our quadratic function
        // terms get multiplied by values[i] and divided by the denominator
        coefficients[0] += termA * data[index0][1] / denominator;
        coefficients[1] += termB * data[index0][1] / denominator;
        coefficients[2] += termC * data[index0][1] / denominator;
    }
}

TestResults TestList_QuadraticFit(const std::vector<size_t>& values, size_t searchValue)
{
    // The idea of this test is that we keep a fit of a quadratic y=Ax^2+Bx+C and use
    // that info to make a guess as to where the value will be.
    //
    // When a guess is wrong, it becomes the new search min or max depending on if it was
    // too low (left) or too high (right).  The "third point" in the quadratic fit is
    // whichever point is closer to the min/max boundaries:  the old min or max that just
    // got replaced, or the old "third point".  We want to keep the data fit as localized
    // as we can for better results.
    //
    // This function returns how many steps it took to find the value
    // but doesn't include the reads at the beginning because those and the monotonic
    // quadratic fit should be done in advance and shared among all queries against it.

    // TODO: maybe the initial quadratic fit should minimize the error of all data points.
    // Since it's stored and re-used it shouldn't matter if it takes a bit to do that.  We
    // probably should literally store it off and re-use it in fact.

    // get the starting min and max value.
    int minIndex = 0;
    int maxIndex = int(values.size() - 1);
    int midIndex = Clamp(minIndex + 1, maxIndex - 1, (minIndex + maxIndex) / 2);
    if (minIndex == maxIndex)
        midIndex = minIndex;
    size_t min = values[minIndex];
    size_t max = values[maxIndex];
    size_t mid = values[midIndex];

    TestResults ret;
    ret.found = true;
    ret.guesses = 0;

    // if we've already found the value, we are done
    if (searchValue < min)
    {
        ret.index = minIndex;
        ret.found = false;
        return ret;
    }
    if (searchValue > max)
    {
        ret.index = maxIndex;
        ret.found = false;
        return ret;
    }
    if (searchValue == min)
    {
        ret.index = minIndex;
        return ret;
    }
    if (searchValue == max)
    {
        ret.index = maxIndex;
        return ret;
    }
    if (searchValue == mid)
    {
        ret.index = midIndex;
        return ret;
    }

    // if 3 or less items in the list, we are done
    if (values.size() <= 3)
    {
        // TODO: return the index to insert it into.
        ret.index = 0;
        ret.found = false;
        return ret;
    }

    if (values.size() == 10)
    {
        int ijkl = 0;
    }

    Vec3 coefficiants;
    {
        Vec2u data[3] =
        {
            {size_t(minIndex), min},
            {size_t(midIndex), mid},
            {size_t(maxIndex), max}
        };
        QuadraticFit(data, coefficiants);
        MakeQuadraticMonotonic_ProjectiveGradientDescent(data, coefficiants);
    }

    while (1)
    {
        Validate(coefficiants);

        // make a guess, by using the quadratic equation to plug in y and get an x
        ret.guesses++;

        // calculate discriminant: B^2-4AC.
        // Note: C = (C - y)
        float discriminant = coefficiants[1] * coefficiants[1] - 4.0f * coefficiants[0] * (coefficiants[2] - float(searchValue));

        // (-b +/- sqrt(discriminant)) / 2A
        float guess1f = (-coefficiants[1] + sqrtf(discriminant)) / (2.0f * coefficiants[0]);
        float guess2f = (-coefficiants[1] - sqrtf(discriminant)) / (2.0f * coefficiants[0]);

        int guess1 = size_t(guess1f);
        int guess2 = size_t(guess2f);

        int guessIndex = (guess1 >= minIndex && guess1 <= maxIndex) ? guess1 : guess2;

        // TODO: re-enable, or delete this
        //if (guessIndex < minIndex || guessIndex > maxIndex)
            //printf("No Guess Valid?!");

        guessIndex = Clamp(minIndex + 1, maxIndex - 1, guessIndex);
        size_t guess = values[guessIndex];

        // if we found it, return success
        if (guess == searchValue)
        {
            ret.index = guessIndex;
            return ret;
        }

        // if we were too low, this is our new minimum
        if (guess < searchValue)
        {
            int oldMinIndex = minIndex;
            size_t oldMin = min;
            minIndex = guessIndex;
            min = guess;

            // for our "third point", we should use the old min, if it's less far out of bounds than the current "third point"
            int midIndexDistance = 0;
            if (midIndex < minIndex)
                midIndexDistance = minIndex - midIndex;
            else if (midIndex > maxIndex)
                midIndexDistance = midIndex - maxIndex;

            int oldMinIndexDistance = minIndex - oldMinIndex;

            if (oldMinIndexDistance < midIndexDistance || minIndex == midIndex)
            {
                midIndex = oldMinIndex;
                mid = oldMin;

                if (midIndex == minIndex || midIndex == maxIndex)
                    int ijkl = 0;
            }
            else
            {
                if (midIndex == minIndex || midIndex == maxIndex)
                    int ijkl = 0;
            }
        }
        // else we were too high, this is our new maximum
        else
        {
            int oldMaxIndex = maxIndex;
            size_t oldMax = max;
            maxIndex = guessIndex;
            max = guess;

            // for our "third point", we should use the old max, if it's less far out of bounds than the current "third point"
            int midIndexDistance = 0;
            if (midIndex < minIndex)
                midIndexDistance = minIndex - midIndex;
            else if (midIndex > maxIndex)
                midIndexDistance = midIndex - maxIndex;

            int oldMaxIndexDistance = maxIndex - oldMaxIndex;

            if (oldMaxIndexDistance < midIndexDistance || maxIndex == midIndex)
            {
                midIndex = oldMaxIndex;
                mid = oldMax;

                if (midIndex == minIndex || midIndex == maxIndex)
                    int ijkl = 0;
            }
            else
            {
                if (midIndex == minIndex || midIndex == maxIndex)
                    int ijkl = 0;
            }
        }

        // TODO: i think we exit out when the remaining list size is <= 3?

        // if we run out of places to look, we didn't find it
        if (minIndex + 1 >= maxIndex)
        {
            ret.index = minIndex;
            ret.found = false;
            return ret;
        }

        // do another quadratic fit!
        // We need a sorted list of minIndex, midIndex, maxIndex.
        // We know minIndex < MaxIndex, but we don't know where midIndex fits.
        Vec2u data[3];
        Vec2u minv = { size_t(minIndex), min };
        Vec2u midv = { size_t(midIndex), mid };
        Vec2u maxv = { size_t(maxIndex), max };

        if (midIndex <= minIndex)
        {
            data[0] = midv;
            data[1] = minv;
            data[2] = maxv;
        }
        else if (midIndex <= minIndex)
        {
            data[0] = minv;
            data[1] = midv;
            data[2] = maxv;
        }
        else
        {
            data[0] = minv;
            data[1] = maxv;
            data[2] = midv;
        }

        // TODO: need to make sure that mid index is never the same as min/max index. the quadratic fit breaks down then.
        // maybe need to make it start the function by checking the "third" point (mid), so we know that the search value is never at mid. can exit when 3 locations left.

        QuadraticFit(data, coefficiants);
        Validate(coefficiants);
        MakeQuadraticMonotonic_ProjectiveGradientDescent(data, coefficiants);
        Validate(coefficiants);

        // TODO: rename mid to "third" since it might not be in the middle
        // TODO: what does the discriminant and guesses look like for non monotonic case? I'm betting the guesses are ambiguous.
        // TODO: need to thoroughly test this!
    }

    return ret;
}

// TODO: should this go in another section?
void InitialFit_QuadraticFitNonMonotonic(TSheet& csv, const std::vector<size_t>& values)
{
    int minIndex = 0;
    int maxIndex = int(values.size() - 1);
    int midIndex = Clamp(minIndex + 1, maxIndex - 1, (minIndex + maxIndex) / 2);
    size_t min = values[minIndex];
    size_t max = values[maxIndex];
    size_t mid = values[midIndex];

    Vec3 coefficiants;
    {
        Vec2u data[3] =
        {
            {size_t(minIndex), min},
            {size_t(midIndex), mid},
            {size_t(maxIndex), max}
        };
        QuadraticFit(data, coefficiants);
    }

    char buffer[256];
    for (size_t i = 0; i < values.size(); ++i)
    {
        sprintf_s(buffer, "%i", int(EvaluateQuadratic(coefficiants, float(i))));
        csv[i+1].push_back(buffer);
    }
}

TestResults TestList_QuadraticFitNonMonotonic(const std::vector<size_t>& values, size_t searchValue)
{
    // The idea of this test is that we keep a fit of a quadratic y=Ax^2+Bx+C and use
    // that info to make a guess as to where the value will be.
    //
    // When a guess is wrong, it becomes the new search min or max depending on if it was
    // too low (left) or too high (right).  The "third point" in the quadratic fit is
    // whichever point is closer to the min/max boundaries:  the old min or max that just
    // got replaced, or the old "third point".  We want to keep the data fit as localized
    // as we can for better results.
    //
    // This function returns how many steps it took to find the value
    // but doesn't include the reads at the beginning because those and the monotonic
    // quadratic fit should be done in advance and shared among all queries against it.

    // TODO: maybe the initial quadratic fit should minimize the error of all data points.
    // Since it's stored and re-used it shouldn't matter if it takes a bit to do that.  We
    // probably should literally store it off and re-use it in fact.

    // get the starting min and max value.
    int minIndex = 0;
    int maxIndex = int(values.size() - 1);
    int midIndex = Clamp(minIndex + 1, maxIndex - 1, (minIndex + maxIndex) / 2);
    if (minIndex == maxIndex)
        midIndex = minIndex;
    size_t min = values[minIndex];
    size_t max = values[maxIndex];
    size_t mid = values[midIndex];

    TestResults ret;
    ret.found = true;
    ret.guesses = 0;

    // if we've already found the value, we are done
    if (searchValue < min)
    {
        ret.index = minIndex;
        ret.found = false;
        return ret;
    }
    if (searchValue > max)
    {
        ret.index = maxIndex;
        ret.found = false;
        return ret;
    }
    if (searchValue == min)
    {
        ret.index = minIndex;
        return ret;
    }
    if (searchValue == max)
    {
        ret.index = maxIndex;
        return ret;
    }
    if (searchValue == mid)
    {
        ret.index = midIndex;
        return ret;
    }

    // if 3 or less items in the list, we are done
    if (values.size() <= 3)
    {
        // TODO: return the index to insert it into.
        ret.index = 0;
        ret.found = false;
        return ret;
    }

    Vec3 coefficiants;
    {
        Vec2u data[3] =
        {
            {size_t(minIndex), min},
            {size_t(midIndex), mid},
            {size_t(maxIndex), max}
        };
        QuadraticFit(data, coefficiants);
    }

    while (1)
    {
        Validate(coefficiants);

        // make a guess, by using the quadratic equation to plug in y and get an x
        ret.guesses++;

        // calculate discriminant: B^2-4AC.
        // Note: C = (C - y)
        float discriminant = coefficiants[1] * coefficiants[1] - 4.0f * coefficiants[0] * (coefficiants[2] - float(searchValue));

        // (-b +/- sqrt(discriminant)) / 2A
        float guess1f = (-coefficiants[1] + sqrtf(discriminant)) / (2.0f * coefficiants[0]);
        float guess2f = (-coefficiants[1] - sqrtf(discriminant)) / (2.0f * coefficiants[0]);

        int guess1 = size_t(guess1f);
        int guess2 = size_t(guess2f);

        int guessIndex = (guess1 >= minIndex && guess1 <= maxIndex) ? guess1 : guess2;

        // TODO: re-enable, or delete this
        //if (guessIndex < minIndex || guessIndex > maxIndex)
            //printf("No Guess Valid?!");

        guessIndex = Clamp(minIndex + 1, maxIndex - 1, guessIndex);
        size_t guess = values[guessIndex];

        // if we found it, return success
        if (guess == searchValue)
        {
            ret.index = guessIndex;
            return ret;
        }

        // if we were too low, this is our new minimum
        if (guess < searchValue)
        {
            int oldMinIndex = minIndex;
            size_t oldMin = min;
            minIndex = guessIndex;
            min = guess;

            // for our "third point", we should use the old min, if it's less far out of bounds than the current "third point"
            int midIndexDistance = 0;
            if (midIndex < minIndex)
                midIndexDistance = minIndex - midIndex;
            else if (midIndex > maxIndex)
                midIndexDistance = midIndex - maxIndex;

            int oldMinIndexDistance = minIndex - oldMinIndex;

            if (oldMinIndexDistance < midIndexDistance || minIndex == midIndex)
            {
                midIndex = oldMinIndex;
                mid = oldMin;

                if (midIndex == minIndex || midIndex == maxIndex)
                    int ijkl = 0;
            }
            else
            {
                if (midIndex == minIndex || midIndex == maxIndex)
                    int ijkl = 0;
            }
        }
        // else we were too high, this is our new maximum
        else
        {
            int oldMaxIndex = maxIndex;
            size_t oldMax = max;
            maxIndex = guessIndex;
            max = guess;

            // for our "third point", we should use the old max, if it's less far out of bounds than the current "third point"
            int midIndexDistance = 0;
            if (midIndex < minIndex)
                midIndexDistance = minIndex - midIndex;
            else if (midIndex > maxIndex)
                midIndexDistance = midIndex - maxIndex;

            int oldMaxIndexDistance = maxIndex - oldMaxIndex;

            if (oldMaxIndexDistance < midIndexDistance || maxIndex == midIndex)
            {
                midIndex = oldMaxIndex;
                mid = oldMax;

                if (midIndex == minIndex || midIndex == maxIndex)
                    int ijkl = 0;
            }
            else
            {
                if (midIndex == minIndex || midIndex == maxIndex)
                    int ijkl = 0;
            }
        }

        // TODO: i think we exit out when the remaining list size is <= 3?

        // if we run out of places to look, we didn't find it
        if (minIndex + 1 >= maxIndex)
        {
            ret.index = minIndex;
            ret.found = false;
            return ret;
        }

        // do another quadratic fit!
        // We need a sorted list of minIndex, midIndex, maxIndex.
        // We know minIndex < MaxIndex, but we don't know where midIndex fits.
        Vec2u data[3];
        Vec2u minv = { size_t(minIndex), min };
        Vec2u midv = { size_t(midIndex), mid };
        Vec2u maxv = { size_t(maxIndex), max };

        if (midIndex <= minIndex)
        {
            data[0] = midv;
            data[1] = minv;
            data[2] = maxv;
        }
        else if (midIndex <= minIndex)
        {
            data[0] = minv;
            data[1] = midv;
            data[2] = maxv;
        }
        else
        {
            data[0] = minv;
            data[1] = maxv;
            data[2] = midv;
        }

        // TODO: need to make sure that mid index is never the same as min/max index. the quadratic fit breaks down then.
        // maybe need to make it start the function by checking the "third" point (mid), so we know that the search value is never at mid. can exit when 3 locations left.

        QuadraticFit(data, coefficiants);
        Validate(coefficiants);

        // TODO: rename mid to "third" since it might not be in the middle
        // TODO: what does the discriminant and guesses look like for non monotonic case? I'm betting the guesses are ambiguous.
        // TODO: need to thoroughly test this!
    }

    // TODO: make this function appropriate for non monotonic fitting.
    // TODO: since there can be 2 ambigious values, try switching which is used

    return ret;
}

TestResults TestList_BinarySearch(const std::vector<size_t>& values, size_t searchValue)
{
    TestResults ret;
    ret.found = false;
    ret.guesses = 0;

    size_t minIndex = 0;
    size_t maxIndex = values.size() - 1;
    while (1)
    {
        // make a guess by looking in the middle of the unknown area
        ret.guesses++;
        size_t guessIndex = (minIndex + maxIndex) / 2;
        size_t guess = values[guessIndex];

        // found it
        if (guess == searchValue)
        {
            ret.found = true;
            ret.index = guessIndex;
            return ret;
        }
        // if our guess was too low, it's the new min
        else if (guess < searchValue)
        {
            minIndex = guessIndex + 1;
        }
        // if our guess was too high, it's the new max
        else if (guess > searchValue)
        {
            // underflow prevention
            if (guessIndex == 0)
            {
                ret.index = guessIndex;
                return ret;
            }
            maxIndex = guessIndex - 1;
        }

        // fail case
        if (minIndex > maxIndex)
        {
            ret.index = guessIndex;
            return ret;
        }
    }

    return ret;
}

struct Point
{
    Point(float xin, float yin) : x(xin), y(yin) {}
    Point(size_t xin, size_t yin) : x(float(xin)), y(float(yin)) {}
    float x;
    float y;
};

struct LinearEquation
{
    float m;
    float b;
};

bool GetLinearEqn(Point a, Point b, LinearEquation& result)
{
    if (a.x > b.x)
        std::swap(a,b);

    if (b.x - a.x == 0)
        return false;
    result.m = (b.y - a.y) / (b.x - a.x);
    result.b = a.y - result.m * a.x;

    return true;
}

// TODO: should this go elsewhere?
// TODO: don't need numvalues and valid i don't think.
thread_local float g_TestList_Gradient_A = 0.0f;
thread_local float g_TestList_Gradient_B = 0.0f;
thread_local float g_TestList_Gradient_C = 0.0f;
thread_local size_t numvalues = 0;
thread_local bool valid = false;
void InitialFit_Gradient(TSheet& csv, const std::vector<size_t>& values)
{
    Vec3 coefficiants = { g_TestList_Gradient_A, g_TestList_Gradient_B, g_TestList_Gradient_C };
    char buffer[512];

    // TODO: temp
    //sprintf_s(buffer, "%f, %f, %f  (%zu), %s", g_TestList_Gradient_A, g_TestList_Gradient_B, g_TestList_Gradient_C, numvalues, valid ? "true" : "false");
    //csv[1].push_back(buffer);

    for (size_t i = 0; i < values.size(); ++i)
    {
        sprintf_s(buffer, "%zu", size_t(EvaluateQuadratic(coefficiants, float(i))));
        csv[i + 1].push_back(buffer);
    }
}

TestResults TestList_Gradient(const std::vector<size_t>& values, size_t searchValue)
{
    // The idea of this test is somewhat similar to that of TestList_LineFit.
    // Instead of assuming that our data fits a linear line between our min
    // and max, we sample around min and max (1 point near each) to get the
    // local gradient. Once we have that, we calculate a linear derivative of
    // the line that approximates the endpoints' locations and the tangent
    // line at each. From there, we propagate up y-intercept points and plug
    // them into the inverse function of our line

    // get the starting min and max value.
    size_t minIndex = 0;
    size_t maxIndex = values.size() - 1;
    size_t min = values[minIndex];
    size_t max = values[maxIndex];

    TestResults ret;
    ret.found = true;
    ret.guesses = 0;

    // if we've already found the value, we are done
    if (searchValue < min)
    {
        ret.index = minIndex;
        ret.found = false;
        return ret;
    }
    if (searchValue > max)
    {
        ret.index = maxIndex;
        ret.found = false;
        return ret;
    }
    if (searchValue == min)
    {
        ret.index = minIndex;
        return ret;
    }
    if (searchValue == max)
    {
        ret.index = maxIndex;
        return ret;
    }

    // Calculate an approximation line
    // Assume y'' = c1
    // y' = xc1 + c2
    // y = x^2/2 * c1 + xc2 + c3
    // 0 = x^2/2 * c1 + xc2 + (c3 - y)
    // x = (-c2 +- sqrt(c2 * c2 - 2 * c1 * (c3 - y))) / c1

    // tan1 = tangent to min, tan2 = tangent to max, prime = y'
    LinearEquation tan1, tan2;
    LinearEquation prime;

    auto updateEquations = [&]() -> bool
    {
        // Update tan1, tan2
        const size_t offset = (maxIndex - minIndex) / 10; // No good reason for choosing 10. There are probably better values
        // const size_t offset = 10; // Can also try an absolute offset 
        if (offset == 0 || offset > (maxIndex - minIndex))
            return false;

        if (!GetLinearEqn({minIndex, values[minIndex]}, {minIndex + offset, values[minIndex + offset]}, tan1))
            return false;

        if (!GetLinearEqn({maxIndex, values[maxIndex]}, {maxIndex - offset, values[maxIndex - offset]}, tan2))
            return false;

        // Update y'
        // prime.m = c1
        // prime.b = c2
        if (!GetLinearEqn({float(minIndex), tan1.m}, {float(maxIndex), tan2.m}, prime))
            return false;

        return true;
    };

    bool first = true;
    valid = false;

    auto getGuess = [&](size_t y, size_t& x) -> bool
    {
        // Solve for c3 using min
        const float c3 = values[minIndex] - minIndex * minIndex / 2.0f * prime.m - minIndex * prime.b;

        // Solve for x.
        // y = x^2/2 * c1 + xc2 + c3
        // 0 = x^2/2 * c1 + xc2 + (c3 - y)
        // x = (-c2 +- sqrt(c2 * c2 - 2 * c1 * (c3 - y))) / c1
        //const float c1 = prime.m;
        const float c2 = prime.b;

        // TODO: make this into an alternate function.
        // c1 = (y - xc2 - x3)/(x^2 / 2)
        const float c1 = (values[maxIndex] - float(maxIndex) * c2 - c3) / float(maxIndex*maxIndex / 2.0f);

        if (first)
        {
            first = false;
            g_TestList_Gradient_A = c1 / 2.0f;
            g_TestList_Gradient_B = c2;
            g_TestList_Gradient_C = c3;
            numvalues = values.size();
        }

        if ( c2 * c2 - 2 * c1 * (c3 - y) < 0.0f )
        {
            return false;
        }

        if ( c1 == 0.0f )
        {
            return false;
        }

        const float x1f = (-c2 + std::sqrt(c2 * c2 - 2 * c1 * (c3 - y))) / c1;
        const float x2f = (-c2 - std::sqrt(c2 * c2 - 2 * c1 * (c3 - y))) / c1;

        const size_t x1 = size_t(x1f + 0.5f);
        const size_t x2 = size_t(x2f + 0.5f);

        const bool valid1 = x1 > minIndex && x1 < maxIndex;
        const bool valid2 = x2 > minIndex && x2 < maxIndex;
        if (!valid1 && !valid2)
        {
            return false;
        }
        else if(valid1 && !valid2)
        {
            x = x1;
            return true;
        }
        else if(!valid1 && valid2)
        {
            x = x2;
            return true;
        }

        // Both x1 and x2 are valid and in range
        // If we're concave up, choose the greater, concave down the lesser
        // This works because we know y' is positive
        if (c1 > 0)
        {
            x = std::max(x1, x2);
            return true;
        }
        else
        {
            x = std::min(x1, x2);
            return true;
        }
    };

    bool validEquations = updateEquations();

    valid = validEquations;

    while (1)
    {
        // make a guess based on our line fit
        ret.guesses++;
        size_t guessIndex;
        if ( !validEquations || !getGuess(searchValue, guessIndex) )
        {
            // Fall back to binary search
            guessIndex = (minIndex + maxIndex) / 2;
        }

        guessIndex = Clamp(minIndex + 1, maxIndex - 1, guessIndex);
        size_t guess = values[guessIndex];

        // if we found it, return success
        if (guess == searchValue)
        {
            ret.index = guessIndex;
            return ret;
        }

        // if we were too low, this is our new minimum
        if (guess < searchValue)
        {
            minIndex = guessIndex;
            min = guess;
        }
        // else we were too high, this is our new maximum
        else
        {
            maxIndex = guessIndex;
            max = guess;
        }

        // if we run out of places to look, we didn't find it
        if (minIndex + 1 >= maxIndex)
        {
            ret.index = minIndex;
            ret.found = false;
            return ret;
        }

        validEquations = updateEquations();
    }

    return ret;
}

// ------------------------ MAIN ------------------------

void VerifyResults(const std::vector<size_t>& values, size_t searchValue, const TestResults& result, const char* list, const char* test)
{
#if VERIFY_RESULT()
    // verify correctness of result by comparing to a linear search
    TestResults actualResult = TestList_LinearSearch(values, searchValue);
    if (result.found != actualResult.found)
    {
        printf("VERIFICATION FAILURE!! (found %s vs %s) %s, %s\n", result.found ? "true" : "false", actualResult.found ? "true" : "false", list, test);
    }
    else if (result.found == true && result.index != actualResult.index && values[result.index] != values[actualResult.index])
    {
        // Note that in the case of duplicates, different algorithms may return different indices, but the values stored in them should be the same
        printf("VERIFICATION FAILURE!! (index %zu vs %zu) %s, %s\n", result.index, actualResult.index, list, test);
    }
    // verify that the index returned is a reasonable place for the value to be inserted, if the value was not found.
    else if (result.found == false)
    {
        bool gte = true;
        bool lte = true;

        if (result.index > 0)
            gte = searchValue >= values[result.index - 1];

        if (result.index + 1 < values.size())
            lte = searchValue <= values[result.index + 1];

        // TODO: re-enable
        //if (gte == false || lte == false)
           // printf("VERIFICATION FAILURE!! Not a valid place to insert a new value! %s, %s\n", list, test);
    }

#endif
}

int main(int argc, char** argv)
{
    MakeListInfo MakeFns[] =
    {
        {"Normal", MakeList_Normal},
        {"Random", MakeList_Random},
        {"Linear", MakeList_Linear},
        {"Linear Outlier", MakeList_Linear_Outlier},
        {"Quadratic", MakeList_Quadratic},
        {"Cubic", MakeList_Cubic},
        {"Log", MakeList_Log},
    };

    TestListInfo TestFns[] =
    {
        {"Linear Search", TestList_LinearSearch, nullptr},
        {"Line Fit", TestList_LineFit, nullptr},

        {"Binary Search", TestList_BinarySearch, nullptr},
        {"Line Fit Hybrid", TestList_LineFitHybridSearch, nullptr},
        {"Quadratic Fit", TestList_QuadraticFit, nullptr}, // TODO: initial fit
        {"Quadratic Fit (Non Monotonic)", TestList_QuadraticFitNonMonotonic, InitialFit_QuadraticFitNonMonotonic},
        {"Gradient", TestList_Gradient, InitialFit_Gradient},

        // TODO: Quadratic Hybrid Fit? if necessary...
    };

#if MAKE_CSVS()

    size_t numThreads = std::thread::hardware_concurrency();
    std::vector<std::thread> threads;
    threads.resize(numThreads);

    // for each numer sequence. Done multithreadedly
    std::atomic<size_t> nextRow(0);
    for (std::thread& t : threads)
    {
        t = std::thread(
            [&]()
            {
                size_t makeIndex = nextRow.fetch_add(1);
                while (makeIndex < countof(MakeFns))
                {
                    printf("Starting %s\n", MakeFns[makeIndex].name);

                    static std::random_device rd("dev/random");
                    static std::seed_seq fullSeed{ rd(), rd(), rd(), rd(), rd(), rd(), rd(), rd() };
                    static std::mt19937 rng(fullSeed);

                    // the data to write to the csv file. a row per sample count plus one more for titles
                    TSheet csv;
                    csv.resize(c_maxNumValues + 1);

                    // make a column for the sample counts
                    char buffer[256];
                    csv[0].push_back("Sample Count");
                    for (size_t numValues = 1; numValues <= c_maxNumValues; ++numValues)
                    {
                        sprintf_s(buffer, "%zu", numValues);
                        csv[numValues].push_back(buffer);
                    }

                    // for each test
                    std::vector<size_t> values;
                    for (size_t testIndex = 0; testIndex < countof(TestFns); ++testIndex)
                    {
                        sprintf_s(buffer, "%s Min", TestFns[testIndex].name);
                        csv[0].push_back(buffer);
                        sprintf_s(buffer, "%s Max", TestFns[testIndex].name);
                        csv[0].push_back(buffer);
                        sprintf_s(buffer, "%s Avg", TestFns[testIndex].name);
                        csv[0].push_back(buffer);
                        sprintf_s(buffer, "%s Single", TestFns[testIndex].name);
                        csv[0].push_back(buffer);

                        // for each result
                        for (size_t numValues = 1; numValues <= c_maxNumValues; ++numValues)
                        {
                            size_t guessMin = ~size_t(0);
                            size_t guessMax = 0;
                            float guessAverage = 0.0f;
                            size_t guessSingle = 0;

                            // repeat it a number of times to gather min, max, average
                            for (size_t repeatIndex = 0; repeatIndex < c_numRunsPerTest; ++repeatIndex)
                            {
                                std::uniform_int_distribution<size_t> dist(0, c_maxValue);
                                size_t searchValue = dist(rng);

                                MakeFns[makeIndex].fn(values, numValues);
                                TestResults result = TestFns[testIndex].fn(values, searchValue);

                                VerifyResults(values, searchValue, result, MakeFns[makeIndex].name, TestFns[testIndex].name);

                                guessMin = std::min(guessMin, result.guesses);
                                guessMax = std::max(guessMax, result.guesses);
                                guessAverage = Lerp(guessAverage, float(result.guesses), 1.0f / float(repeatIndex + 1));
                                guessSingle = result.guesses;
                            }

                            sprintf_s(buffer, "%zu", guessMin);
                            csv[numValues].push_back(buffer);

                            sprintf_s(buffer, "%zu", guessMax);
                            csv[numValues].push_back(buffer);

                            sprintf_s(buffer, "%f", guessAverage);
                            csv[numValues].push_back(buffer);

                            sprintf_s(buffer, "%zu", guessSingle);
                            csv[numValues].push_back(buffer);
                        }
                    }

                    // make a column for the sampling sequence itself
                    csv[0].push_back("Sequence");
                    for (size_t numValues = 1; numValues <= c_maxNumValues; ++numValues)
                    {
                        sprintf_s(buffer, "%zu", values[numValues-1]);
                        csv[numValues].push_back(buffer);
                    }

                    // make a column for each test that wants to show an initial fit
                    for (size_t testIndex = 0; testIndex < countof(TestFns); ++testIndex)
                    {
                        if (TestFns[testIndex].initialFitFn == nullptr)
                            continue;

                        sprintf_s(buffer, "%s Fit", TestFns[testIndex].name);
                        csv[0].push_back(buffer);

                        TestFns[testIndex].initialFitFn(csv, values);
                    }


                    char fileName[256];
                    sprintf_s(fileName, "out/%s.csv", MakeFns[makeIndex].name);
                    FILE* file = nullptr;
                    fopen_s(&file, fileName, "w+b");

                    for (const TRow& row : csv)
                    {
                        for (const std::string& cell : row)
                            fprintf(file, "\"%s\",", cell.c_str());
                        fprintf(file, "\n");
                    }

                    fclose(file);

                    printf("Done with %s\n", MakeFns[makeIndex].name);

                    makeIndex = nextRow.fetch_add(1);
                }
            }
        );
    }

    for (std::thread& t : threads)
        t.join();

#endif // MAKE_CSVS()

#if DO_PERF_TESTS()
    // Do perf tests
    {
        static std::random_device rd("dev/random");
        static std::seed_seq fullSeed{ rd(), rd(), rd(), rd(), rd(), rd(), rd(), rd() };
        static std::mt19937 rng(fullSeed);

        std::vector<size_t> values, searchValues;
        searchValues.resize(c_perfTestNumSearches);
        values.resize(c_maxNumValues);

        // make the search values that are going to be used by all the tests
        {
            std::uniform_int_distribution<size_t> dist(0, c_maxValue);
            for (size_t & v : searchValues)
                v = dist(rng);
        }

        // binary search, linear search, etc
        for (size_t testIndex = 0; testIndex < countof(TestFns); ++testIndex)
        {
            // quadratic numbers, random numbers, etc
            double timeTotal = 0.0f;
            size_t totalGuesses = 0;
            for (size_t makeIndex = 0; makeIndex < countof(MakeFns); ++makeIndex)
            {
                MakeFns[makeIndex].fn(values, c_maxNumValues);

                size_t guesses = 0;

                std::chrono::high_resolution_clock::time_point start = std::chrono::high_resolution_clock::now();

                // do the searches
                for (size_t searchValue : searchValues)
                {
                    TestResults ret = TestFns[testIndex].fn(values, searchValue);
                    guesses += ret.guesses;
                    totalGuesses += ret.guesses;
                }

                std::chrono::high_resolution_clock::time_point end = std::chrono::high_resolution_clock::now();

                std::chrono::duration<double> duration = std::chrono::duration_cast<std::chrono::duration<double>>(end - start);

                timeTotal += duration.count();
                printf("  %s %s : %f seconds\n", TestFns[testIndex].name, MakeFns[makeIndex].name, duration.count());
            }

            double timePerGuess = (timeTotal * 1000.0 * 1000.0 * 1000.0f) / double(totalGuesses);
            printf("%s total : %f seconds  (%zu guesses = %f nanoseconds per guess)\n\n", TestFns[testIndex].name, timeTotal, totalGuesses, timePerGuess);
        }
    }
#endif // DO_PERF_TESTS()

    system("pause");

    return 0;
}

/*

TODO:

* your test for exiting line search loop seems like it's not quite right. look into it!

* tune line backtracking constants
* make sure everything goes through without going infinite
* add early stopping, and take best thing found instead of last thing found
* probably should have a max iteration count on that line search or a threshold for t?

* need to do initial fit for quadratic it
* need to verify quadratic search function works. Maybe make one function that is templated by the type of fit it does?

* line backtracking works by searching along the gradient up to step size, to find the best improvement it can.
 * (hybrid) line fit search may be good here?! maybe not though.

* make sure and implement line backtracking, even if it works without it!
* keep the coefficients with the lowest error, not just the final ones!

* maybe make a version of the gradient search that preserves ending position instead of ending derivative

* maybe make everything except monotonic fit, then figure that out last.
 * show the initial quadratic fit of data


 ! the problem with gradient descent may be from using unsigned numbers and it causing wrap around to large numbers. that was happening with reporting.
  * frankly, that might be happening with guesses too.

* clean up non monitonic fit.

* get initial quadratic fit in the graph
 * and same for monotonic when you have it

PROBLEM: the step size is not always convergent. Try doing "back tracking line search"?

* Next: maybe get an actual quadratic interpolation search working?

* include the initial (monotonic) quadratic curve fit in the csv so we can graph it along w/ the graph shape.

* this is the quadratic fit search, clean out linear stuff when ready.
* to get a quadratic least squares fit, maybe could do gradient descent for now, and say "if you could calculate this more optimally, it would get more compelling"?
 * could maybe take a curve, make it monotonic, then move it up or down to minimize error (gradient descent C, or better if we can).
* the real solution might have to be something like this: https://en.wikipedia.org/wiki/Quadratic_programming

* try your idea of moving point 0 down til non negative gradient, or point 2 up til same, then moving the entire curve to balance the error.
 * can compare results vs the better gradient descent found curve

* yes, you can invert a non monotonic quadratic. use quadratic equation and do +/- test like you do here. i think the issue is you have to "randomly" choose one instead of getting to know which to use.

* Include normal distribution.
 * Show how the last post things did with normal distribution.

? should you do gradient search? If so, need to compare it to hybrid etc.

! monotonic quadratic fit idea: make the curve monotonic by moving middle point up or down. move the whole curve ("c") up or down by half that amount to center it.
* Revised:
 * if it starts out at a negative derivative, move p0 down until it's 0. whatever that amount moved down, move the entire curve up 2/3 of that amount, so each point is 1/3 that distance away from the fit curve.
 * if it ends at a negative derivative, move p2 up until it's 0. Whatever that amount moved up, move the entire curve down 2/3 of that amount, so each point is 1/3 that distance away from the fit curve.

 * need to test against a normal distribution.
  * include gradient descent version?
  * https://github.com/Atrix256/LinearFitSearch/pull/1

* gradient and quadratic do very poorly on linear outlier. i wonder how a hybrid would do? lots more data to visualize but shrug.

NOTES:
? why wouldn't you just read the beginning and end of the list to get min/max?
 * 2 less memory reads. They come in a single cache line with the count.
 * also, what if you weren't searching an array, but were searching a function f(x)? (restricting to integer values, let's say). and that function was expensive. Say you are raymarching.
  * if you KNOW the min / max due to the usage case, you can use that instead of trying to calculate it from the function each time. If it is costly, you are caching it off.
? when would perf be obviously super compelling?
 * when it's very slow to read from the list, or calculate the item! (like ray marching, or maybe you are trying to find a "minimum error" of a machine learning thing?)

* perf isn't appropriate to measure. not even trying to do things quickly (eg gradient descent in search iteration!), only looking at number of guesses.

* projective gradient descent...
 * do gradient descent but each step, project the point back to a valid point - aka if derivatives need to be > 0, fix it so they are.

! the way i'm doing a quadratic monotonic function fit isn't the only way, and probably isn't the best way for minimizing error of the fit vs the actual data set.
 ? monotonic least squares would be cool. I guess the general case you'd have to say if you want any anchor points?

 * proof of non monotocity on quadratic fit of monotonic points: https://www.wolframalpha.com/input/?i=quadratic+fit+(0,1),(1,2),(2,10)

? quadratic programming? convex optimization? lagrange multipliers.
 * i wish i knew these things. Maybe later & i can explain them with blog posts of their own :P

* thank wayne (and others?) for the math help?

* mention the usage cases: raymarching, querying some external service, being on disk, etc

* the linear outlier really seems like a worst case scenario for all searches except binary search.

? should we show the last blog post test w/ normal distribution numbers? i think so... maybe as a first item before getting into the new stuff.


! lots of cool links here about gradient descent. extract them? also link twitter conversation?
* https://twitter.com/Atrix256/status/1116551349330513920

! the gradient fit doesn't always match the end points!  There are 4 constraints (start/end pos/derivative) and 3 unknowns (A,B,C) so it's overconstrained.
 * still, how does it do?
 * it does make a monotonic function though.


*/

#include "stdio.h"
#include <vector>
#include <random>
#include <thread>
#include <atomic>
#include <string>
#include <chrono>
#include <array>
#include <assert.h>

static const size_t c_maxValue = 2000;           // the sorted arrays will have values between 0 and this number in them (inclusive)
static const size_t c_maxNumValues = 1000;       // the graphs will graph between 1 and this many values in a sorted array
static const size_t c_numRunsPerTest = 100;      // how many times does it do the same test to gather min, max, average?
static const size_t c_perfTestNumSearches = 100000; // how many searches are going to be done per list type, to come up with timing for a search type.

#define VERIFY_RESULT() 1 // verifies that the search functions got the right answer. prints out a message if they didn't.
#define MAKE_CSVS() 1 // the main test

struct TestResults
{
    bool found;
    size_t index;
    size_t guesses;
};

using MakeListFn = void(*)(std::vector<size_t>& values, size_t count);
using TestListFn = TestResults(*)(const std::vector<size_t>& values, size_t searchValue);

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
        float y = log(x+1) / maxValue;
        y *= c_maxValue;
        values[index] = size_t(y);
    }

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
    // We are calculating / minimizing "Error Squared" so that we treat negative
    // and positive error the same, and try to reach zero error.
    //
    // dError^2 / dC = 2Ax^2 + 2Bx   + 2C    - 2y
    // dError^2 / dB = 2Ax^3 + 2Bx^2 + 2Cx   - 2xy
    // dError^2 / dA = 2Ax^4 + 2Bx^3 + 2Cx^2 - 2x^2y
    //
    // Observation: start with dEerror^2 / dC
    //              multiply by x to get dEerror^2 / dB
    //              multiply by x to get dEerror^2 / dA

    // how much the error squared for a single point changes as C changes
    auto dErrorSquared_dC_SinglePoint = [](const Vec3& coefficients, const Vec2u& point) -> double
    {
        double A = coefficients[0];
        double B = coefficients[1];
        double C = coefficients[2];

        double x = double(point[0]);
        double y = double(point[1]);

        // 2Ax^2 + 2Bx + 2C - 2y
        return 2.0*A*x*x +
               2.0*B*x +
               2.0*C -
               2.0*y;
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

    MakeQuadraticMonotonic_ProjectiveGradientDescent_ProjectValid(data, coefficients);

    Validate(coefficients);

    printf("Starting Error = %f\n", CalculateMeanSquaredError(data, coefficients));

    for (int i = 0; i < c_numIterations; ++i)
    {
        // TODO: remove when done
        float MSEBefore = CalculateMeanSquaredError(data, coefficients);

        Vec3 errorGradient = CalculateErrorSquaredGradient(data, coefficients);
        coefficients[0] -= errorGradient[0] * c_stepSize;
        coefficients[1] -= errorGradient[1] * c_stepSize;
        coefficients[2] -= errorGradient[2] * c_stepSize;

        float MSEAfter = CalculateMeanSquaredError(data, coefficients);

        if (MSEAfter > MSEBefore * 2 && MSEBefore > 1.0f)
            int ijkl = 0;

        Validate(coefficients);

        MakeQuadraticMonotonic_ProjectiveGradientDescent_ProjectValid(data, coefficients);

        Validate(coefficients);

        //float gradientLength = sqrtf(errorGradient[0] * errorGradient[0] + errorGradient[1] * errorGradient[1] + errorGradient[2] * errorGradient[2]);
        //printf("[%i] Error = %f (grad length = %f)\n", i, CalculateMeanSquaredError(data, coefficients), gradientLength);
    }

    printf("Ending Error = %f\n", CalculateMeanSquaredError(data, coefficients));
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

        if (guessIndex < minIndex || guessIndex > maxIndex)
            printf("No Guess Valid?!");

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

TestResults TestList_LineFitBlind(const std::vector<size_t>& values, size_t searchValue)
{
    // If you want to know how this does against binary search without first knowing the min and max, this result is for you.
    // It takes 2 extra samples to get the min and max, so we are counting those as guesses (memory reads).
    TestResults ret = TestList_LineFit(values, searchValue);
    ret.guesses += 2;
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

        if (gte == false || lte == false)
            printf("VERIFICATION FAILURE!! Not a valid place to insert a new value! %s, %s\n", list, test);
    }

#endif
}

// TODO: delete?
void QuadraticFitTest(const Vec2u data[3])
{
    Vec3 coefficients = { 0.0f, 0.0f, 0.0f };
    QuadraticFit(data, coefficients);

    printf("Fit: %f * x^2 + %f * x + %f\n\n", coefficients[0], coefficients[1], coefficients[2]);
    printf("A = %f\n", coefficients[0]);
    printf("B = %f\n", coefficients[1]);
    printf("C = %f\n\n", coefficients[2]);

    printf("f(%zu) = %f\n", data[0][0], EvaluateQuadratic(coefficients, float(data[0][0])));
    printf("f(%zu) = %f\n", data[1][0], EvaluateQuadratic(coefficients, float(data[1][0])));
    printf("f(%zu) = %f\n", data[2][0], EvaluateQuadratic(coefficients, float(data[2][0])));

    printf("Error of exact fit = %f\n", CalculateMeanSquaredError(data, coefficients));

    MakeQuadraticMonotonic_ProjectiveGradientDescent(data, coefficients);

    printf("Fit: %f * x^2 + %f * x + %f\n\n", coefficients[0], coefficients[1], coefficients[2]);
    printf("A = %f\n", coefficients[0]);
    printf("B = %f\n", coefficients[1]);
    printf("C = %f\n\n", coefficients[2]);

    printf("f(%zu) = %f\n", data[0][0], EvaluateQuadratic(coefficients, float(data[0][0])));
    printf("f(%zu) = %f\n", data[1][0], EvaluateQuadratic(coefficients, float(data[1][0])));
    printf("f(%zu) = %f\n", data[2][0], EvaluateQuadratic(coefficients, float(data[2][0])));
}

int main(int argc, char** argv)
{
    // TODO: delete when you feel like it.
    Vec2u data[3] = { {0,1}, {1,2}, {2, 10} };
    QuadraticFitTest(data);

    MakeListInfo MakeFns[] =
    {
        /*
        {"Random", MakeList_Random},
        {"Linear", MakeList_Linear},
        {"Linear Outlier", MakeList_Linear_Outlier},
        */
        {"Quadratic", MakeList_Quadratic},
        /*
        {"Cubic", MakeList_Cubic},
        {"Log", MakeList_Log},
        */
    };

    TestListInfo TestFns[] =
    {
        /*
        {"Linear Search", TestList_LinearSearch},
        {"Line Fit", TestList_LineFit},
        {"Line Fit Blind", TestList_LineFitBlind},
        {"Binary Search", TestList_BinarySearch},
        {"Life Fit Hybrid", TestList_LineFitHybridSearch},
        */
        {"Quadratic Fit", TestList_QuadraticFit},

        // Quadratic Hybrid Fit!
    };

#if MAKE_CSVS()

    size_t numThreads = std::thread::hardware_concurrency();
    std::vector<std::thread> threads;
    threads.resize(numThreads);

    typedef std::vector<std::string> TRow;
    typedef std::vector<TRow> TSheet;

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

    system("pause");

    return 0;
}

/*

TODO:

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

*/

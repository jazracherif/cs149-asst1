#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <immintrin.h>
#include <cstring>

int VECTOR_WIDTH = 8;

void printVect(__m256 vect){
    float values[VECTOR_WIDTH];

    _mm256_store_ps(values, vect);

    printf("%f,%f,%f,%f,%f,%f,%f,%f\n", values[0],values[1],values[2],values[3],values[4],values[5],values[6],values[7]);

}

void sqrtVectorized(int N,
                float initialGuess,
                const float *values,
                float output[])
{

    static const float kThreshold = 0.00001f;
    __m256 _x;
    __m256 doneLanes;

    __m256 guess = _mm256_set1_ps(initialGuess);
    __m256 allHalf = _mm256_set1_ps(0.5f);
    __m256 allOne = _mm256_set1_ps(1.f);
    __m256 allThrees = _mm256_set1_ps(3.f);

    // for absolute
    __m256i int_mask = _mm256_set1_epi32(0x7FFFFFFF);
    __m256 float_mask = _mm256_castsi256_ps(int_mask);

    __m256 kThreshold_ = _mm256_set1_ps(kThreshold);

    float out[VECTOR_WIDTH]; // for the result

    for (int i=0; i<N; i += VECTOR_WIDTH) {
        _x = _mm256_loadu_ps(values + i);

        guess = _mm256_set1_ps(initialGuess);

        // error = fabs(guess * guess * x - 1.f)
        __m256 res = _mm256_sub_ps(_mm256_mul_ps(_mm256_mul_ps(guess, guess), _x), allOne);
        __m256 error = _mm256_and_ps(res, float_mask); // clear the sign bit of each lane
        
        // active lanes are those whose error values is smaller than threshold
        doneLanes = _mm256_cmp_ps(error, kThreshold_, _CMP_LT_OS);
        int done = _mm256_movemask_ps(doneLanes);

        // int iter =0;
        while (done != 0xFF){ // check if all lanes are inactive
            // we need to keep a mask of the lanes 
            //  guess = (3.f * guess - x * guess * guess * guess) * 0.5f;
            guess =  _mm256_mul_ps(
                        _mm256_sub_ps(
                            _mm256_mul_ps(guess, allThrees), 
                            _mm256_mul_ps(_x, 
                                _mm256_mul_ps(
                                    _mm256_mul_ps(guess, guess), 
                                    guess)
                                )
                            ),
                        allHalf          
                    );

            // error = fabs(guess * guess * x - 1.f)
            res = _mm256_sub_ps( _mm256_mul_ps(_mm256_mul_ps(guess, guess), _x), allOne);
            error = _mm256_and_ps(res, float_mask);

            doneLanes = _mm256_cmp_ps(error, kThreshold_, _CMP_LT_OS);
            done = _mm256_movemask_ps(doneLanes);            
        }

        _mm256_store_ps(out, _mm256_mul_ps(_x, guess));
        memcpy(output + i, out, sizeof(float) * VECTOR_WIDTH);
    }
}

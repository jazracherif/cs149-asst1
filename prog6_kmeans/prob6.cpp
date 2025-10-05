#include <algorithm>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <thread>
#include <immintrin.h>

#include "CycleTimer.h"

using namespace std;

extern double dist(double *x, double *y, int nDim);


typedef struct {
  // Control work assignments
  int start, end;

  // Shared by all functions
  double *data;
  double *clusterCentroids;
  int *clusterAssignments;
  double *currCost;
  int M, N, K;

  // use for assignment case
  int numPoints;
  int threadId;
} WorkerArgs;


/**
 * Checks if the algorithm has converged.
 * 
 * @param prevCost Pointer to the K dimensional array containing cluster costs 
 *    from the previous iteration.
 * @param currCost Pointer to the K dimensional array containing cluster costs 
 *    from the current iteration.
 * @param epsilon Predefined hyperparameter which is used to determine when
 *    the algorithm has converged.
 * @param K The number of clusters.
 * 
 * NOTE: DO NOT MODIFY THIS FUNCTION!!!
 */
static bool stoppingConditionMet(double *prevCost, double *currCost,
                                 double epsilon, int K) {
  for (int k = 0; k < K; k++) {
    if (abs(prevCost[k] - currCost[k]) > epsilon)
      return false;
  }
  return true;
}

/**
 * Vectorized version of dist that computes L2 distance between two points of dimension nDim. 
 * 
 * @param x Pointer to the beginning of the array representing the first
 *     data point.
 * @param y Poitner to the beginning of the array representing the second
 *     data point.
 * @param nDim The dimensionality (number of elements) in each data point
 *     (must be the same for x and y).
 */

int VECTOR_WIDTH = 4;
double zeros[4] = {0.0, 0.0, 0.0, 0.0};


double distVectorized( double *x,  double *y, int nDim) {

  double accum = 0.0;
  
  int nonVec = nDim % VECTOR_WIDTH;
  if (nonVec != 0){
    for (int i = 0; i < nonVec; i++) {
        accum += pow((x[i] - y[i]), 2);
      }
  }

  double res[VECTOR_WIDTH];

  __m256d acc = _mm256_loadu_pd(zeros);

  // do the rest vectorized
  for (int i = nonVec; i < nDim; i += VECTOR_WIDTH) {

    __m256d x_ = _mm256_loadu_pd(x + i);
    __m256d y_ = _mm256_loadu_pd(y + i);

    // x[i] - y[i]
    __m256d sub = _mm256_sub_pd(x_, y_);

    // (x[i] - y[i])^2
    __m256d sq = _mm256_mul_pd(sub, sub);

    // Accumulate
    acc = _mm256_add_pd(acc, sq);
  
  }

  // sum up the remaining values together and square the result
  _mm256_store_pd(res, acc);
  for (int j = 0; j < VECTOR_WIDTH; j++){
      accum += res[j];
  }
   
  return sqrt(accum);
}


void computeAssignmentsParallel(WorkerArgs *const args) {
  // compute the assignment all points in range args->M to args->M + num_points
  // and update the corresponding index in args->clusterCentroids

  double *minDist = new double[args->numPoints];
  
  // Initialize arrays
  for (int m = args->M; m < args->M + args->numPoints; m++) {
    minDist[m - args->M] = 1e30;
    args->clusterAssignments[m] = -1;
  }

  // Assign datapoints to closest centroids
  for (int k = args->start; k < args->end; k++) {
    for (int m = args->M; m < args->M + args->numPoints; m++) {

      // printf("assign point m: %d\n", m);
      double d = distVectorized(&args->data[m * args->N],
                      &args->clusterCentroids[k * args->N], args->N);

      if (d < minDist[m - args->M]) {
        minDist[m - args->M] = d;
        args->clusterAssignments[m] = k;
      }
    }
  }

  free(minDist);
}


/**
 * Given the cluster assignments, computes the new centroid locations for
 * each cluster.
 */
void computeCentroids_base(WorkerArgs *const args) {
  int *counts = new int[args->K];

  // Zero things out
  for (int k = 0; k < args->K; k++) {
    counts[k] = 0;
    for (int n = 0; n < args->N; n++) {
      args->clusterCentroids[k * args->N + n] = 0.0;
    }
  }


  // Sum up contributions from assigned examples
  for (int m = 0; m < args->M; m++) {
    int k = args->clusterAssignments[m];
    for (int n = 0; n < args->N; n++) {
      args->clusterCentroids[k * args->N + n] +=
          args->data[m * args->N + n];
    }
    counts[k]++;
  }

  // Compute means
  for (int k = 0; k < args->K; k++) {
    counts[k] = max(counts[k], 1); // prevent divide by 0
    for (int n = 0; n < args->N; n++) {
      args->clusterCentroids[k * args->N + n] /= counts[k];
    }
  }

  free(counts);
}


void computeCostVectorized(WorkerArgs *const args) {
  double *accum = new double[args->K];

  // Zero things out
  for (int k = 0; k < args->K; k++) {
    accum[k] = 0.0;
  }

  // Sum cost for all data points assigned to centroid
  for (int m = 0; m < args->M; m++) {
    int k = args->clusterAssignments[m];
    accum[k] += distVectorized(&args->data[m * args->N],
                     &args->clusterCentroids[k * args->N], args->N);
  }

  // Update costs
  for (int k = args->start; k < args->end; k++) {
    args->currCost[k] = accum[k];
  }

  free(accum);
}

/**
 * Assign work to threads for the assign step of kmeans
 */
void assignThreadsPoints(WorkerArgs args[], int numThreads, double *currCost, double *data, double *clusterCentroids, int *clusterAssignments,
               int M, int N, int K ){
  // thread assignment
  int numPoints =  M / numThreads;

  for (int i=0; i <numThreads; i++) {   
    args[i].data = data;
    args[i].clusterCentroids = clusterCentroids;
    args[i].clusterAssignments = clusterAssignments;
    args[i].currCost = currCost;
    args[i].numPoints = (i == numThreads - 1) ? M - i * numPoints :  numPoints;
    args[i].M = i * numPoints; 
    args[i].N = N;
    args[i].K = K;
    args[i].start = 0;
    args[i].end = K;
    args[i].threadId = i;
  }
}

/**
 * Run the assign step of k-means in parallel
 */
void assignParallel(int numThreads, double *currCost, double *data, double *clusterCentroids, int *clusterAssignments,
               int M, int N, int K){
  std::thread workers[numThreads];
  WorkerArgs assignArgs[numThreads];

  assignThreadsPoints(assignArgs, numThreads, currCost, data, clusterCentroids, clusterAssignments, M, N, K );

  for (int i=0; i< numThreads; i++) {
      workers[i] = std::thread(computeAssignmentsParallel, &assignArgs[i]);
  }
  // join worker threads
  for (int i=0; i<numThreads; i++) {
      workers[i].join();
  }
}

/**
 * Computes the K-Means algorithm, using std::thread to parallelize the work.
 *
 * @param data Pointer to an array of length M*N representing the M different N 
 *     dimensional data points clustered. The data is layed out in a "data point
 *     major" format, so that data[i*N] is the start of the i'th data point in 
 *     the array. The N values of the i'th datapoint are the N values in the 
 *     range data[i*N] to data[(i+1) * N].
 * @param clusterCentroids Pointer to an array of length K*N representing the K 
 *     different N dimensional cluster centroids. The data is laid out in
 *     the same way as explained above for data.
 * @param clusterAssignments Pointer to an array of length M representing the
 *     cluster assignments of each data point, where clusterAssignments[i] = j
 *     indicates that data point i is closest to cluster centroid j.
 * @param M The number of data points to cluster.
 * @param N The dimensionality of the data points.
 * @param K The number of cluster centroids.
 * @param epsilon The algorithm is said to have converged when
 *     |currCost[i] - prevCost[i]| < epsilon for all i where i = 0, 1, ..., K-1
 */
void kMeansThread(double *data, double *clusterCentroids, int *clusterAssignments,
               int M, int N, int K, double epsilon) {

  double startTime, endTime;
  printf("kMeansThread Optimized version\n");

  // Used to track convergence
  double *prevCost = new double[K];
  double *currCost = new double[K];

  // The WorkerArgs array is used to pass inputs to and return output from
  // functions.
  WorkerArgs args;
  args.data = data;
  args.clusterCentroids = clusterCentroids;
  args.clusterAssignments = clusterAssignments;
  args.currCost = currCost;
  args.M = M;
  args.N = N;
  args.K = K;

  // Initialize arrays to track cost
  for (int k = 0; k < K; k++) {
    prevCost[k] = 1e30;
    currCost[k] = 0.0;
  }

  int MAX_THREADS = 4;
  int numThreads = MAX_THREADS;
  std::thread workers[MAX_THREADS];
  WorkerArgs assignArgs[MAX_THREADS];

  /* Main K-Means Algorithm Loop */
  int iter = 0;
  while (!stoppingConditionMet(prevCost, currCost, epsilon, K)) {
    // Update cost arrays (for checking convergence criteria)
    for (int k = 0; k < K; k++) {
      prevCost[k] = currCost[k];
    }

    // Setup args struct
    args.start = 0;
    args.end = K;

    // Multithreaded implementation of the assignment part of k-means + use vectorized dist implementation
    assignParallel(numThreads, currCost, data, clusterCentroids, clusterAssignments, M, N, K );

    // Same as the baseline implementation
    computeCentroids_base(&args);

    // Compute cost with vectorized dist implementation 
    computeCostVectorized(&args);

    iter++;
  }

  free(currCost);
  free(prevCost);
}
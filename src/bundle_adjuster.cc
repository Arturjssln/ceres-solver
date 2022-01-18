// Ceres Solver - A fast non-linear least squares minimizer
// Copyright 2015 Google Inc. All rights reserved.
// http://ceres-solver.org/
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
// * Redistributions of source code must retain the above copyright notice,
//   this list of conditions and the following disclaimer.
// * Redistributions in binary form must reproduce the above copyright notice,
//   this list of conditions and the following disclaimer in the documentation
//   and/or other materials provided with the distribution.
// * Neither the name of Google Inc. nor the names of its contributors may be
//   used to endorse or promote products derived from this software without
//   specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
// LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
// CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
// SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
// INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
// CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
// ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.
//
// Author: sameeragarwal@google.com (Sameer Agarwal)
//
// An example of solving a dynamically sized problem with various
// solvers and loss functions.
//
// For a simpler bare bones example of doing bundle adjustment with
// Ceres, please see simple_bundle_adjuster.cc.
//
// NOTE: This example will not compile without gflags and SuiteSparse.
//
// The problem being solved here is known as a Bundle Adjustment
// problem in computer vision. Given a set of 3d points X_1, ..., X_n,
// a set of cameras P_1, ..., P_m. If the point X_i is visible in
// image j, then there is a 2D observation u_ij that is the expected
// projection of X_i using P_j. The aim of this optimization is to
// find values of X_i and P_j such that the reprojection error
//
//    E(X,P) =  sum_ij  |u_ij - P_j X_i|^2
//
// is minimized.
//
// The problem used here comes from a collection of bundle adjustment
// problems published at University of Washington.
// http://grail.cs.washington.edu/projects/bal

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

#include "ba_problem.h"
#include "ceres/ceres.h"
#include "gflags/gflags.h"
#include "glog/logging.h"
#include "snavely_reprojection_error.h"

// clang-format makes the gflags definitions too verbose
// clang-format off

DEFINE_string(input, "", "Input File name");
DEFINE_string(output, "", "Output File name");
DEFINE_string(trust_region_strategy, "levenberg_marquardt",
              "Options are: levenberg_marquardt, dogleg.");
DEFINE_string(dogleg, "traditional_dogleg", "Options are: traditional_dogleg,"
              "subspace_dogleg.");

DEFINE_bool(inner_iterations, false, "Use inner iterations to non-linearly "
            "refine each successful trust region step.");

DEFINE_string(blocks_for_inner_iterations, "automatic", "Options are: "
              "automatic, cameras, points, cameras,points, points,cameras");

DEFINE_string(linear_solver, "dense_schur", "Options are: "
              "sparse_schur, dense_schur, iterative_schur, sparse_normal_cholesky, "
              "dense_qr, dense_normal_cholesky and cgnr.");
DEFINE_bool(explicit_schur_complement, false, "If using ITERATIVE_SCHUR "
            "then explicitly compute the Schur complement.");
DEFINE_string(preconditioner, "jacobi", "Options are: "
              "identity, jacobi, schur_jacobi, cluster_jacobi, "
              "cluster_tridiagonal.");
DEFINE_string(visibility_clustering, "canonical_views",
              "single_linkage, canonical_views");

DEFINE_string(sparse_linear_algebra_library, "cx_sparse",
              "Options are: suite_sparse and cx_sparse.");
DEFINE_string(dense_linear_algebra_library, "eigen",
              "Options are: eigen and lapack.");
DEFINE_string(ordering, "automatic", "Options are: automatic, user.");

DEFINE_bool(optimize_cam, false, "If true, optimize the camera pose and rotation. "
            "If false, cameras paramaters are not optimized.");
DEFINE_bool(use_confidence, false, "If true, use confidence. "
            "If false, all estimation have same weights.");
DEFINE_bool(gd_points, false, "If true, consider points as groundtruth. "
            "If false, points are estimated.");           

DEFINE_bool(robustify, false, "Use a robust loss function.");

DEFINE_double(eta, 1e-2, "Default value for eta. Eta determines the "
              "accuracy of each linear solve of the truncated newton step. "
              "Changing this parameter can affect solve performance.");

DEFINE_int32(num_threads, 1, "Number of threads.");
DEFINE_int32(num_iterations, 1000, "Number of iterations.");
DEFINE_double(max_solver_time, 1e32, "Maximum solve time in seconds.");
DEFINE_bool(nonmonotonic_steps, false, "Trust region algorithm can use"
            " nonmonotic steps.");

DEFINE_double(rotation_sigma, 0.0, "Standard deviation of camera rotation "
              "perturbation.");
DEFINE_double(translation_sigma, 0.0, "Standard deviation of the camera "
              "translation perturbation.");
DEFINE_double(point_sigma, 0.0, "Standard deviation of the point "
              "perturbation.");
DEFINE_int32(random_seed, 38401, "Random seed used to set the state "
             "of the pseudo random number generator used to generate "
             "the pertubations.");
DEFINE_bool(line_search, false, "Use a line search instead of trust region "
            "algorithm.");
DEFINE_bool(mixed_precision_solves, false, "Use mixed precision solves.");
DEFINE_int32(max_num_refinement_iterations, 0, "Iterative refinement iterations");
DEFINE_string(initial_ply, "", "Export the BAL file data as a PLY file.");
DEFINE_string(final_ply, "", "Export the refined BAL file data as a PLY "
              "file.");

// clang-format on

namespace ceres {
namespace {

void SetLinearSolver(Solver::Options* options) {
  CHECK(StringToLinearSolverType(CERES_GET_FLAG(FLAGS_linear_solver),
                                 &options->linear_solver_type));
  CHECK(StringToPreconditionerType(CERES_GET_FLAG(FLAGS_preconditioner),
                                   &options->preconditioner_type));
  CHECK(StringToVisibilityClusteringType(
      CERES_GET_FLAG(FLAGS_visibility_clustering),
      &options->visibility_clustering_type));
  CHECK(StringToSparseLinearAlgebraLibraryType(
      CERES_GET_FLAG(FLAGS_sparse_linear_algebra_library),
      &options->sparse_linear_algebra_library_type));
  CHECK(StringToDenseLinearAlgebraLibraryType(
      CERES_GET_FLAG(FLAGS_dense_linear_algebra_library),
      &options->dense_linear_algebra_library_type));
  options->use_explicit_schur_complement =
      CERES_GET_FLAG(FLAGS_explicit_schur_complement);
  options->use_mixed_precision_solves =
      CERES_GET_FLAG(FLAGS_mixed_precision_solves);
  options->max_num_refinement_iterations =
      CERES_GET_FLAG(FLAGS_max_num_refinement_iterations);
}

void SetOrdering(BAProblem* ba_problem, Solver::Options* options) {
  const int num_points = ba_problem->num_points();
  const int point_block_size = ba_problem->point_block_size();
  double* points = ba_problem->mutable_points();

  const int num_cameras = ba_problem->num_cameras();
  const int camera_block_size = ba_problem->camera_block_size();
  double* cameras = ba_problem->mutable_cameras();

  if (options->use_inner_iterations) {
    if (CERES_GET_FLAG(FLAGS_blocks_for_inner_iterations) == "cameras") {
      LOG(INFO) << "Camera blocks for inner iterations";
      options->inner_iteration_ordering.reset(new ParameterBlockOrdering);
      for (int i = 0; i < num_cameras; ++i) {
        options->inner_iteration_ordering->AddElementToGroup(
            cameras + camera_block_size * i, 0);
      }
    } else if (CERES_GET_FLAG(FLAGS_blocks_for_inner_iterations) == "points") {
      LOG(INFO) << "Point blocks for inner iterations";
      options->inner_iteration_ordering.reset(new ParameterBlockOrdering);
      for (int i = 0; i < num_points; ++i) {
        options->inner_iteration_ordering->AddElementToGroup(
            points + point_block_size * i, 0);
      }
    } else if (CERES_GET_FLAG(FLAGS_blocks_for_inner_iterations) ==
               "cameras,points") {
      LOG(INFO) << "Camera followed by point blocks for inner iterations";
      options->inner_iteration_ordering.reset(new ParameterBlockOrdering);
      for (int i = 0; i < num_cameras; ++i) {
        options->inner_iteration_ordering->AddElementToGroup(
            cameras + camera_block_size * i, 0);
      }
      for (int i = 0; i < num_points; ++i) {
        options->inner_iteration_ordering->AddElementToGroup(
            points + point_block_size * i, 1);
      }
    } else if (CERES_GET_FLAG(FLAGS_blocks_for_inner_iterations) ==
               "points,cameras") {
      LOG(INFO) << "Point followed by camera blocks for inner iterations";
      options->inner_iteration_ordering.reset(new ParameterBlockOrdering);
      for (int i = 0; i < num_cameras; ++i) {
        options->inner_iteration_ordering->AddElementToGroup(
            cameras + camera_block_size * i, 1);
      }
      for (int i = 0; i < num_points; ++i) {
        options->inner_iteration_ordering->AddElementToGroup(
            points + point_block_size * i, 0);
      }
    } else if (CERES_GET_FLAG(FLAGS_blocks_for_inner_iterations) ==
               "automatic") {
      LOG(INFO) << "Choosing automatic blocks for inner iterations";
    } else {
      LOG(FATAL) << "Unknown block type for inner iterations: "
                 << CERES_GET_FLAG(FLAGS_blocks_for_inner_iterations);
    }
  }

  // Bundle adjustment problems have a sparsity structure that makes
  // them amenable to more specialized and much more efficient
  // solution strategies. The SPARSE_SCHUR, DENSE_SCHUR and
  // ITERATIVE_SCHUR solvers make use of this specialized
  // structure.
  //
  // This can either be done by specifying Options::ordering_type =
  // ceres::SCHUR, in which case Ceres will automatically determine
  // the right ParameterBlock ordering, or by manually specifying a
  // suitable ordering vector and defining
  // Options::num_eliminate_blocks.
  if (CERES_GET_FLAG(FLAGS_ordering) == "automatic") {
    return;
  }

  ceres::ParameterBlockOrdering* ordering = new ceres::ParameterBlockOrdering;

  // The points come before the cameras.
  for (int i = 0; i < num_points; ++i) {
    ordering->AddElementToGroup(points + point_block_size * i, 0);
  }

  for (int i = 0; i < num_cameras; ++i) {
    // When using axis-angle, there is a single parameter block for
    // the entire camera.
    ordering->AddElementToGroup(cameras + camera_block_size * i, 1);
  }

  options->linear_solver_ordering.reset(ordering);
}

void SetMinimizerOptions(Solver::Options* options) {
  options->max_num_iterations = CERES_GET_FLAG(FLAGS_num_iterations);
  options->minimizer_progress_to_stdout = true;
  options->num_threads = CERES_GET_FLAG(FLAGS_num_threads);
  options->eta = CERES_GET_FLAG(FLAGS_eta);
  options->max_solver_time_in_seconds = CERES_GET_FLAG(FLAGS_max_solver_time);
  options->use_nonmonotonic_steps = CERES_GET_FLAG(FLAGS_nonmonotonic_steps);
  if (CERES_GET_FLAG(FLAGS_line_search)) {
    options->minimizer_type = ceres::LINE_SEARCH;
  }

  CHECK(StringToTrustRegionStrategyType(
      CERES_GET_FLAG(FLAGS_trust_region_strategy),
      &options->trust_region_strategy_type));
  CHECK(
      StringToDoglegType(CERES_GET_FLAG(FLAGS_dogleg), &options->dogleg_type));
  options->use_inner_iterations = CERES_GET_FLAG(FLAGS_inner_iterations);
}

void SetSolverOptionsFromFlags(Solver::Options* options) {
  SetMinimizerOptions(options);
  SetLinearSolver(options);
  //SetOrdering(ba_problem, options);
}

void BuildProblem(BAProblem* ba_problem, Problem* problem, int scene_id) {
  // Observations is 2*num_observations long array observations =
  // [u_1, u_2, ... , u_n], where each u_i is two dimensional, the x
  // and y positions of the observation.
  int start = ba_problem->scene_start_for_scene_id(scene_id);
  int length = ba_problem->scene_length_for_scene_id(scene_id);
  const double* observations = ba_problem->observations();
  const double* confidence_scores = ba_problem->confidence();
  
  for (int i = start; i < start + length; ++i) {
    CostFunction* cost_function;
    if (CERES_GET_FLAG(FLAGS_gd_points)) {
      cost_function = SnavelyReprojectionErrorNoPoint::Create(
          observations[2 * i + 0],
          observations[2 * i + 1],
          ba_problem->point_for_observation(i));
    } else {
      if (CERES_GET_FLAG(FLAGS_optimize_cam)) {
        if (CERES_GET_FLAG(FLAGS_use_confidence)) {
          cost_function = SnavelyReprojectionError::Create(
                            observations[2 * i + 0],
                            observations[2 * i + 1], 
                            confidence_scores[i]);
        } else {
          cost_function = SnavelyReprojectionErrorNoConfidence::Create(
                            observations[2 * i + 0],
                            observations[2 * i + 1]);
        }
      } else {
        if (CERES_GET_FLAG(FLAGS_use_confidence)) {
          cost_function = SnavelyReprojectionErrorNoCam::Create(
                            observations[2 * i + 0],
                            observations[2 * i + 1], 
                            confidence_scores[i],
                            ba_problem->camera_for_observation(i));
        } else {
          cost_function = SnavelyReprojectionErrorNoConfidenceNoCam::Create(
                            observations[2 * i + 0],
                            observations[2 * i + 1], 
                            ba_problem->camera_for_observation(i));
        }
      }
    }
    // Each observation correponds to a pair of a camera and a point
    double* point = ba_problem->mutable_point_for_observation(i);
    // If it's not the first scene, initialize the point with the previous scene's
    if (scene_id > 0) {
      double* prev_point = ba_problem->mutable_point_for_observation(i, true);
      for (int j = 0; j < ba_problem->point_block_size(); ++j) {
        point[j] = prev_point[j];
      }
    }
    double* camera = ba_problem->mutable_camera_for_observation(i);
    // If enabled use Huber's loss function.
    LossFunction* loss_function =
        CERES_GET_FLAG(FLAGS_robustify) ? new HuberLoss(1.0) : NULL;
    
    if (CERES_GET_FLAG(FLAGS_gd_points)) {
      problem->AddResidualBlock(cost_function, 
                                loss_function,  
                                camera);
    } else if (CERES_GET_FLAG(FLAGS_optimize_cam)) {
      problem->AddResidualBlock(cost_function, 
                                loss_function,  
                                camera, 
                                point);
    } else {
      problem->AddResidualBlock(cost_function, 
                                loss_function, 
                                point);
    }
  }
}

void SolveProblem(const char* filename) {
  BAProblem ba_problem(filename);

  if (!CERES_GET_FLAG(FLAGS_initial_ply).empty()) {
    ba_problem.WriteToPLYFile(CERES_GET_FLAG(FLAGS_initial_ply));
  }

  Solver::Options options;
  SetSolverOptionsFromFlags(&options);


  srand(CERES_GET_FLAG(FLAGS_random_seed));
  // ba_problem.Normalize();
  // ba_problem.Perturb(CERES_GET_FLAG(FLAGS_rotation_sigma),
  //                     CERES_GET_FLAG(FLAGS_translation_sigma),
  //                     CERES_GET_FLAG(FLAGS_point_sigma));
  Problem problem;
  for (int scene_id = 0; scene_id < ba_problem.num_scenes(); ++scene_id) {
    BuildProblem(&ba_problem, &problem, scene_id);
  }
  options.gradient_tolerance = 1e-16;
  options.function_tolerance = 1e-16;
  Solver::Summary summary;
  Solve(options, &problem, &summary);
  std::cout << summary. FullReport() << std::endl;

  if (!CERES_GET_FLAG(FLAGS_final_ply).empty()) {
    ba_problem.WriteToPLYFile(CERES_GET_FLAG(FLAGS_final_ply));
  }
  if (!CERES_GET_FLAG(FLAGS_output).empty()) {
    ba_problem.WriteToFile(CERES_GET_FLAG(FLAGS_output));
  }
}

}  // namespace
}  // namespace ceres

int main(int argc, char** argv) {
  GFLAGS_NAMESPACE::ParseCommandLineFlags(&argc, &argv, true);
  google::InitGoogleLogging(argv[0]);
  if (CERES_GET_FLAG(FLAGS_input).empty()) {
    LOG(ERROR) << "Usage: bundle_adjuster --input=ba_problem";
    return 1;
  }

  ceres::SolveProblem(CERES_GET_FLAG(FLAGS_input).c_str());
  return 0;
}

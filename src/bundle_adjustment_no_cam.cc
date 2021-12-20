#include <cmath>
#include <cstdio>
#include <iostream>
#include "ceres/ceres.h"
#include "ceres/rotation.h"

// Read a Bundle Adjustment in the Large dataset.
class BAProblem {
 public:
  ~BAProblem() {
    delete[] point_index_;
    delete[] camera_index_;
    delete[] observations_;
    delete[] parameters_;
  }
  int num_observations() const { return num_observations_; }
  const double* observations() const { return observations_; }
  const double* cameras() { return parameters_; }
  const double* camera_for_observation(int i) { return cameras() + camera_index_[i] * 9; }
  double* mutable_points() { return parameters_ + 9 * num_cameras_; }
  double* mutable_point_for_observation(int i) { return mutable_points() + point_index_[i] * 3; }
  
  bool LoadFile(const char* filename) {
    FILE* fptr = fopen(filename, "r");
    if (fptr == NULL) {
      return false;
    };
    FscanfOrDie(fptr, "%d", &num_cameras_);
    FscanfOrDie(fptr, "%d", &num_points_);
    FscanfOrDie(fptr, "%d", &num_observations_);
    point_index_ = new int[num_observations_];
    camera_index_ = new int[num_observations_];
    observations_ = new double[2 * num_observations_];
    num_parameters_ = 9 * num_cameras_ + 3 * num_points_;
    parameters_ = new double[num_parameters_];
    for (int i = 0; i < num_observations_; ++i) {
      FscanfOrDie(fptr, "%d", camera_index_ + i);
      FscanfOrDie(fptr, "%d", point_index_ + i);
      for (int j = 0; j < 2; ++j) {
        FscanfOrDie(fptr, "%lf", observations_ + 2 * i + j);
      }
    }
    for (int i = 0; i < num_parameters_; ++i) {
      FscanfOrDie(fptr, "%lf", parameters_ + i);
    }
    return true;
  }
 private:
  template <typename T>
  void FscanfOrDie(FILE* fptr, const char* format, T* value) {
    int num_scanned = fscanf(fptr, format, value);
    if (num_scanned != 1) {
      LOG(FATAL) << "Invalid UW data file.";
    }
  }
  int num_cameras_;
  int num_points_;
  int num_observations_;
  int num_parameters_;
  int* point_index_;
  int* camera_index_;
  double* observations_;
  // The parameter vector is laid out as follows
  // [camera_1, ..., camera_n, point_1, ..., point_m]
  double* parameters_;
};

// Templated pinhole camera model for used with Ceres.  The camera is
// parameterized using 9 parameters: 3 for rotation, 3 for translation, 1 for
// focal length and 2 for radial distortion. The principal point is not modeled
// (i.e. it is assumed be located at the image center).
struct SnavelyReprojectionError {
  SnavelyReprojectionError(double observed_x, double observed_y, const double* camera)
      : observed_x(observed_x), observed_y(observed_y), camera(camera) {}
  template <typename T>
  bool operator()(const T* const point,
                  T* residuals) const {
    // camera[0,1,2] are the angle-axis rotation.
    T p[3];
    const T angle_axis[3] = {
      T(camera[0]),
      T(camera[1]),
      T(camera[2])
    };
    ceres::AngleAxisRotatePoint(angle_axis, point, p);
    // camera[3,4,5] are the translation.
    p[0] += T(camera[3]);
    p[1] += T(camera[4]);
    p[2] += T(camera[5]);
    // Compute the center of distortion. The sign change comes from
    // the camera model that Noah Snavely's Bundler assumes, whereby
    // the camera coordinate system has a negative z axis.
    T xp = -p[0] / p[2];
    T yp = -p[1] / p[2];
    // Apply second and fourth order radial distortion.
    const T& l1 = T(camera[7]);
    const T& l2 = T(camera[8]);
    T r2 = xp * xp + yp * yp;
    T distortion = 1.0 + r2 * (l1 + l2 * r2);
    // Compute final projected point position.
    const T& focal = T(camera[6]);
    T predicted_x = focal * distortion * xp;
    T predicted_y = focal * distortion * yp;
    // The error is the difference between the predicted and observed position.
    residuals[0] = predicted_x - observed_x;
    residuals[1] = predicted_y - observed_y;
    return true;
  }
  // Cost function assumes no camera intrinsics uncertainty.
  static ceres::CostFunction* Create(const double observed_x, const double observed_y, const double* camera) {
    return (new ceres::AutoDiffCostFunction<SnavelyReprojectionError, 2, 3>(
        new SnavelyReprojectionError(observed_x, observed_y, camera)));
  }

  double observed_x;
  double observed_y;
  const double* camera;
};

int main(int argc, char** argv) {
  google::InitGoogleLogging(argv[0]);
  if (argc != 2) {
    std::cerr << "usage: simple_bundle_adjuster <bal_problem>\n";
    return 1;
  }
  BAProblem ba_problem;
  if (!ba_problem.LoadFile(argv[1])) {
    std::cerr << "ERROR: unable to open file " << argv[1] << "\n";
    return 1;
  }
  const double* observations = ba_problem.observations();
  // Create residuals for each observation in the bundle adjustment problem. The
  // parameters for cameras and points are added automatically.
  ceres::Problem problem;
  for (int i = 0; i < ba_problem.num_observations(); ++i) {
    // Each Residual block takes a point as input and outputs a 2 dimensional residual. 
    // Internally, the cost function stores the observed image location and compares 
    // the reprojection against the observation.
    ceres::CostFunction* cost_function = SnavelyReprojectionError::Create(observations[2 * i + 0], observations[2 * i + 1], ba_problem.camera_for_observation(i));
    problem.AddResidualBlock(cost_function,
                             NULL /* squared loss */,
                             ba_problem.mutable_point_for_observation(i));
                            //  ba_problem.mutable_camera_for_observation(i));

  }
  // Make Ceres automatically detect the bundle structure. Note that the
  // standard solver, SPARSE_NORMAL_CHOLESKY, also works fine but it is slower
  // for standard bundle adjustment problems.
  ceres::Solver::Options options;
  options.linear_solver_type = ceres::DENSE_SCHUR;
  options.minimizer_progress_to_stdout = true;
  ceres::Solver::Summary summary;
  ceres::Solve(options, &problem, &summary);
  std::cout << summary.FullReport() << "\n";
  return 0;
}
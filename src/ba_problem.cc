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

#include "ba_problem.h"

#include <iostream>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <string>
#include <vector>

#include "Eigen/Core"
#include "ceres/rotation.h"
#include "glog/logging.h"
#include "random.h"

namespace ceres {
namespace {
typedef Eigen::Map<Eigen::VectorXd> VectorRef;
typedef Eigen::Map<const Eigen::VectorXd> ConstVectorRef;

template <typename T>
void FscanfOrDie(FILE* fptr, const char* format, T* value) {
  int num_scanned = fscanf(fptr, format, value);
  if (num_scanned != 1) {
    LOG(FATAL) << "Invalid UW data file.";
  }
}

void PerturbPoint3(const double sigma, double* point) {
  for (int i = 0; i < 3; ++i) {
    point[i] += RandNormal() * sigma;
  }
}

double Median(std::vector<double>* data) {
  int n = data->size();
  std::vector<double>::iterator mid_point = data->begin() + n / 2;
  std::nth_element(data->begin(), mid_point, data->end());
  return *mid_point;
}

}  // namespace

BAProblem::BAProblem(const std::string& filename) {
  FILE* fptr = fopen(filename.c_str(), "r");

  if (fptr == NULL) {
    LOG(FATAL) << "Error: unable to open file " << filename;
    return;
  };

  // This wil die horribly on invalid files. Them's the breaks.
  FscanfOrDie(fptr, "%d", &num_cameras_);
  FscanfOrDie(fptr, "%d", &num_points_);
  FscanfOrDie(fptr, "%d", &num_observations_);

  VLOG(1) << "Header: " << num_cameras_ << " " << num_points_ << " "
          << num_observations_;

  point_index_ = new int[num_observations_];
  camera_index_ = new int[num_observations_];
  observations_ = new double[2 * num_observations_];

  num_parameters_ = BAProblem::camera_block_size() * num_cameras_ + BAProblem::point_block_size() * num_points_;
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

  fclose(fptr);
}

// This function writes the problem to a file in the same format that
// is read by the constructor.
void BAProblem::WriteToFile(const std::string& filename) const {
  FILE* fptr = fopen(filename.c_str(), "w");

  if (fptr == NULL) {
    LOG(FATAL) << "Error: unable to open file " << filename;
    return;
  };

  fprintf(fptr, "%d %d %d\n", num_cameras_, num_points_, num_observations_);

  for (int i = 0; i < num_observations_; ++i) {
    fprintf(fptr, "%d %d", camera_index_[i], point_index_[i]);
    for (int j = 0; j < 2; ++j) {
      fprintf(fptr, " %g", observations_[2 * i + j]);
    }
    fprintf(fptr, "\n");
  }

  for (int i = 0; i < num_cameras(); ++i) {
    double angleaxis[BAProblem::camera_block_size()];
    memcpy(angleaxis, parameters_ + BAProblem::camera_block_size() * i, BAProblem::camera_block_size() * sizeof(double));
    for (int j = 0; j < BAProblem::camera_block_size(); ++j) {
      fprintf(fptr, " %.16g", angleaxis[j]);
    }
    fprintf(fptr, "\n");
  }

  const double* points = parameters_ + camera_block_size() * num_cameras_;
  for (int i = 0; i < num_points(); ++i) {
    const double* point = points + i * point_block_size();
    for (int j = 0; j < point_block_size(); ++j) {
      fprintf(fptr, " %.16g", point[j]);
    }
    fprintf(fptr, "\n");
  }

  fclose(fptr);
}

// Write the problem to a PLY file for inspection in Meshlab or CloudCompare.
void BAProblem::WriteToPLYFile(const std::string& filename) const {
  std::ofstream of(filename.c_str());

  of << "ply" << '\n'
     << "format ascii 1.0" << '\n'
     << "element vertex " << num_cameras_ + num_points_ << '\n'
     << "property float x" << '\n'
     << "property float y" << '\n'
     << "property float z" << '\n'
     << "property uchar red" << '\n'
     << "property uchar green" << '\n'
     << "property uchar blue" << '\n'
     << "end_header" << std::endl;

  // Export extrinsic data (i.e. camera centers) as green points.
  double angle_axis[3];
  double center[3];
  for (int i = 0; i < num_cameras(); ++i) {
    const double* camera = cameras() + camera_block_size() * i;
    CameraToAngleAxisAndCenter(camera, angle_axis, center);
    of << center[0] << ' ' << center[1] << ' ' << center[2] << " 0 255 0"
       << '\n';
  }

  // Export the structure (i.e. 3D Points) as white points.
  const double* points = parameters_ + camera_block_size() * num_cameras_;
  for (int i = 0; i < num_points(); ++i) {
    const double* point = points + i * point_block_size();
    for (int j = 0; j < point_block_size(); ++j) {
      of << point[j] << ' ';
    }
    of << "255 255 255\n";
  }
  of.close();
}

void BAProblem::CameraToAngleAxisAndCenter(const double* camera,
                                            double* angle_axis,
                                            double* center) const {
  VectorRef angle_axis_ref(angle_axis, 3);
  angle_axis_ref = ConstVectorRef(camera, 3);

  // c = -R't
  Eigen::VectorXd inverse_rotation = -angle_axis_ref;
  // AngleAxisRotatePoint(inverse_rotation.data(), camera + camera_block_size() - 6, center);
  AngleAxisRotatePoint(inverse_rotation.data(), camera + 3, center);
  VectorRef(center, 3) *= -1.0;
}

void BAProblem::AngleAxisAndCenterToCamera(const double* angle_axis,
                                            const double* center,
                                            double* camera) const {
  ConstVectorRef angle_axis_ref(angle_axis, 3);
  VectorRef(camera, 3) = angle_axis_ref;

  // t = -R * c
  AngleAxisRotatePoint(angle_axis, center, camera + 3);
  VectorRef(camera + 3, 3) *= -1.0;
}

void BAProblem::Normalize() {
  // Compute the marginal median of the geometry.
  std::vector<double> tmp(num_points_);
  Eigen::Vector3d median;
  double* points = mutable_points();
  for (int i = 0; i < 3; ++i) {
    for (int j = 0; j < num_points_; ++j) {
      tmp[j] = points[3 * j + i];
    }
    median(i) = Median(&tmp);
  }

  for (int i = 0; i < num_points_; ++i) {
    VectorRef point(points + 3 * i, 3);
    tmp[i] = (point - median).lpNorm<1>();
  }

  const double median_absolute_deviation = Median(&tmp);

  // Scale so that the median absolute deviation of the resulting
  // reconstruction is 100.
  const double scale = 100.0 / median_absolute_deviation;

  std::cout << "median: " << median.transpose() << std::endl;
  std::cout << "median absolute deviation: " << median_absolute_deviation << std::endl;
  std::cout << "scale: " << scale << std::endl;

  VLOG(2) << "median: " << median.transpose();
  VLOG(2) << "median absolute deviation: " << median_absolute_deviation;
  VLOG(2) << "scale: " << scale;

  // X = scale * (X - median)
  for (int i = 0; i < num_points_; ++i) {
    VectorRef point(points + 3 * i, 3);
    point = scale * (point - median);
  }

  double* cameras = mutable_cameras();
  double angle_axis[3];
  double center[3];
  for (int i = 0; i < num_cameras_; ++i) {
    double* camera = cameras + camera_block_size() * i;
    CameraToAngleAxisAndCenter(camera, angle_axis, center);
    // center = scale * (center - median)
    VectorRef(center, 3) = scale * (VectorRef(center, 3) - median);
    AngleAxisAndCenterToCamera(angle_axis, center, camera);
  }
}

void BAProblem::Perturb(const double rotation_sigma,
                         const double translation_sigma,
                         const double point_sigma) {
  CHECK_GE(point_sigma, 0.0);
  CHECK_GE(rotation_sigma, 0.0);
  CHECK_GE(translation_sigma, 0.0);

  double* points = mutable_points();
  if (point_sigma > 0) {
    for (int i = 0; i < num_points_; ++i) {
      PerturbPoint3(point_sigma, points + 3 * i);
    }
  }

  for (int i = 0; i < num_cameras_; ++i) {
    double* camera = mutable_cameras() + camera_block_size() * i;

    double angle_axis[3];
    double center[3];
    // Perturb in the rotation of the camera in the angle-axis
    // representation.
    CameraToAngleAxisAndCenter(camera, angle_axis, center);
    if (rotation_sigma > 0.0) {
      PerturbPoint3(rotation_sigma, angle_axis);
    }
    AngleAxisAndCenterToCamera(angle_axis, center, camera);

    if (translation_sigma > 0.0) {
      PerturbPoint3(translation_sigma, camera + camera_block_size() - 6);
    }
  }
}

BAProblem::~BAProblem() {
  delete[] point_index_;
  delete[] camera_index_;
  delete[] observations_;
  delete[] parameters_;
}

}  // namespace ceres

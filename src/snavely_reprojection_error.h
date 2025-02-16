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
// Templated struct implementing the camera model and residual
// computation for bundle adjustment used by Noah Snavely's Bundler
// SfM system. This is also the camera model/residual for the bundle
// adjustment problems in the BAL dataset. It is templated so that we
// can use Ceres's automatic differentiation to compute analytic
// jacobians.
//
// For details see: http://phototour.cs.washington.edu/bundler/
// and http://grail.cs.washington.edu/projects/bal/

#ifndef CERES_SNAVELY_REPROJECTION_ERROR_H_
#define CERES_SNAVELY_REPROJECTION_ERROR_H_

#include "ceres/rotation.h"

namespace ceres {

// Templated pinhole camera model for used with Ceres.  The camera is
// parameterized using 12 parameters: 3 for rotation, 3 for translation, 2 for
// focal length, 2 for radial distortion and 2 for the principal point.
struct SnavelyReprojectionError {
  SnavelyReprojectionError(double observed_x, double observed_y, double confidence)
      : observed_x(observed_x), observed_y(observed_y), confidence(confidence) {}

  template <typename T>
  bool operator()(const T* const camera,
                  const T* const point,
                  T* residuals) const {
    // camera[0,1,2] are the angle-axis rotation.
    T p[3];
    AngleAxisRotatePoint(camera, point, p);
    // camera[3,4,5] are the translation.
    p[0] += camera[3];
    p[1] += camera[4];
    p[2] += camera[5];

    const T xp = p[0] / p[2];
    const T yp = p[1] / p[2];

    // Apply second and fourth order radial distortion.
    const T& l1 = camera[8];
    const T& l2 = camera[9];
    // const T& l3 = camera[10];
    // const T& t1 = camera[11];
    // const T& t2 = camera[12];
    const T r2 = xp * xp + yp * yp;
    const T distortion = 1.0 + r2 * (l1 + l2 * r2);
    // const T distortion_r = 1.0 + r2 * (l1 + l2 * r2 + l3 * r2 * r2);
    // const T distortion_t_x = t1 * (r2 + 2 * xp * xp) + 2.0 * t2 * xp * yp;
    // const T distortion_t_y = t2 * (r2 + 2 * yp * yp) + 2.0 * t1 * xp * yp;
    // const T undistorded_x = distortion_r * xp + distortion_t_x;
    // const T undistorded_y = distortion_r * yp + distortion_t_y;
    // Compute final projected point position.
    const T& focal_x = camera[6];
    const T& focal_y = camera[7];
    const T& principal_x = camera[10];
    const T& principal_y = camera[11];
    const T predicted_x = focal_x * distortion * xp + principal_x;
    const T predicted_y = focal_y * distortion * yp + principal_y;
    // const T& principal_x = camera[13];
    // const T& principal_y = camera[14];
    // const T predicted_x = focal_x * undistorded_x + principal_x;
    // const T predicted_y = focal_y * undistorded_y + principal_y;

    // The error is the difference between the predicted and observed position.
    residuals[0] = (predicted_x - observed_x) * confidence;
    residuals[1] = (predicted_y - observed_y) * confidence;

    return true;
  }

  // Factory to hide the construction of the CostFunction object from
  // the client code.
  static ceres::CostFunction* Create(const double observed_x,
                                     const double observed_y,
                                     const double confidence) {
    return (new ceres::AutoDiffCostFunction<SnavelyReprojectionError, 2, 12, 3>(
        new SnavelyReprojectionError(observed_x, observed_y, confidence)));
  }

  double observed_x;
  double observed_y;
  double confidence;
};

// Templated pinhole camera model for used with Ceres.  The camera is
// parameterized using 12 parameters: 3 for rotation, 3 for translation, 2 for
// focal length, 2 for radial distortion and 2 for the principal point.
struct SnavelyReprojectionErrorNoCam {
  SnavelyReprojectionErrorNoCam(double observed_x, double observed_y, double confidence, const double* camera)
      : observed_x(observed_x), observed_y(observed_y), confidence(confidence), camera(camera) {}

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
    AngleAxisRotatePoint(angle_axis, point, p);

    // camera[3,4,5] are the translation.
    p[0] += T(camera[3]);
    p[1] += T(camera[4]);
    p[2] += T(camera[5]);

    const T xp = p[0] / p[2];
    const T yp = p[1] / p[2];

    // Apply second and fourth order radial distortion.
    const T l1 = T(camera[8]);
    const T l2 = T(camera[9]);
    const T r2 = xp * xp + yp * yp;
    const T distortion = 1.0 + r2 * (l1 + l2 * r2);

    // Compute final projected point position.
    const T focal_x = T(camera[6]);
    const T focal_y = T(camera[7]);
    const T principal_x = T(camera[10]);
    const T principal_y = T(camera[11]);
    const T predicted_x = focal_x * distortion * xp + principal_x;
    const T predicted_y = focal_y * distortion * yp + principal_y;

    // The error is the difference between the predicted and observed position.
    residuals[0] = (predicted_x - observed_x) * confidence;
    residuals[1] = (predicted_y - observed_y) * confidence;

    return true;
  }

  // Factory to hide the construction of the CostFunction object from
  // the client code.
  static ceres::CostFunction* Create(const double observed_x, 
                                     const double observed_y, 
                                     const double confidence, 
                                     const double* camera) {
    return (new ceres::AutoDiffCostFunction<SnavelyReprojectionErrorNoCam, 2, 3>(
        new SnavelyReprojectionErrorNoCam(observed_x, observed_y, confidence, camera)));
  }

  double observed_x;
  double observed_y;
  double confidence;
  const double* camera;
};

// ------------- WITHOUT CONFIDENCE ------------- 
struct SnavelyReprojectionErrorNoConfidence {
  SnavelyReprojectionErrorNoConfidence(double observed_x, double observed_y)
      : observed_x(observed_x), observed_y(observed_y) {}

  template <typename T>
  bool operator()(const T* const camera,
                  const T* const point,
                  T* residuals) const {
    // camera[0,1,2] are the angle-axis rotation.
    T p[3];
    AngleAxisRotatePoint(camera, point, p);

    // camera[3,4,5] are the translation.
    p[0] += camera[3];
    p[1] += camera[4];
    p[2] += camera[5];

    const T xp = p[0] / p[2];
    const T yp = p[1] / p[2];

    // Apply second and fourth order radial distortion.
    const T& l1 = camera[8];
    const T& l2 = camera[9];
    const T r2 = xp * xp + yp * yp;
    const T distortion = 1.0 + r2 * (l1 + l2 * r2);

    // Compute final projected point position.
    const T& focal_x = camera[6];
    const T& focal_y = camera[7];
    const T& principal_x = camera[10];
    const T& principal_y = camera[11];
    const T predicted_x = focal_x * distortion * xp + principal_x;
    const T predicted_y = focal_y * distortion * yp + principal_y;

    // The error is the difference between the predicted and observed position.
    residuals[0] = predicted_x - observed_x;
    residuals[1] = predicted_y - observed_y;

    return true;
  }

  // Factory to hide the construction of the CostFunction object from
  // the client code.
  static ceres::CostFunction* Create(const double observed_x, const double observed_y) {
    return (new ceres::AutoDiffCostFunction<SnavelyReprojectionErrorNoConfidence, 2, 12, 3>(
        new SnavelyReprojectionErrorNoConfidence(observed_x, observed_y)));
  }

  double observed_x;
  double observed_y;
};
struct SnavelyReprojectionErrorNoConfidenceNoCam {
  SnavelyReprojectionErrorNoConfidenceNoCam(double observed_x, double observed_y, const double* camera)
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
    AngleAxisRotatePoint(angle_axis, point, p);

    // camera[3,4,5] are the translation.
    p[0] += T(camera[3]);
    p[1] += T(camera[4]);
    p[2] += T(camera[5]);

    const T xp = p[0] / p[2];
    const T yp = p[1] / p[2];

    // Apply second and fourth order radial distortion.
    const T l1 = T(camera[8]);
    const T l2 = T(camera[9]);
    const T r2 = xp * xp + yp * yp;
    const T distortion = 1.0 + r2 * (l1 + l2 * r2);

    // Compute final projected point position.
    const T focal_x = T(camera[6]);
    const T focal_y = T(camera[7]);
    const T principal_x = T(camera[10]);
    const T principal_y = T(camera[11]);
    const T predicted_x = focal_x * distortion * xp + principal_x;
    const T predicted_y = focal_y * distortion * yp + principal_y;

    // The error is the difference between the predicted and observed position.
    residuals[0] = (predicted_x - observed_x);
    residuals[1] = (predicted_y - observed_y);

    return true;
  }

  // Factory to hide the construction of the CostFunction object from
  // the client code.
  static ceres::CostFunction* Create(const double observed_x, const double observed_y, const double* camera) {
    return (new ceres::AutoDiffCostFunction<SnavelyReprojectionErrorNoConfidenceNoCam, 2, 3>(
        new SnavelyReprojectionErrorNoConfidenceNoCam(observed_x, observed_y, camera)));
  }

  double observed_x;
  double observed_y;
  const double* camera;
};

// Templated pinhole camera model for used with Ceres.  The camera is
// parameterized using 12 parameters: 3 for rotation, 3 for translation, 2 for
// focal length, 2 for radial distortion and 2 for the principal point.
struct SnavelyReprojectionErrorNoPoint {
  SnavelyReprojectionErrorNoPoint(double observed_x, double observed_y, const double* point)
      : observed_x(observed_x), observed_y(observed_y), point(point) {}

  template <typename T>
  bool operator()(const T* const camera,
                  T* residuals) const {
    // camera[0,1,2] are the angle-axis rotation.
    T p[3];
    const T pt[3] = {
      T(point[0]),
      T(point[1]),
      T(point[2])
    };
    AngleAxisRotatePoint(camera, pt, p);
    // camera[3,4,5] are the translation.
    p[0] += T(camera[3]);
    p[1] += T(camera[4]);
    p[2] += T(camera[5]);

    const T xp = p[0] / p[2];
    const T yp = p[1] / p[2];

    // Apply second and fourth order radial distortion.
    const T l1 = T(camera[8]);
    const T l2 = T(camera[9]);
    const T r2 = xp * xp + yp * yp;
    const T distortion = 1.0 + r2 * (l1 + l2 * r2);

    // Compute final projected point position.
    const T focal_x = T(camera[6]);
    const T focal_y = T(camera[7]);
    const T principal_x = T(camera[10]);
    const T principal_y = T(camera[11]);
    const T predicted_x = focal_x * distortion * xp + principal_x;
    const T predicted_y = focal_y * distortion * yp + principal_y;

    // The error is the difference between the predicted and observed position.
    residuals[0] = predicted_x - observed_x;
    residuals[1] = predicted_y - observed_y;
    return true;
  }

  // Factory to hide the construction of the CostFunction object from
  // the client code.
  static ceres::CostFunction* Create(const double observed_x, 
                                     const double observed_y, 
                                     const double* point) {
    return (new ceres::AutoDiffCostFunction<SnavelyReprojectionErrorNoPoint, 2, 12>(
        new SnavelyReprojectionErrorNoPoint(observed_x, observed_y, point)));
  }

  double observed_x;
  double observed_y;
  const double* point;
};

}  // namespace ceres

#endif  // CERES_SNAVELY_REPROJECTION_ERROR_H_

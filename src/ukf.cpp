#include "ukf.h"
#include "Eigen/Dense"
#include <iostream>

using Eigen::MatrixXd;
using Eigen::VectorXd;

/**
 * Initializes Unscented Kalman filter
 */
UKF::UKF() {
	// set is initialized to false
	is_initialized_ = false;

	// initialize time
	time_us_ = 0;

   // if this is false, laser measurements will be ignored (except during init)
   use_laser_ = true;

   // if this is false, radar measurements will be ignored (except during init)
   use_radar_ = true;

   // initial state vector
   x_ = VectorXd(5);

   // actual and augmented state vector size
   n_x_ = 5;
   n_aug_ = n_x_ + 2;

   // lambda initialization
   lambda_ = 3 - n_aug_;
     
   /** Measurement Noise **/

   // Laser measurement noise standard deviation position1 in m
   std_laspx_ = 0.15;

   // Laser measurement noise standard deviation position2 in m
   std_laspy_ = 0.15;

   // Radar measurement noise standard deviation radius in m
   std_radr_ = 0.3;

   // Radar measurement noise standard deviation angle in rad
   std_radphi_ = 0.03;

   // Radar measurement noise standard deviation radius change in m/s
   std_radrd_ = 0.3;

   // Process covariance matrix
   P_ = MatrixXd(n_x_, n_x_);

   // Process noise standard deviation longitudinal acceleration in m/s^2
   std_a_ = 1.5;

   // Process noise standard deviation yaw acceleration in rad/s^2
   std_yawdd_ = 0.875;

   /** Process Noise initialization */
   P_.fill(0.0);
   P_(0, 0) = 0.5;
   P_(1, 1) = 0.5;
   P_(2, 2) = 0.25;
   P_(3, 3) = 0.6;
   P_(4, 4) = 0.2;
    
   /** Measurement Noise initialization */
   // Radar measurement covariance
   R_radar = MatrixXd(3, 3);
   R_radar.fill(0.0);

   // initialize diagonals
   R_radar(0, 0) = std_radr_ * std_radr_;
   R_radar(1, 1) = std_radphi_ * std_radphi_;
   R_radar(2, 2) = std_radrd_ * std_radrd_; 

   // Radar measurement covariance
   R_lidar = MatrixXd(2, 2);
   R_lidar.fill(0.0);

   // initialize diagonals
   R_lidar(0, 0) = std_laspx_ * std_laspx_;
   R_lidar(1, 1) = std_laspy_ * std_laspy_;

   /* UKF Parameters */
   // weights
   weights_ = VectorXd(2 * n_aug_ + 1);
   weights_.fill(0.5/(lambda_ + n_aug_));
   weights_(0) = lambda_/(lambda_ + n_aug_);

   // Sigma State Matrix
   Xsig_pred_ = MatrixXd(n_x_, 2 * n_aug_ + 1);

   /* Consistency check*/
   NIS_radar = 0;
   NIS_lidar = 0;

}

UKF::~UKF() {}

void UKF::ProcessMeasurement(MeasurementPackage meas_package) {
	// check if state matrix has been initialized
	if (!is_initialized_){
		// parameters
		double x = 0;
		double y = 0;
		double v = 0;
		double yaw = 0;
		double yaw_rate = 0;
		
		// initialize per measurement type
		if (meas_package.sensor_type_ == MeasurementPackage::RADAR){
			// radar returns [range, bearing, velocity in radial direction]
			double range = meas_package.raw_measurements_[0];
			double bearing =  meas_package.raw_measurements_[1];
			double rad_vel =  meas_package.raw_measurements_[2];

			// extract x, y, and velocity
			x = range * cos(bearing);
			y = range * sin(bearing);
			double vx = rad_vel * cos(bearing);
			double vy = rad_vel * sin(bearing);
			v = sqrt(vx * vx + vy * vy);

		} else {
			// lidar returns [x, y] position
			x = meas_package.raw_measurements_[0];
			y = meas_package.raw_measurements_[1];
		}

		// set initialized to true
		is_initialized_ = true;

		// insert into state vector
		x_ << x, y, v, yaw, yaw_rate;

		// exit
		return;
	}

	/* If previously initialized then we begin prediction and update steps*/
	// time update
	double delta_t = (meas_package.timestamp_ - time_us_) * 1e-6;
	time_us_ = meas_package.timestamp_;
	
	// call preduction function
	Prediction(delta_t);

	// call update based on sensor type
	if (meas_package.sensor_type_ == MeasurementPackage::RADAR && use_radar_){
		// update the radar
		UpdateRadar(meas_package);

	} 
	if (meas_package.sensor_type_ == MeasurementPackage::LASER && use_laser_){
		// update the lidar
		UpdateLidar(meas_package);
	}
	
}

void UKF::Prediction(double delta_t) {
	// Augmented spaceholders initialization
	VectorXd x_aug = VectorXd(n_aug_);
	MatrixXd P_aug = MatrixXd(n_aug_, n_aug_);
	MatrixXd Xsig_aug = MatrixXd(n_aug_, 2 * n_aug_ + 1);

	// fill in augmented mean with current state mean and noise parameters
	x_aug.head(5) = x_;
	x_aug(5) = 0;
	x_aug(6) = 0;

	P_aug.fill(0.0);
	P_aug.topLeftCorner(5,5) = P_;
	P_aug(5,5) = std_a_*std_a_;
	P_aug(6,6) = std_yawdd_*std_yawdd_;


	// create cholesky decomp
	MatrixXd chol = P_aug.llt().matrixL();
	// bucket zero is the current state predicition
	Xsig_aug.col(0) = x_aug;

	// aug points create
	for (int index = 0; index < n_aug_; index++){
		Xsig_aug.col(index + 1) = x_aug + sqrt(lambda_ + n_aug_)*chol.col(index);
		Xsig_aug.col(index + 1 + n_aug_) = x_aug - sqrt(lambda_ + n_aug_)*chol.col(index);
	}
	
	/* Prediction based on augmented values*/
	for (int index = 0; index < 2 * n_aug_ + 1; index++){
		// initialize values
		double p_x = Xsig_aug(0, index);
		double p_y = Xsig_aug(1, index);
		double v = Xsig_aug(2, index);
		double yaw = Xsig_aug(3, index);
		double yaw_r = Xsig_aug(4, index);
		double nu_a = Xsig_aug(5, index);
		double nu_yaw_r = Xsig_aug(6, index);

		// predicted values 
		double px_p, py_p, v_p, yaw_p, yaw_r_p;

		/* Without noise */
		// equation for velocity different with straight line motion (yaw rate = 0)
		if (fabs(yaw_r) > 1e-6){
			px_p = p_x + (v/ yaw_r) *(sin(yaw + yaw_r * delta_t) - sin(yaw));
			py_p = p_y + (v/ yaw_r) *(-cos(yaw + yaw_r * delta_t) + cos(yaw));

		} else {
			px_p = p_x + v * delta_t * cos(yaw);
			py_p = p_y + v * delta_t * sin(yaw);
		}
		// velocity, heading, turn rate
		v_p = v;
		yaw_p = yaw + yaw_r*delta_t;
		yaw_r_p = yaw_r;

		/* Adding noise */
		px_p = px_p + 0.5 * nu_a * delta_t * delta_t * cos(yaw);
		py_p = py_p + 0.5 * nu_a * delta_t * delta_t * sin(yaw);
		v_p = v_p + nu_a * delta_t;
		yaw_p = yaw_p + 0.5 * nu_yaw_r * delta_t * delta_t;
		yaw_r_p = yaw_r_p + nu_yaw_r * delta_t;
		

		/* Add to predicted sigma points */
     	Xsig_pred_(0, index) = px_p;
     	Xsig_pred_(1, index) = py_p;
     	Xsig_pred_(2, index) = v_p;
     	Xsig_pred_(3, index) = yaw_p;
     	Xsig_pred_(4, index) = yaw_r_p;
     	
	}
	
	/* Predict new state mean */
	//zero out current
	x_.fill(0.0);

	for(int index = 0; index < 2 * n_aug_ + 1; index++){
		x_ = x_ + weights_(index) * Xsig_pred_.col(index);
	}
	
	/* Predict new state covariance */
	P_.fill(0.0);
	for(int index=0; index < 2 * n_aug_ + 1; index++){
		// calculate difference predicted state and each predicted column
		VectorXd diff = Xsig_pred_.col(index) - x_;

		// ensure angle between [-pi, pi]
		while(diff(3) > M_PI) diff(3) -= 2.*M_PI;
        while(diff(3) < -M_PI) diff(3) += 2.*M_PI;
        P_ = P_ + weights_(index) * diff * diff.transpose();
	}
	
	/* END OF PREDICTION MODULE*/
}

void UKF::UpdateLidar(MeasurementPackage meas_package) {
	// extract measurement
	VectorXd z_ = meas_package.raw_measurements_;

	// create sigma matrix -> only x and y components
	int n_z_ = 2;
	MatrixXd Zsig_ = MatrixXd(n_z_, 2*n_aug_+1);

	// each weight holds the predicted estimate
	for(int index = 0; index < 2*n_aug_+1; index++){
		Zsig_(0, index) = Xsig_pred_(0, index);
		Zsig_(1, index) = Xsig_pred_(1, index);
	}

	// predict the mean 
	VectorXd z_pred_ = VectorXd(n_z_);
	z_pred_.fill(0.0);
	for (int index = 0; index < 2 * n_aug_ + 1; index++){
		z_pred_ = z_pred_ + weights_(index) * Zsig_.col(index);
	}

	// calculate the covariance of measurement
	MatrixXd Lidar_conv = MatrixXd(n_z_, n_z_);
	Lidar_conv.fill(0.0);
	for(int index=0; index < 2 * n_aug_ + 1; index++){
		// calculate difference predicted state and each predicted column
		VectorXd diff = Zsig_.col(index) - z_pred_;

        Lidar_conv = Lidar_conv + weights_(index) * diff * diff.transpose();
	}

	// add noise
	Lidar_conv = Lidar_conv + R_lidar;

	// Calculate the cross correlation matrix
	MatrixXd Tc = MatrixXd(n_x_, n_z_);
	Tc.fill(0.0);

	for(int index=0; index < 2 * n_aug_ + 1; index++){
		// calculate difference predicted state and each predicted column
		VectorXd diff = Xsig_pred_.col(index) - x_;
		VectorXd l_diff = Zsig_.col(index) - z_pred_;

		// corellation matrix
		Tc = Tc + weights_(index) * diff * l_diff.transpose();
	}

	// calculate the kalman gain
	MatrixXd K = Tc * Lidar_conv.inverse();

	// updare state mean and covariance
	VectorXd z_diff = z_ - z_pred_;
	x_ = x_ + K*z_diff;
   	P_ = P_ - K*Lidar_conv*K.transpose();

   	//calculate NIS -> probably put it as a vector so we can tally later?
   	NIS_lidar = z_diff.transpose() * Lidar_conv.inverse() * z_diff;

   	//if (NIS_lidar > )

}

void UKF::UpdateRadar(MeasurementPackage meas_package) {
  // extract measurement
	VectorXd z_ = meas_package.raw_measurements_;

	// create sigma matrix -> only x and y components
	int n_z_ = 3;
	MatrixXd Zsig_ = MatrixXd(n_z_, 2*n_aug_+1);

	// each weight holds the predicted estimate
	for(int index = 0; index < 2*n_aug_+1; index++){
		// extract values for better readability
	    double p_x = Xsig_pred_(0,index);
	    double p_y = Xsig_pred_(1,index);
	    double v  = Xsig_pred_(2,index);
	    double yaw = Xsig_pred_(3,index);

	    double v1 = cos(yaw)*v;
	    double v2 = sin(yaw)*v;

	    // measurement model
	    Zsig_(0,index) = sqrt(p_x*p_x + p_y*p_y);                       // r
	    Zsig_(1,index) = atan2(p_y,p_x);                                // phi
	    Zsig_(2,index) = (p_x*v1 + p_y*v2) / sqrt(p_x*p_x + p_y*p_y);   // r_dot
	}

	// predict the mean 
	VectorXd z_pred_ = VectorXd(n_z_);
	z_pred_.fill(0.0);
	for (int index = 0; index < 2 * n_aug_ + 1; index++){
		z_pred_ = z_pred_ + weights_(index) * Zsig_.col(index);
	}

	// calculate the covariance of measurement
	MatrixXd Radar_conv = MatrixXd(n_z_, n_z_);
	Radar_conv.fill(0.0);
	for(int index=0; index < 2 * n_aug_ + 1; index++){
		// calculate difference predicted state and each predicted column
		VectorXd diff = Zsig_.col(index) - z_pred_;

		// keep angle bounded
	    while (diff(1)> M_PI) diff(1)-=2.*M_PI;
	    while (diff(1)<-M_PI) diff(1)+=2.*M_PI;

        Radar_conv = Radar_conv + weights_(index) * diff * diff.transpose();
	}

	// add noise
	Radar_conv = Radar_conv + R_radar;

	// Calculate the cross correlation matrix
	MatrixXd Tc = MatrixXd(n_x_, n_z_);
	Tc.fill(0.0);

	for(int index=0; index < 2 * n_aug_ + 1; index++){
		// calculate difference predicted state and each predicted column
		VectorXd diff = Xsig_pred_.col(index) - x_;
		// keep angle bounded
	    while (diff(1)> M_PI) diff(1)-=2.*M_PI;
	    while (diff(1)<-M_PI) diff(1)+=2.*M_PI;

		VectorXd l_diff = Zsig_.col(index) - z_pred_;
		// keep angle bounded
	    while (l_diff(1)> M_PI) l_diff(1)-=2.*M_PI;
	    while (l_diff(1)<-M_PI) l_diff(1)+=2.*M_PI;

		// corellation matrix
		Tc = Tc + weights_(index) * diff * l_diff.transpose();
	}

	// calculate the kalman gain
	MatrixXd K = Tc * Radar_conv.inverse();

	// updare state mean and covariance
	VectorXd z_diff = z_ - z_pred_;
	
	// keep angle bounded
	while (z_diff(1)> M_PI) z_diff(1)-=2.*M_PI;
	while (z_diff(1)<-M_PI) z_diff(1)+=2.*M_PI;
	
	x_ = x_ + K*z_diff;
   	P_ = P_ - K*Radar_conv*K.transpose();

   	//calculate NIS -> probably put it as a vector so we can tally later?
   	NIS_radar = z_diff.transpose() * Radar_conv.inverse() * z_diff;
}
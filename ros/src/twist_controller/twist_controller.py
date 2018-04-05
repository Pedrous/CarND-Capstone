import pid
import yaw_controller
import lowpass
import rospy

GAS_DENSITY = 2.858
ONE_MPH = 0.44704


class Controller(object):
    def __init__(self, wheel_base, steer_ratio, max_lat_accel, max_steer_angle, vehicle_mass, wheel_radius, decel_limit, accel_limit):
        # TODO: Implement
        self.pid = pid.PID(0.5, 0.1, 0.0, 0.0, 0.8)
        self.yaw_control = yaw_controller.YawController(wheel_base, steer_ratio, 0.1, max_lat_accel, max_steer_angle)
        self.vehicle_mass = vehicle_mass
        self.wheel_radius = wheel_radius
        self.filter = lowpass.LowPassFilter(0.5, 0.02)
        self.max_brake = decel_limit
        self.max_acc = accel_limit
        
        self.last_time = rospy.get_time()
        self.last_vel = 0
        
    def control(self, cmd_linear_vel, cmd_angular_vel, current_linear_vel, enable):
        # TODO: Change the arg, kwarg list to suit your needs
        # Return throttle, brake, steer
        print("control")
        if not enable:
          self.pid.reset()
          print("not enabled")
          return 0.0, 0.0, 0.0

        current_linear_vel = self.filter.filt(current_linear_vel)
        
        vel_error = cmd_linear_vel - current_linear_vel
        self.last_vel = current_linear_vel
        
        current_time = rospy.get_time()
        sample_time = current_time - self.last_time
        self.last_time = current_time
        
        brake_moment = 0
        vel_control = self.pid.step(vel_error, sample_time)
        
        if cmd_linear_vel == 0.0 and current_linear_vel < 0.1:
          vel_control = 0
          brake_moment = 400
        
        elif vel_control < 0.1 and vel_error < 0.0:
          vel_control = 0
          brake_acc = max(vel_error, self.max_brake)
          brake_moment = abs(brake_acc)*self.vehicle_mass*self.wheel_radius*brake_acc
          vel_control = 0
          
          
        steering = self.yaw_control.get_steering(cmd_linear_vel, cmd_angular_vel, current_linear_vel)
        
        return vel_control, brake_moment, steering

cmake_minimum_required(VERSION 3.8)
project(arm_exchange_interfaces)

find_package(ament_cmake REQUIRED)
find_package(rosidl_default_generators REQUIRED)
find_package(std_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(trajectory_msgs REQUIRED)

rosidl_generate_interfaces(${PROJECT_NAME}
  "msg/ArmCtrlAdvanceMsg.msg"
  "msg/ArmCtrlGimbalControlMsg.msg"
  "msg/ArmCtrlQMsg.msg"
  "msg/ArmCtrlStartMsg.msg"
  "msg/ArmCtrlWithdrawMsg.msg"
  "msg/ArmCtrlXYZMsg.msg"
  "msg/ArmFeedforwardWrenchMsg.msg"
  "msg/ArmHost2MCUMsg.msg"
  "msg/ArmMCU2HostMsg.msg"
  "msg/ApproachPlanningRequest.msg"
  "msg/ApproachPlanningResult.msg"
  "msg/ExchangeStationState.msg"
  "msg/OperatorInputState.msg"
  "msg/RecoveryPlanningRequest.msg"
  "msg/RecoveryPlanningResult.msg"
  "msg/Type3PlanningRequest.msg"
  "msg/Type3PlanningResult.msg"
  DEPENDENCIES std_msgs geometry_msgs sensor_msgs trajectory_msgs
)

ament_package()

(define (problem drone_navigate_26)
  (:domain drone-warehouse)
  
  (:objects
    drone1 - drone
    frisbee_26 - package
    grid_0_top_left - location
    grid_1_top_center - location
    grid_2_top_right - location
    grid_3_middle_left - location
    grid_4_center - location
    grid_5_middle_right - location
    grid_6_bottom_left - location
    grid_7_bottom_center - location
    grid_8_bottom_right - location
  )

  (:init
    (at drone1 grid_0_top_left)
    (at-pkg frisbee_26 grid_1_top_center)
    (free drone1)
  )

  (:goal
    (at drone1 grid_1_top_center)
  )
)

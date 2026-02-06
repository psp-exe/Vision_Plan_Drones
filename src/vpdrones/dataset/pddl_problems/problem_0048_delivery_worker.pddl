(define (problem drone_delivery_48)
  (:domain drone-warehouse)
  
  (:objects
    drone1 - drone
    worker_48 - package
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
    (at drone1 grid_5_middle_right)
    (at-pkg worker_48 grid_6_bottom_left)
    (free drone1)
  )

  (:goal
    (and
      (at-pkg worker_48 grid_0_top_left)
      (delivered worker_48)
    )
  )
)

(define (problem drone_delivery_3)
  (:domain drone-warehouse)
  
  (:objects
    drone1 - drone
    worker_3 - package
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
    (at drone1 grid_7_bottom_center)
    (at-pkg worker_3 grid_5_middle_right)
    (free drone1)
  )

  (:goal
    (and
      (at-pkg worker_3 grid_2_top_right)
      (delivered worker_3)
    )
  )
)

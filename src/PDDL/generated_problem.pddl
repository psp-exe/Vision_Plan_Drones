;; Auto-generated PDDL Problem
;; Generated: 2026-02-03T10:36:06.102128
;; Targets detected: 1

(define (problem vlm_generated_20260203_103606)
  (:domain warehouse-drone)

  (:objects
    drone1 - drone
    haz_container_01 - target
    charging_station - zone
  )

  (:init
    (landed drone1)
    (calibrated drone1)
    (docked drone1)
    (at-zone drone1 charging_station)

    ;; Drone position
    (= (x drone1) 0.0)
    (= (y drone1) 0.0)
    (= (z drone1) 0.0)

    ;; Battery parameters
    (= (battery-level drone1) 95.0)
    (= (battery-capacity drone1) 100.0)
    (= (discharge-rate-fly drone1) 0.5)
    (= (discharge-rate-hover drone1) 0.2)
    (= (recharge-rate drone1) 1.0)

    ;; Kinematics
    (= (fly-speed drone1) 2.0)

    ;; Operational thresholds
    (= (scan-range) 2.0)
    (= (min-battery) 20.0)
    (= (takeoff-altitude) 1.5)

    ;; Target: haz_container_01 (VLM detected)
    (= (tx haz_container_01) 10.5)
    (= (ty haz_container_01) 2.0)
    (= (tz haz_container_01) 3.5)
    (= (distance-to drone1 haz_container_01) 11.25)

    ;; Zone: charging_station
    (= (zone-x charging_station) 0.0)
    (= (zone-y charging_station) 0.0)
    (= (zone-z charging_station) 0.0)
    (= (distance-to-zone drone1 charging_station) 0.0)
  )

  (:goal
    (and
      (scanned haz_container_01)
      (landed drone1)
    )
  )

  (:metric minimize (total-time))
)
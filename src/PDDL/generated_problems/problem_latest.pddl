;; Auto-generated PDDL Problem
;; Generated: 2026-02-12T14:24:56.234069
;; Targets detected: 1

(define (problem vlm_generated_20260212_142456)
  (:domain warehouse-drone)

  (:objects
    drone1 - drone
    delivery_zone_001 - target
  )

  (:init
    (landed drone1)

    ;; Drone initial position
    (= (x drone1) 0.0)
    (= (y drone1) 0.0)
    (= (z drone1) 0.0)

    ;; Target: delivery_zone_001
    (= (tx delivery_zone_001) 0.7)
    (= (ty delivery_zone_001) -1.28)
    (= (tz delivery_zone_001) 1.65)

  )

  (:goal
    (and
      (scanned delivery_zone_001)
      (landed drone1)
    )
  )

  (:metric minimize (total-time))
)
(define (domain warehouse-drone)
  (:requirements :typing :durative-actions :numeric-fluents)

  (:types drone target)

  (:predicates
    (airborne ?d - drone)
    (landed ?d - drone)
    (scanned ?t - target)
    (near ?d - drone ?t - target)
  )

  (:functions
    ;; Drone 3D position
    (x ?d - drone)
    (y ?d - drone)
    (z ?d - drone)
    ;; Target 3D position
    (tx ?t - target)
    (ty ?t - target)
    (tz ?t - target)
  )

  ;; take_off: Drone goes from landed to airborne
  (:durative-action take_off
    :parameters (?d - drone)
    :duration (= ?duration 5)
    :condition (at start (landed ?d))
    :effect (and
      (at start (not (landed ?d)))
      (at end (airborne ?d))
      (at end (assign (z ?d) 1.5))
    )
  )

  ;; fly_to_target: Drone flies to a target location
  (:durative-action fly_to_target
    :parameters (?d - drone ?t - target)
    :duration (= ?duration 5)
    :condition (at start (airborne ?d))
    :effect (and
      (at end (near ?d ?t))
      (at end (assign (x ?d) (tx ?t)))
      (at end (assign (y ?d) (ty ?t)))
      (at end (assign (z ?d) (tz ?t)))
    )
  )

  ;; scan_target: Drone scans a nearby target
  (:durative-action scan_target
    :parameters (?d - drone ?t - target)
    :duration (= ?duration 4)
    :condition (and
      (at start (airborne ?d))
      (at start (near ?d ?t))
    )
    :effect (at end (scanned ?t))
  )

  ;; land: Drone lands at current position
  (:durative-action land
    :parameters (?d - drone)
    :duration (= ?duration 5)
    :condition (at start (airborne ?d))
    :effect (and
      (at start (not (airborne ?d)))
      (at end (landed ?d))
      (at end (assign (z ?d) 0.0))
    )
  )
)

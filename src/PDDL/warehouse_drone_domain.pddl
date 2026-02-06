;; =============================================================================
;; Domain: Warehouse Drone Navigation (Continuous Space)
;; Version: 1.0
;; Features: PDDL 2.1, Durative Actions, Numeric Fluents, Typing
;; Based on: "Drone Navigation PDDL Domain Design" technical report
;; =============================================================================
;; 
;; This domain enables autonomous drone navigation in warehouse environments
;; WITHOUT predefined waypoints. Locations are dynamically generated from
;; Vision Language Model (VLM) perception in real-time.
;;
;; Key Design Principles:
;; 1. Continuous Space: 3D coordinates (x, y, z) as numeric fluents
;; 2. Late-Binding Symbols: Targets created at runtime by VLM
;; 3. Resource Management: Battery consumption modeled per meter/second
;; 4. Temporal Planning: Durative actions with duration dependent on distance
;; =============================================================================

(define (domain warehouse-drone)

  (:requirements 
    :strips 
    :typing 
    :fluents 
    :durative-actions 
    :negative-preconditions
    :equality
  )

  ;; =========================================================================
  ;; TYPES
  ;; =========================================================================
  (:types
    drone                    ;; The UAV agent
    target - object          ;; Anything to inspect/visit (VLM-detected objects)
    zone                     ;; Logical zones (e.g., charging_station)
  )

  ;; =========================================================================
  ;; PREDICATES (Boolean state variables)
  ;; =========================================================================
  (:predicates
    (airborne ?d - drone)              ;; Is the drone flying?
    (landed ?d - drone)                ;; Is the drone on the ground?
    (scanned ?t - target)              ;; Has the target been inspected?
    (docked ?d - drone)                ;; Is the drone in the charger?
    (calibrated ?d - drone)            ;; Is the vision system ready?
    (at-zone ?d - drone ?z - zone)     ;; Drone is at a logical zone
  )

  ;; =========================================================================
  ;; FUNCTIONS (Numeric fluents for continuous state)
  ;; =========================================================================
  (:functions
    ;; 3D Coordinates of the drone (Continuous State)
    (x ?d - drone)                     ;; Current X position (meters)
    (y ?d - drone)                     ;; Current Y position (meters)
    (z ?d - drone)                     ;; Current Z position (altitude, meters)
    
    ;; 3D Coordinates of targets (Set by VLM perception at runtime)
    (tx ?t - target)                   ;; Target X position
    (ty ?t - target)                   ;; Target Y position
    (tz ?t - target)                   ;; Target Z position
    
    ;; Zone coordinates (for return-to-base, charging)
    (zone-x ?z - zone)                 ;; Zone center X
    (zone-y ?z - zone)                 ;; Zone center Y
    (zone-z ?z - zone)                 ;; Zone center Z
    
    ;; Battery Management
    (battery-level ?d - drone)         ;; Current charge (percentage 0-100)
    (battery-capacity ?d - drone)      ;; Maximum charge capacity (%)
    (discharge-rate-fly ?d - drone)    ;; Battery usage per meter flown
    (discharge-rate-hover ?d - drone)  ;; Battery usage per second hovering
    (recharge-rate ?d - drone)         ;; Battery gain per second docked
    
    ;; Kinematics
    (fly-speed ?d - drone)             ;; Speed in m/s
    (distance-to ?d - drone ?t - target) ;; Euclidean distance (precomputed by VLM)
    (distance-to-zone ?d - drone ?z - zone) ;; Distance to zone center
    
    ;; Operational Thresholds (Constants)
    (scan-range)                       ;; Max distance to inspect an object (meters)
    (min-battery)                      ;; Safety threshold for operations (%)
    (takeoff-altitude)                 ;; Default takeoff height (meters)
  )

  ;; =========================================================================
  ;; ACTION: take_off
  ;; =========================================================================
  ;; Description: Transitions the drone from ground to hover state.
  ;;              Sets Z-coordinate to safe flight altitude.
  ;; Duration: Fixed 5 seconds for stability
  ;; =========================================================================
  (:durative-action take_off
    :parameters (?d - drone)
    :duration (= ?duration 5)
    :condition (and 
      (at start (landed ?d))
      (at start (>= (battery-level ?d) (min-battery)))
      (at start (calibrated ?d))
    )
    :effect (and 
      (at start (not (landed ?d)))
      (at end (airborne ?d))
      (at end (assign (z ?d) (takeoff-altitude)))
      (at end (decrease (battery-level ?d) 2))
    )
  )

  ;; =========================================================================
  ;; ACTION: fly_to_target
  ;; =========================================================================
  ;; Description: Moves drone to the 3D coordinates of a target object.
  ;;              Duration and battery cost are functions of distance.
  ;; Note: 'distance-to' is populated by the VLM/Perception layer.
  ;; =========================================================================
  (:durative-action fly_to_target
    :parameters (?d - drone ?t - target)
    :duration (= ?duration (/ (distance-to ?d ?t) (fly-speed ?d)))
    :condition (and 
      (over all (airborne ?d))
      (at start (>= (battery-level ?d) (min-battery)))
      ;; Battery check: Must have enough to reach target + safety margin
      (at start (>= (battery-level ?d) 
                    (* (distance-to ?d ?t) (discharge-rate-fly ?d))))
    )
    :effect (and 
      ;; Update Coordinates at end (Discrete update of continuous state)
      (at end (assign (x ?d) (tx ?t)))
      (at end (assign (y ?d) (ty ?t)))
      (at end (assign (z ?d) (tz ?t)))
      ;; Consume Battery linearly with distance
      (at end (decrease (battery-level ?d) 
                        (* (distance-to ?d ?t) (discharge-rate-fly ?d))))
    )
  )

  ;; =========================================================================
  ;; ACTION: fly_to_zone
  ;; =========================================================================
  ;; Description: Navigate to a logical zone (e.g., charging station)
  ;; =========================================================================
  (:durative-action fly_to_zone
    :parameters (?d - drone ?z - zone)
    :duration (= ?duration (/ (distance-to-zone ?d ?z) (fly-speed ?d)))
    :condition (and 
      (over all (airborne ?d))
      (at start (>= (battery-level ?d) (min-battery)))
      (at start (>= (battery-level ?d) 
                    (* (distance-to-zone ?d ?z) (discharge-rate-fly ?d))))
    )
    :effect (and 
      (at end (assign (x ?d) (zone-x ?z)))
      (at end (assign (y ?d) (zone-y ?z)))
      (at end (assign (z ?d) (zone-z ?z)))
      (at end (at-zone ?d ?z))
      (at end (decrease (battery-level ?d) 
                        (* (distance-to-zone ?d ?z) (discharge-rate-fly ?d))))
    )
  )

  ;; =========================================================================
  ;; ACTION: scan_target
  ;; =========================================================================
  ;; Description: Capture high-resolution imagery of target. Requires proximity.
  ;; Duration: 4 seconds for image acquisition
  ;; =========================================================================
  (:durative-action scan_target
    :parameters (?d - drone ?t - target)
    :duration (= ?duration 4)
    :condition (and 
      (over all (airborne ?d))
      ;; Geographic Constraint: Must be close to target to scan
      (at start (<= (distance-to ?d ?t) (scan-range)))
      (at start (>= (battery-level ?d) 5))
    )
    :effect (and 
      (at end (scanned ?t))
      ;; Hovering consumes battery based on time
      (at end (decrease (battery-level ?d) 
                        (* 4 (discharge-rate-hover ?d))))
    )
  )

  ;; =========================================================================
  ;; ACTION: land
  ;; =========================================================================
  ;; Description: Safe return to ground state.
  ;; Duration: 5 seconds for controlled descent
  ;; =========================================================================
  (:durative-action land
    :parameters (?d - drone)
    :duration (= ?duration 5)
    :condition (and 
      (at start (airborne ?d))
    )
    :effect (and 
      (at start (not (airborne ?d)))
      (at end (landed ?d))
      (at end (assign (z ?d) 0))
      (at end (decrease (battery-level ?d) 1))
    )
  )

  ;; =========================================================================
  ;; ACTION: dock
  ;; =========================================================================
  ;; Description: Connect to charging station. Requires being at charging zone.
  ;; =========================================================================
  (:durative-action dock
    :parameters (?d - drone ?z - zone)
    :duration (= ?duration 3)
    :condition (and 
      (at start (landed ?d))
      (at start (at-zone ?d ?z))
    )
    :effect (and 
      (at end (docked ?d))
    )
  )

  ;; =========================================================================
  ;; ACTION: undock
  ;; =========================================================================
  ;; Description: Disconnect from charging station.
  ;; =========================================================================
  (:durative-action undock
    :parameters (?d - drone)
    :duration (= ?duration 2)
    :condition (and 
      (at start (docked ?d))
    )
    :effect (and 
      (at end (not (docked ?d)))
    )
  )

  ;; =========================================================================
  ;; ACTION: recharge
  ;; =========================================================================
  ;; Description: Refills battery. Requires being docked at charging station.
  ;; Duration: Proportional to battery deficit
  ;; =========================================================================
  (:durative-action recharge
    :parameters (?d - drone)
    :duration (= ?duration (/ (- (battery-capacity ?d) (battery-level ?d)) 
                              (recharge-rate ?d)))
    :condition (and 
      (over all (docked ?d))
      (at start (landed ?d))
    )
    :effect (and 
      (at end (assign (battery-level ?d) (battery-capacity ?d)))
    )
  )

  ;; =========================================================================
  ;; ACTION: calibrate_sensors
  ;; =========================================================================
  ;; Description: Initialize vision system before flight.
  ;; =========================================================================
  (:durative-action calibrate_sensors
    :parameters (?d - drone)
    :duration (= ?duration 3)
    :condition (and 
      (at start (landed ?d))
    )
    :effect (and 
      (at end (calibrated ?d))
    )
  )

)

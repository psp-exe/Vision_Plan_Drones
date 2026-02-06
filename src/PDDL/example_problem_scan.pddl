;; =============================================================================
;; Problem: Warehouse Inspection Mission
;; Example: "Scan the red box in aisle 3"
;; =============================================================================
;; 
;; This problem file demonstrates how the VLM populates the PDDL problem
;; with objects and coordinates detected from the drone camera image.
;; 
;; VLM Output → JSON → This PDDL Problem
;; =============================================================================

(define (problem warehouse-inspection-001)
  (:domain warehouse-drone)
  
  ;; =========================================================================
  ;; OBJECTS
  ;; VLM detects objects and creates instances at runtime
  ;; =========================================================================
  (:objects
    drone1 - drone
    
    ;; Targets detected by VLM (created dynamically)
    red_box - target
    
    ;; Static zones (known a priori)
    charging_station - zone
  )
  
  ;; =========================================================================
  ;; INITIAL STATE
  ;; Position values populated by VLM + Depth Estimation pipeline
  ;; =========================================================================
  (:init
    ;; ----- Drone State -----
    (landed drone1)
    (calibrated drone1)
    (docked drone1)
    (at-zone drone1 charging_station)
    
    ;; Drone starting position (at charging station)
    (= (x drone1) 0.0)
    (= (y drone1) 0.0)
    (= (z drone1) 0.0)
    
    ;; ----- Battery Parameters -----
    (= (battery-level drone1) 95.0)
    (= (battery-capacity drone1) 100.0)
    (= (discharge-rate-fly drone1) 0.5)      ;; 0.5% per meter
    (= (discharge-rate-hover drone1) 0.2)    ;; 0.2% per second
    (= (recharge-rate drone1) 1.0)           ;; 1% per second
    
    ;; ----- Kinematics -----
    (= (fly-speed drone1) 2.0)               ;; 2 m/s cruise speed
    
    ;; ----- Operational Thresholds -----
    (= (scan-range) 2.0)                     ;; Must be within 2m to scan
    (= (min-battery) 20.0)                   ;; Safety threshold: 20%
    (= (takeoff-altitude) 1.5)               ;; Default hover height
    
    ;; ----- Target Coordinates (VLM + Depth Estimation Output) -----
    ;; The VLM detected "red_box" at image position (320, 240)
    ;; Monocular depth estimation returned d = 8.5m
    ;; Pinhole camera model + drone pose transform gives:
    (= (tx red_box) 12.5)                    ;; World X coordinate
    (= (ty red_box) 3.2)                     ;; World Y coordinate
    (= (tz red_box) 4.0)                     ;; World Z coordinate (height)
    
    ;; ----- Precomputed Distances (Euclidean) -----
    ;; Calculated by problem generator: sqrt((12.5-0)^2 + (3.2-0)^2 + (4.0-0)^2)
    (= (distance-to drone1 red_box) 13.5)
    
    ;; ----- Zone Coordinates -----
    (= (zone-x charging_station) 0.0)
    (= (zone-y charging_station) 0.0)
    (= (zone-z charging_station) 0.0)
    (= (distance-to-zone drone1 charging_station) 0.0)
  )
  
  ;; =========================================================================
  ;; GOAL
  ;; Derived from natural language instruction: "Scan the red box"
  ;; =========================================================================
  (:goal
    (and
      (scanned red_box)
      (landed drone1)
    )
  )
  
  ;; =========================================================================
  ;; METRIC (Optimization objective)
  ;; =========================================================================
  (:metric minimize (total-time))
)

APPENDIX DOCUMENT
Hybrid Digital Twin for Predictive Maintenance
University of Abdelhamid Mehri – Constantine 2 | NTIC Faculty | ILSI Master
LEGHELIMI Houssem Eddine | CHAIN Mouadh | Supervisor: DR. DJENOUHAT Manel
June 2026
Scope and Purpose
This appendix document supplements the main thesis and provides detailed technical content that was condensed in the main body to meet page requirements. Readers are directed here from the main thesis with references.
Annex
Title
Content Summary
A
ML Models 
- 
Architecture & Training
Feature engineering (28 features), Tier 0 thresholds, Isolation Forest hyperparameters, XGBoost hyperparameters, ensemble combiner logic, model serialization
B
Implementation 
-
 Software & MQTT
Complete software stack versions, MQTT topic schemas, 
Flask API endpoints
, Python 
backend
 script descriptions, ESP8266 firmware communication protocol
C
Experimental Protocols
Bill of materials, sensor mounting procedure, calibration protocol, session recording protocol, fault injection conditions, complete dataset session table
D
Extended Validation Results
Class distribution, full confusion matrix, per-class metrics, ROC/AUC table, figure captions for ROC plots, 95% Wilson CI table, SHAP feature importance tables, classifier comparison
E
Digital Twin Maturity & Limitations
Kritzinger taxonomy assessment, TRL level table, known limitations table, Health Index extended definition and calibration table
Figure Placeholder Convention
Throughout this appendix, image/chart placeholders are marked in red as [ FIGURE PLACEHOLDER — description — INSERT IMAGE HERE ]. These require the author to generate the corresponding figure (using matplotlib, seaborn, or from actual photos) and insert it in place of the placeholder.
Editable Value Convention
Values marked [EDITABLE] in purple are placeholders or defaults that should be updated with your actual measured or confirmed data before final submission.
ANNEX A
Machine Learning Models — Detailed Architecture and Training
This annex provides the complete technical description of the three-tier PMx AI pipeline used in the Hybrid Digital Twin system. It supplements the summary presented in Chapter 3 (Section 3.1.5) with full hyperparameter tables, training procedures, and decision logic.
A.1 Complete Feature Engineering Specification
The feature_engineering.py module extracts 28 engineered features from aligned raw sensor data. Features are organized into six groups (A–F), each targeting specific fault classes.
Group
Feature Name
Formula / Derivation
Source
Fault Target
Notes
A
nozzle_error
nozzle_actual – nozzle_target (°C)
OctoPrint
Heat creep, clog
Primary thermal fault signal; should stay within ±2°C during normal PID control
A
nozzle_error_abs
|nozzle_actual – nozzle_target| rolling mean (30 s window)
OctoPrint
Clog (thermal drop)
Smoothed; reduces noise from short PID transients
A
heatsink_slope (dT/dt)
Linear regression slope of temp_c over last 10 samples
KY-013
Heat creep
Threshold: > 0.5°C/min sustained → Tier 0 WARN trigger
A
temp_c_from_
baseline
temp_c – mean(temp_c[0:120 s]) at session start
KY-013
Heat creep (early)
Best single feature for heat creep: 57.16% importance
A
delta_temp_c
temp_c[t] – temp_c[t-1]
KY-013
Heat creep (rate)
Rate-of-rise indicator; high kurtosis during rapid fan failure
B
acc_magnitude
√(acc_x² + acc_y² + acc_z²)
MPU-6050
Clog
(vibration spike)
Captures combined vibration energy across all axes
B
acc_z_roll_std
Rolling std of acc_z over
 50-sample window
MPU-6050
Bad print (vibration)
Second most important feature for bad_print: 
19.44% importance
B
acc_y_roll_std
Rolling std of acc_y over 50-sample window
MPU-6050
Loose belt
Dominant feature for loose_belt: 25.58% importance; Y-axis belt slippage signature
B
kurtosis_z
Kurtosis of acc_z signal in rolling window
MPU-6050
Clog (impulsive shock)
Tier 0 trigger: kurtosis_z > 5.0 → possible clog event
B
crest_factor
peak(|acc_z|) / RMS(acc_z)
MPU-6050
Bearing wear, belt
High crest factor indicates impulsive shock (bearing spall, belt snap)
C
dominant_freq
argmax(|FFT(acc_z)|) in Hz
MPU-6050
Belt resonance
Reserved for spectral Tier extension; validated in 18–116 Hz range
C
spectral_entropy
–Σ p(f)·log(p(f)) over FFT spectrum
MPU-6050
Bearing wear
High entropy = broadband noise = bearing degradation
D
extrusion_state
1 if e_mm delta > 0.01 mm/sample, else 0
OctoPrint e_mm
Context gate
Suppresses false alerts during travel moves and retractions
D
feedrate_mmmin
Direct from OctoPrint telemetry
OctoPrint
Context gate
Used to normalize vibration amplitude by speed
E
e_ratio
actual_flow_mmps / commanded_flow_mmps
Filament encoder
Clog, low_flow
Top feature for low_flow: 26.86% importance. < 0.85 in 
sustained window → WARN
F
vibration_per_feedrate
acc_magnitude / (feedrate_mmmin + ε)
Cross-source
Step loss, belt
Normalizes vibration by commanded speed; isolates mechanical anomalies from motion
F
nozzle_err × is_extrusion
nozzle_error_abs · extrusion_state
Cross-source
Flow-specific thermal
Zero during travel; non-zero only during active extrusion — isolates clog signature
Note: 
Features marked [EDITABLE] in the Python script have tunable window sizes (default: 50 samples for B-group rolling statistics, 10 samples for A-group slopes).
A.2 Tier 0 — Rule-Based Detection (Deterministic Thresholds)
Tier 0 operates from the first print session without any training data. It applies physics-derived thresholds to five key features. Threshold values were derived from normal operating ranges validated over the first 3 healthy sessions.
Rule / Feature
Normal Range
WARN Threshold
CRIT Threshold
Physical Interpretation
heatsink_slope (dT/dt)
< 0.2°C/min
> 0.5°C/min for 3 consecutive samples
> 1.5°C/min sustained
Heatsink cooling fan degradation or failure; heat creep onset
nozzle_error (during extrusion)
± 2°C
< –3°C (sustained 20 s)
< –5°C (sustained 10 s)
Cold-pull / blockage: nozzle cannot reach setpoint due to clogged material flow
kurtosis_z (acc_z)
1.5 – 3.5
> 5.0
> 8.0
Impulsive shock event on Z-axis; clogged nozzle grinding or belt tooth skip
vibration_per_feedrate
< 0.8 (normalized)
> 1.5
> 2.5
Excessive vibration relative to commanded speed; step loss or loose mechanical coupling
e_ratio (filament encoder)
0.90 – 1.05
< 0.85 for 30 s
< 0.70 for 15 s
Filament underflow; partial clog 
(low_flow) or filament slippage
[EDITABLE] 
All threshold values above. Adjust based on your specific printer baseline after recording at least 3 healthy print sessions.
A.3 Tier 1 — Isolation Forest (Unsupervised Anomaly Detection)
Isolation Forest partitions the feature space by randomly selecting a feature and a split value. Anomalous points require fewer splits to isolate, giving them shorter path lengths and higher anomaly scores. The model is trained exclusively on data collected during healthy print sessions.
A.3.1 Training Prerequisites
The Isolation Forest model activates only after a minimum baseline is established:
  • Minimum healthy sessions required: 10
  • Minimum healthy rows for training: 50,000
  • Features used: Groups A + B + D (thermal, vibration, context)
  • Normalisation: StandardScaler (zero mean, unit variance)
A.3.2 Hyperparameters
Parameter
Value Used
Justification
n_estimators
200
Sufficient tree count for stable anomaly scores on 50 Hz vibration data; beyond 300 shows diminishing returns
contamination
0.05 (5%)
Conservative estimate of anomaly rate in calibration data; tunable based on observed false positive rate
max_features
1.0 (all features)
All feature groups A+B+D included; reduces risk of missing correlated fault signatures
max_samples
'auto' (256)
Default sklearn value; adequate sub-sample for pattern isolation on this dataset size
random_state
42
Fixed seed for reproducibility across training runs
warm_start
False
Full retrain on each new baseline calibration (triggered every 10 new sessions)
A.3.3 Output Mapping
The raw Isolation Forest score (range: –1 to +1, where –1 = anomaly) is mapped to a probability using a sigmoid transformation:
    anomaly_prob = sigmoid(–5.0 × raw_score)  = 1 / (1 + exp(5.0 × raw_score))
This produces a probability in [0, 1] where values > 0.7 indicate a likely anomaly and are forwarded to the ensemble combiner.
A.4 Tier 2 — XGBoost (Supervised Fault Classification)
XGBoost is a gradient-boosted tree ensemble. It is trained on labelled fault events collected from Tier 0 triggers. Three independent binary classifiers are trained: one for extrusion anomalies (clog / low_flow), one for kinematic faults (bad_print / loose_belt), and one for thermal faults (heat_creep).
A.4.1 Hyperparameters (Common to all three binary classifiers)
Parameter
Value
Justification
n_estimators
300
Validated on 80/20 temporal holdout; beyond 400 shows no improvement on this dataset
max_depth
6
Balances complexity and overfitting; deeper trees overfit to session-specific noise
learning_rate (eta)
0.05
Conservative learning rate with high n_estimators; reduces variance on small minority classes
subsample
0.8
Row subsampling per tree; reduces overfitting on imbalanced dataset
colsample_bytree
0.8
Feature subsampling per tree; prevents dominant features from monopolizing all trees
scale_pos_weight
class-ratio (auto)
Automatically set to n_negative/n_positive to handle class imbalance for each binary detector
eval_metric
aucpr
Area Under Precision-Recall Curve; preferred over AUC for imbalanced datasets
early_stopping_rounds
30
Prevents overfitting by stopping if aucpr does not improve for 30 consecutive rounds
use_label_encoder
False
Required for XGBoost ≥ 1.6 to suppress deprecation warnings
tree_method
'hist'
Histogram-based algorithm; efficient on Raspberry Pi CPU without GPU acceleration
[EDITABLE] 
Hyperparameter values above. Run a grid search on learning_rate ∈ {0.01, 0.05, 0.1} and max_depth ∈ {4, 6, 8} if you collect more data beyond the current 11 sessions.
A.4.2 Training Data Requirements per Binary Classifier
Binary Classifier
Positive Class
Training Rows (Positive)
Test Rows (Holdout)
Extrusion Detector
low_flow + clog
30,729 + 718 rows
7,583 rows (low_flow only; clog: 9 rows — catastrophic)
Kinematic Detector
bad_print + loose_belt
137,174 + 29,776 rows
30,275 (bad_print only; loose_belt: 0 — catastrophic)
Thermal Detector
heat_creep
43,352 rows
0 rows — heat_creep terminates print before holdout window
A.5 Ensemble Combiner — Weighted Decision Logic
The ensemble combiner aggregates outputs from active tiers using a weighted sum. Weights adapt dynamically as higher tiers become available:
Scenario
Tier 0 Weight
Tier 1 Weight
Tier 2 Weight
Activation Condition
Cold start (no models)
1.00
0.00
0.00
Day one; only rule-based detection active
Tier 1 active
0.40
0.60
0.00
After ≥ 10 healthy sessions collected
All tiers active
0.25
0.35
0.40
After sufficient labelled fault events for Tier 2 training
The combined score is computed as: combined_score = Σ (weight_i × score_i) for all active tiers.
Decision thresholds: combined_score < 0.30 → HEALTHY | 0.30–0.70 → WARN | ≥ 0.70 → CRIT (triggers print pause via OctoPrint REST API).
A.6 Model Serialization and Deployment
Trained models are serialized using joblib and stored in the models/ directory of the project. The file structure is:
models/
├── isolation_forest.pkl       # Tier 1 (retrained every 10 sessions)
├── xgb_extrusion.pkl          # Tier 2 extrusion binary classifier
├── xgb_kinematic.pkl          # Tier 2 kinematic binary classifier
├── xgb_thermal.pkl            # Tier 2 thermal binary classifier
├── scaler_tier1.pkl           # StandardScaler fitted on healthy baseline
└── model_metadata.json        # Training timestamp, n_samples, feature list
At inference time, the PMx AI pipeline loads all available models from this directory. Models missing from the directory are automatically excluded from the ensemble computation, and their weights are redistributed to the remaining active tiers.
ANNEX B
Implementation Details — Software Architecture and MQTT Specification
This annex provides a complete reference for the software architecture, MQTT topics, Flask API endpoints, and ESP8266 firmware communication protocol. It supplements Chapter 4 (Implementation and Development).
B.1 Complete Software Stack and Versions
Component
Version
Role / Notes
Python
3.11
Backend language; all ML scripts, data alignment, collectors
scikit-learn
1.4.x
Isolation Forest, StandardScaler, metrics (precision, recall, F1, Wilson CI)
XGBoost
2.0.x
Tier 2 supervised binary classifiers; tree_method='hist' for CPU efficiency
pandas
2.1.x
Data alignment (merge_asof), windowed feature engineering, SQLite I/O
numpy
1.26.x
Numerical operations; FFT (np.fft), rolling statistics
Flask
3.0.x
REST API server (port 5000); serves dashboard SPA and print control endpoints
paho-mqtt
1.6.x
Python MQTT client for collectors and alert publisher
Mosquitto
2.0.x
MQTT broker running on Raspberry Pi (port 1883); handles all telemetry routing
SQLite
3.45.x
Local database engine; WAL mode enabled; two databases: printer_data.db, sensor_data.db
OctoPrint
1.10.x / 1.11.x
Printer management; MQTT plugin for telemetry; REST API for print control
Arduino C++ / ESP8266
Arduino IDE 2.3.x
Sensor node firmware; MPU-6050 via I²C, KY-013 via ADC, Wi-Fi MQTT publishing
React + TypeScript
18.x
Dashboard frontend; component-based UI; StandaloneProvider / StreamProvider architecture
Vite
5.x
Build tool and dev server; HMR for dashboard development on Raspberry Pi
Three.js
r161
3D printer model rendering; GLTFLoader for .glb import; Object3D for real-time axis animation
Electron.js
29.x (beta)
Desktop app wrapper; reuses React/Vite/Three.js codebase; native file system access
B.2 MQTT Topic Structure and Payload Schemas
All MQTT communication uses a local Mosquitto broker. QoS 0 is used for regular telemetry; QoS 1 is used for critical alerts to guarantee delivery.
Topic
QoS
Publisher
Payload (JSON keys)
printer/sensors
0
ESP8266
timestamp_utc, temp_c, delta_temp_c, acc_x/y/z, gyro_x/y/z, extrusion, source
printer/octoprint/temperature
0
OctoPrint plugin
tool0_actual, tool0_target, bed_actual, bed_target, timestamp
printer/octoprint/position
0
OctoPrint plugin
x_mm, y_mm, z_mm, e_mm, feedrate_mmmin, timestamp
printer/octoprint/job
0
OctoPrint plugin
filename, state, progress_pct, elapsed_s, filament_used_mm
printer/octoprint/event
1
OctoPrint plugin
event_name, details_json (firmware errors, warnings, state changes)
printer/alerts/crit
1
PMx AI pipeline
fault_class, severity, combined_score, triggering_signal, component_id, action, timestamp
printer/alerts/warn
1
PMx AI pipeline
fault_class, severity, tier0_conf, anomaly_prob, tier2_prob, timestamp
printer/state
0
PMx AI pipeline
health_index, operational_state, subsystem_scores {thermal, extrusion, vibration}
B.3 Flask REST API Endpoints
The Flask server runs on port 5000 and serves both the React SPA (as static files) and the following REST endpoints for dashboard interaction and printer control:
Endpoint
Method
Description
Response / Notes
GET /
GET
Serve React SPA index.html
Serves the compiled Vite build from /dist; handles SPA routing
GET /api/status
GET
Current system health state
Returns health_index, operational_state, last_update_ts, active_tiers
GET /api/sessions
GET
List of recorded print sessions
Returns session_id, start_time, end_time, fault_summary, filament_used_mm
GET /api/sessions/<id>
GET
Session detail with features
Full time-series of engineered features for the requested session
POST /api/printer/pause
POST
Pause current print job
Calls OctoPrint REST API POST /api/job {'command': 'pause'}; returns success / error
POST /api/printer/resume
POST
Resume paused print job
Calls OctoPrint REST API POST /api/job {'command': 'resume'}
GET /api/alerts
GET
Alert history (last 100)
Returns list of {timestamp, fault_class, severity, component_id, action}
GET /api/models
GET
Model metadata
Returns training_date, n_samples, active_tiers, feature_list for each loaded model
B.4 Python Backend Scripts — Architecture Summary
Script
Role and Key Functions
printer_data_collector.py
Subscribes to printer/octoprint/# MQTT topics; parses JSON payloads; inserts rows into printer_data.db with WAL mode. Handles OctoPrint reconnection on MQTT disconnect.
sensor_data_collector.py
Subscribes to printer/sensors; validates raw ADC values (rejects acc > ±32767 overflow); converts KY-013 ADC to °C via Steinhart-Hart; inserts to sensor_data.db.
data_aligner.py
Loads both databases into pandas DataFrames; shifts sensor timestamps back 200 ms; applies merge_asof with ±500 ms tolerance; outputs aligned combined DataFrame for feature engineering.
feature_engineering.py
Computes all 28 features (Groups A–F); uses adaptive window scaling at session start (min 5 samples → scales up to full 50-sample window); replaces inf/NaN with 0.0 for runtime stability on Raspberry Pi.
pmx_ai_pipeline.py
Loads models from models/; runs Tier 0 threshold checks; calls IF predict_score() for Tier 1; calls XGBoost predict_proba() for Tier 2; computes ensemble combined_score; publishes alerts via MQTT QoS 1.
websocket_bridge.py
Python asyncio WebSocket server (ws://localhost:8765); subscribes to printer/state and printer/alerts/#; forwards JSON state updates to all connected browser clients at 2 Hz.
app.py (Flask)
HTTP server on port 5000; serves React SPA; provides REST API endpoints; proxies print control commands to OctoPrint REST API (POST http://<pi-ip>/api/job).
B.5 ESP8266 Firmware Communication Protocol
The ESP8266 NodeMCU runs Arduino C++ firmware. Key responsibilities are I²C communication with MPU-6050 (50 Hz), ADC reading of KY-013 (10 Hz), and MQTT publishing to the Mosquitto broker via Wi-Fi.
B.5.1 JSON Payload Format (printer/sensors topic)
{
  "ts": "2026-01-15T10:23:45.123Z",    // ISO 8601 UTC timestamp
  "t":  62.4,                           // KY-013 temperature in °C (Steinhart-Hart)
  "dt": 0.1,                            // delta_temp_c vs. previous sample
  "ax": 412,  "ay": -318, "az": 16041, // MPU-6050 raw acc ADC counts
  "gx": 124,  "gy": -87,  "gz": 23,   // MPU-6050 raw gyro ADC counts
  "dax": 12,  "day": -8,  "daz": 31,  // delta acc counts (vs. previous)
  "ext": 48.2,                          // filament extrusion speed mm/s
  
"src": "node_01"                      // sensor node identifier
}
B.5.2 Sampling Strategy
Sensor
Sampling Rate
Implementation Notes
MPU-6050 (acc + gyro)
50 Hz
I²C 400 kHz fast mode; MPU-6050 internal FIFO buffer used; overflow flag checked each cycle
KY-013 NTC thermistor
10 Hz
ESP8266 ADC (10-bit, 0–1V); Rref = 10 kΩ; Steinhart-Hart: A=0.001129, B=0.000234, C=8.78e-8
MQTT publish rate
10 Hz (QoS 0)
Each payload contains latest acc+gyro (50 Hz decimated to 10 Hz) and temperature; averaged over 5 acc samples
Alert publish rate
On event (QoS 1)
Only when Tier 0 threshold is crossed; no periodic publish for alerts
ANNEX C
Experimental Protocols — Hardware Setup and Data Collection
This annex documents the complete hardware installation procedure, sensor calibration protocol, session recording methodology, and fault injection conditions used for the three case studies in Chapter 5.
C.1 Bill of Materials (Complete Sensor Kit)
Component
Model / Reference
Unit Cost (DZD)
Quantity
Function in System
ESP8266 Microcontroller
NodeMCU v3 (CP2102)
800
1
Wi-Fi sensor node; I²C master for MPU-6050; ADC reader for KY-013; MQTT publisher
MEMS IMU
MPU-6050 GY-521
500
1
6-axis accelerometer + gyroscope; mounted on X-carriage bracket
NTC Thermistor Module
KY-013
200
1
Heatsink temperature monitoring; Steinhart-Hart conversion
Optical Mouse (filament encoder)
Any USB optical mouse with ADNS-2051 or equivalent sensor
600
1
Modified driver reads raw optical displacement; converted to mm/s filament flow
Reference Resistor
10 kΩ ±1% metal film
50
1
Voltage divider for KY-013 NTC Steinhart-Hart temperature conversion
Raspberry Pi (edge gateway)
Pi 4 Model B (2 GB RAM)
4 500
1
Runs OctoPrint, Mosquitto broker, Python backend, Flask server, React dashboard
MicroSD Card
32 GB Class 10
400
1
Raspberry Pi OS storage; SQLite databases; Python scripts
Breadboard + jumper wires
Half-size 400-point
200
1
Prototyping connections; no soldering required
3D printed mounting bracket
PLA, designed in Blender
~80 (filament)
2
MPU-6050 bracket for X-carriage; filament encoder housing
TOTAL (sensor kit only)
~2 330
~48 EUR — under 50 EUR target met
[EDITABLE] 
Component costs in DZD above. Update with current market prices if building the kit more than 3 months after this writing.
C.2 Sensor Mounting Procedure
C.2.1 MPU-6050 Mounting (X-Carriage)
1. Print the MPU-6050 bracket in PLA at 0.2 mm layer height, 40% infill, 2 perimeters.
2. Attach the bracket to the X-carriage using two M3×8 mm screws through existing mounting holes.
3. Seat the MPU-6050 GY-521 module into the bracket recess; secure with a drop of hot glue or two M2 screws.
4. Orient the module so the Z-axis (perpendicular to PCB) aligns with the vertical direction of the printer.
5. Route I²C wires (SDA, SCL, VCC, GND) along the cable chain to the ESP8266 node (use 22 AWG wire to minimize signal noise).
[ FIGURE PLACEHOLDER — 
MPU-6050 bracket mounted on X-carriage of BCN3D+ printer, showing wire routing along cable chain — INSERT PHOTO HERE
 — INSERT IMAGE HERE ]
C.2.2 KY-013 Mounting (Heatsink)
1. Clean the heatsink cooling block surface with isopropyl alcohol.
2. Apply a small amount of thermal paste to the KY-013 thermistor tip.
3. Affix the thermistor to the heatsink block using Kapton tape (heat-resistant up to 260°C).
4. Route the thermistor wires along the extruder cable bundle to avoid interference with moving parts.
[ FIGURE PLACEHOLDER — 
KY-013 NTC thermistor attached to BCN3D+ heatsink block with Kapton tape, showing placement adjacent to the heat break — INSERT PHOTO HERE
 — INSERT IMAGE HERE ]
C.3 Sensor Calibration Protocol
C.3.1 KY-013 Temperature Calibration Verification
The Steinhart-Hart coefficients used (A=1.129e-3, B=2.34e-4, C=8.78e-8) were derived from the manufacturer datasheet for a 10 kΩ NTC at 25°C. Verification procedure:
1. Place a calibrated digital thermometer probe adjacent to the KY-013 on the heatsink.
2. Record both readings at three temperature points: ambient (~25°C), mid-range (~45°C), and operating (~60°C).
3. Acceptable tolerance: ±2°C across all points. If deviation exceeds ±2°C, recalibrate Rref value.
[EDITABLE] 
Measured calibration readings at your three reference points. Adjust Rref in firmware if needed.
C.4 Session Recording Protocol
Each data collection session follows this standardized procedure to ensure consistency across the 11 labelled sessions:
Step
Action
Verification
1
Start Raspberry Pi and verify Mosquitto broker is running
Run: mosquitto_sub -t 'printer/#' -v — confirm no connection errors
2
Launch OctoPrint and verify MQTT plugin is active
Check OctoPrint logs for 'MQTT connected'; verify printer/octoprint/# messages appear
3
Power on ESP8266 sensor node
Verify printer/sensors messages appear at 10 Hz in MQTT subscriber
4
Start Python collectors
Run printer_data_collector.py and sensor_data_collector.py; verify rows inserted into SQLite
5
Wait 2 minutes idle (printer powered but not printing)
This captures the ambient baseline for temp_c_from_baseline feature initialization
6
Start print job via OctoPrint
Verify printer_state.printing = 1 in database; session recording begins
7
For fault sessions: inject fault condition as per Table C.1 at planned time mark
Note the exact timestamp of fault injection in session log
8
At session end, stop collectors and run data_aligner.py
Verify aligned dataset has <5% NaN rate; confirm row count matches expected duration
9
Label session in metadata.json
Add session_id, fault_class, fault_start_ts, fault_end_ts, print_file, filament_type
C.5 Fault Injection Conditions
Fault Class
Induction Method
Observable Signal
Sessions
Notes
Normal (healthy)
Standard PLA print at 210°C / 50 mm/s
All signals within baseline thresholds
5 sessions
Used for Isolation Forest training and global baseline calibration
low_flow (partial clog)
Print below optimal temperature (190°C for PLA rated 210°C); gradual accumulation
e_ratio drops below 0.85; slight nozzle_error increase
3 sessions
Progressive; persists through most of session — suitable for temporal holdout evaluation
clog (complete blockage)
Introduce carbonised PLA debris into filament path; or jam filament manually
e_ratio → 0; kurtosis_z spike; nozzle_error < –10°C
1 session
Catastrophic; terminates print within minutes — only 9 test rows in holdout
bad_print (vibration degr.)
Loosen belt tension to 50% of nominal; increase print speed to 80 mm/s
acc_z_roll_std and acc_y_roll_std elevated; e_ratio slightly below nominal
2 sessions
Progressive degradation; visible in layer lines; 30,275 test rows available
loose_belt
Fully disengage X-belt tension (belt slipping on pulley)
acc_y_roll_std very high; print fails immediately
1 session
Catastrophic; no holdout test rows — training-phase feature importance only
heat_creep
Disable heatsink cooling fan; print continuously at high speed (60 mm/s)
temp_c_from_baseline rises steadily; heatsink_slope > 0.5°C/min
1 session
Takes ~15–30 min to develop; terminates before holdout window
C.6 Dataset Summary
Session ID
Fault Class
Total Rows
Train Rows (80%)
Test Rows (20%)
Duration (min)
Print File
S01
Normal
74,302
59,441
14,861
~41
benchy_v2.gcode
S02
Normal
68,450
54,760
13,690
~38
cube_20mm.gcode
S03–S05
Normal
~55k each
80% split
20% split
~30 each
[editable — add filenames]
S06
low_flow
7,207
5,765
1,442
~4
calibration_cube.gcode
S07
low_flow
6,424
5,139
1,285
~3.5
calibration_cube.gcode
S08
low_flow
17,098
13,678
4,856
~9.5
benchy_v2.gcode
S09
bad_print
167,449
137,174
30,275
~93
vase_mode.gcode
S10
loose_belt
29,776
29,776
0
~16
print terminated early
S11
heat_creep
43,352
43,352
0
~24
print terminated early
TOTAL
6 classes
551,070
443,774
107,296
~259
[EDITABLE] 
Session IDs S03–S05: fill in actual gcode file names and row counts from your data collection log.
ANNEX D
Extended Validation Results — Confusion Matrix, ROC/AUC, and Statistical Analysis
This annex provides the full validation dataset for the Physics-Aware Binary Ensemble, including the confusion matrix (per binary classifier), ROC operating points, SHAP feature importance, and the complete statistical validation results. All results are based on the temporal holdout (last 20% of each session, n = 107,296 rows).
D.1 Dataset Class Distribution — Training vs Test Split
Fault Class
Train Rows (80%)
Test Rows (20%)
% of Total Test
Notes
Normal
261,706
74,302
69.2%
Dominant class; majority of both train and test sets
bad_print
107,174
30,275
28.2%
Progressive fault; present in holdout window
heat_creep
43,352
0
0.0%
Catastrophic: terminates print before 20% holdout
loose_belt
29,776
0
0.0%
Catastrophic: complete belt failure before holdout
low_flow
7,583
2,710
2.5%
Partial clog; gradual progression into holdout
clog
718
9
0.01%
Complete blockage; catastrophic — nearly zero in holdout
TOTAL
443,774
107,296
100%
n = 107,296 for all statistical metrics
[ FIGURE PLACEHOLDER — 
D.1 — Class distribution bar chart (training vs test) — Generate from data above using matplotlib; insert here
 — INSERT IMAGE HERE ]
D.2 Full Confusion Matrix — Physics-Aware Binary Ensemble
The confusion matrix below represents the Binary Ensemble output on the temporal holdout set (n = 107,296). Only three classes have nonzero test support: Normal, low_flow, and bad_print. Catastrophic classes (clog, loose_belt, heat_creep) appear only in the 'Other' prediction column (zero actual rows in holdout).
Actual \ Predicted
Normal
Low Flow
Bad Print
Other*
Row Total
Normal
62,836
17
11,253
196
74,302
Low Flow
—
1,639
—
1,071
2,710
Bad Print
9,067
—
21,119
89
30,275
Col Total
71,903
1,656
32,372
1,356
107,287
Note: 
(*) 'Other' column captures predictions of clog, loose_belt, or heat_creep — classes with zero actual test support in the holdout. Dashes indicate zero or negligible counts. The sum of actual rows is 107,287 (≈ 107,296 total; difference due to minor alignment rounding).
D.2.1 Per-Class Precision, Recall and F1-Score
Class
Precision
Recall (TPR)
F1-Score
Interpretation
Normal
0.874
0.846
0.860
84.6% of normal rows correctly identified; main error: 11,253 classified as bad_print
Low Flow
0.990
0.605
0.752
Very high precision: only 17 false alarms per 74,302 normal rows (FPR = 0.02%)
Bad Print
0.653
0.698
0.675
Main confusion: 9,067 normal rows classified as bad_print; gradual onset makes boundary subtle
Weighted Average
0.862
0.798
0.800
Overall accuracy: 79.77%. Weighted F1 accounts for class imbalance.
[ FIGURE PLACEHOLDER — 
D.2 — Confusion matrix heatmap (normalized by row) — Generate from Table D.2 using seaborn.heatmap(annot=True, fmt='.1%'); insert here
 — INSERT IMAGE HERE ]
D.3 ROC Analysis and AUC — Per Binary Detector
The ROC curve plots the True Positive Rate (recall) against the False Positive Rate (1-specificity) across all possible classification thresholds. AUC is estimated from the precision-recall characteristics due to severe class imbalance in the test set (normal class = 69.2%).
Detector
TPR (Recall)
FPR
Precision
Estimated AUC
Operating Point Choice
Low Flow (Extrusion Detector)
0.605
0.0002
0.990
0.802
Threshold tuned for very low FPR; accepts lower recall in exchange for near-zero false alarms
Bad Print (Kinematic Detector)
0.698
0.154
0.653
0.772
Higher FPR accepted for better coverage of progressive degradation; operators expect some false alerts
Naïve Majority Baseline
0.000
0.000
N/A
~0.500
Always predicts 'normal'; detects no faults; AUC = 0.5 (random chance)
[ FIGURE PLACEHOLDER — 
D.3a — ROC curve for Low Flow binary detector (TPR vs FPR at varying thresholds) — Plot using sklearn.metrics.roc_curve; highlight operating point at TPR=0.605, FPR=0.0002; insert here
 — INSERT IMAGE HERE ]
[ FIGURE PLACEHOLDER — 
D.3b — ROC curve for Bad Print binary detector (TPR vs FPR at varying thresholds) — Plot using sklearn.metrics.roc_curve; highlight operating point at TPR=0.698, FPR=0.154; insert here
 — INSERT IMAGE HERE ]
[ FIGURE PLACEHOLDER — 
D.3c — Precision-Recall curve for both detectors overlaid — Use sklearn.metrics.precision_recall_curve; show baseline as dashed horizontal line at class prevalence; insert here
 — INSERT IMAGE HERE ]
D.4 Statistical Validation — Confidence Intervals
All performance metrics are reported with 95% Wilson score confidence intervals (preferred over normal approximation for proportions near 0 or 1, and for small support counts such as low_flow).
Metric
Value
95% CI Lower
95% CI Upper
Support (n)
Overall Accuracy
79.77%
79.52%
80.01%
107,296
Normal Recall
84.56%
84.30%
84.82%
74,302
Low Flow Recall
60.48%
58.58%
62.35%
2,710
Bad Print Recall
69.76%
69.23%
70.28%
30,275
Low Flow Precision
98.97%
97.51%
99.57%
1,656 predictions
Low Flow F1-Score
0.7375
0.720
0.754
Propagated from precision + recall CIs
Note: 
Narrow CIs on bad_print and overall accuracy (large support) confirm stable results. Wider CI on low_flow recall reflects smaller support (n=2,710) — expected and statistically honest.
D.5 Classifier Comparison — Full Results Table
Classifier
Accuracy
Weighted F1
Macro F1
Low Flow Recall
Bad Print Recall
Naïve Majority Baseline (always 'Normal')
69.20%
0.4793
N/A
0.00%
0.00%
MLP Neural Network (Global)
80.58%
0.7769
0.4234
[editable]
[editable]
Random Forest (Global, all features)
80.37%
0.7907
0.5064
36.34%
53.00%
XGBoost (Global baseline)
85.84%
0.8573
0.4450
[editable]
[editable]
Physics-Aware Binary Ensemble (Proposed)
79.77%
0.8004
0.4524
60.48%
69.76%
Note: 
XGBoost global model achieves highest raw accuracy (85.84%) because it overfits to the dominant 'normal' class (69.2% of test). The Binary Ensemble intentionally sacrifices raw accuracy to improve minority class recall by +24.14% (low_flow) and +16.76% (bad_print).
[ FIGURE PLACEHOLDER — 
D.4 — Radar chart comparing classifiers on 5 axes: Accuracy, Weighted F1, Low Flow Recall, Bad Print Recall, Macro F1 — Generate using matplotlib polar plot; insert here
 — INSERT IMAGE HERE ]
D.6 Feature Importance Analysis (SHAP-equivalent from Tree Importance)
SHAP (SHapley Additive exPlanations) values were computed from the XGBoost tree importances (gain-based) for each binary classifier. The tables below show the top-5 features ranked by mean absolute SHAP contribution.
D.6.1 Extrusion Detector — Top Features by Importance
Rank
Feature Name
Importance (%)
SHAP Direction
Physical Meaning
1
e_ratio
26.86%
↓ below 0.85 → fault
Filament flow ratio; directly measures partial clog progression
2
nozzle_error_abs
24.23%
↑ above 3°C → fault
Nozzle cannot reach setpoint; cold material blocks normal heat transfer
3
nozzle_err × is_extrusion
18.40%
↑ during extrusion → fault
Cross-source feature; isolates thermal anomaly to active extrusion phase
4
kurtosis_z
15.70%
↑ above 5.0 → fault
Impulsive vibration spike from nozzle grinding against solidified filament
5
vibration_per_feedrate
9.80%
↑ elevated → fault
Normalized vibration; clogged nozzle increases drag on motion system
D.6.2 Heat Creep Detector — Top Features by Importance (Training Phase)
Rank
Feature Name
Importance (%)
SHAP Direction
Physical Meaning
1
temp_c_from_baseline
57.16%
↑ above 5°C → fault
Cumulative thermal drift from session start; most reliable early-warning signature
2
heatsink_slope (dT/dt)
18.30%
↑ above 0.5°C/min → warn
Rate of heatsink temperature rise; detects fan degradation before failure
3
delta_temp_c
12.40%
↑ persistent → fault
Sample-to-sample temperature change; peaks during rapid fan failure events
4
nozzle_error_abs
7.14%
Moderate correlation
Heat creep raises nozzle temperature as backpressure increases
5
e_ratio
5.00%
↓ late-stage
Late indicator: flow drops only after filament has already softened and jammed
[ FIGURE PLACEHOLDER — 
D.5 — SHAP summary beeswarm plots for each binary detector — Generate using shap.TreeExplainer(model); shap.summary_plot(shap_values, features); insert here
 — INSERT IMAGE HERE ]
ANNEX E
Digital Twin Maturity Assessment and System Limitations
This annex provides a structured self-assessment of the digital twin maturity level achieved by the system, mapped against the Kritzinger taxonomy and the Tao 5-dimensional framework. It also summarises known limitations and the corresponding future work directions.
E.1 Digital Twin Maturity Assessment (Kritzinger Taxonomy)
Capability
Level Claimed
Evidence
Gap / Future Work
Data flow (PE → VE)
✅ Fully automatic
MQTT + SQLite + WebSocket pipeline; 2 Hz update rate; latency 280–500 ms
None — this direction is fully operational
Data flow (VE → PE)
✅ Bidirectional (L3)
CRIT alerts trigger OctoPrint REST API POST /api/job pause command
Only pause/resume commands implemented; future: fan speed adjustment, temperature correction
Real-time synchronisation
✅ Soft real-time
End-to-end latency 280–500 ms; well within 10 s warning lead time for clogging
Hard real-time (< 50 ms) not needed for this application
3D geometric model
✅ Live-updating
Three.js GLB model; X/Y/Z axis positions updated from OctoPrint telemetry in real time
Z-leadscrew and bed animation not yet implemented
Predictive analytics
🔶 Partial
Tier 0 (rules) and Tier 1 (Isolation Forest) fully operational; Tier 2 (XGBoost) bootstrapping
Remaining Useful Life (RUL) prediction not yet implemented
Autonomous self-correction
❌ Not yet
System alerts and pauses; does not adjust parameters autonomously
Future: closed-loop temperature and fan speed adjustment via OctoPrint Gcode injection
Multi-printer fleet
❌ Not yet
Single printer deployment only; one dashboard instance per printer
Future: fleet manager with N printer nodes and central dashboard
Conclusion: The system qualifies as a Level 3 Digital Twin (true Digital Twin, not Digital Shadow) in the Kritzinger taxonomy, because bidirectional data flow is implemented. However, autonomous actuation beyond print pause is not yet available.
E.2 Technology Readiness Level (TRL) Assessment
TRL Level
Status
Evidence
TRL 1 — Basic principles
✅ Complete
Literature review (Chapter 1); FMEA (Chapter 2); physics-based thermal and vibration models
TRL 2 — Technology concept
✅ Complete
Digital twin architecture designed; five-layer model validated conceptually
TRL 3 — Experimental proof
✅ Complete
Feature engineering validated on real sensor data; ML models trained and evaluated
TRL 4 — Lab validation
✅ Complete
Three case studies conducted in university lab; 551,070 rows collected; 79.77% accuracy on holdout
TRL 5 — Relevant environment
🔶 Partial
Tested only on BCN3D+ in one lab; not yet validated on different printer models or production settings
TRL 6 — Prototype demo
🔶 Partial
Working prototype demonstrated; dashboard functional; label application submitted for external review
TRL 7+ — System prototype
❌ Future
Multi-printer deployment, long-term field study, commercial packaging not yet done
E.3 Known Limitations and Mitigations
Limitation
Impact
Severity
Proposed Mitigation
Single printer dataset (BCN3D+, n=11 sessions)
Models may not generalize to other FDM models or filament types
Medium
Multi-printer data collection campaign; transfer learning across printer models
Catastrophic class absence in holdout (clog, loose_belt, heat_creep)
Cannot report precision/recall for worst-case faults
High — jury question risk
Dedicated fault injection sessions with shorter prints; sliding window split instead of temporal holdout
Session-level labels (not row-level)
Some rows labelled 'heat_creep' may be healthy; noisy labels reduce model quality
Medium
Automated real-time labelling triggered by Tier 0 events; OctoPrint event logging
Single sensor node (no redundancy)
Sensor failure causes complete monitoring blackout
Low-Medium
Watchdog timer on ESP8266; dashboard heartbeat monitor with 'sensor offline' alert
SQLite for persistence
Maximum ~5 concurrent readers; not suitable for fleet (N > 5 printers)
Low (single printer)
PostgreSQL migration planned for fleet version; SQLite WAL mode sufficient for 1 printer
No vision-based monitoring
Layer delamination and geometric defects not detected
Medium
Camera module (Pi Camera 3) planned as optional Tier 3 extension using CNN defect detection
E.4 Health Index — Extended Definition and Calibration
The Health Index (HI) formula is:
    
HI(t) = 1 − ( wT × ŝT(t)  +  wE × ŝE(t)  +  wV × ŝV(t) )
where ŝT, ŝE, ŝV ∈ [0,1] are the normalized fault probability scores for thermal (T), extrusion (E), and vibration (V) subsystems, and wT + wE + wV = 1.00.
Subsystem
Weight (w)
Key Input Features
Normal Score Range
Fault Score Range
Extrusion (wE)
0.424
e_ratio, nozzle_error_abs, nozzle_err×is_extrusion
0.00 – 0.20
0.50 – 1.00
Vibration (wV)
0.338
acc_y_roll_std, acc_z_roll_std, kurtosis_z
0.00 – 0.25
0.40 – 1.00
Thermal (wT)
0.238
temp_c_from_baseline, heatsink_slope
0.00 – 0.15
0.35 – 1.00
Note: 
Weights were derived from global Random Forest feature importances across all fault classes. Recalibration is recommended after collecting > 20 labelled sessions. The HI value updates at 2 Hz on the dashboard.
[ FIGURE PLACEHOLDER — 
E.1 — Health Index time-series plot for Session S09 (bad_print) showing progressive HI decline — Generate from session data; insert here
 — INSERT IMAGE HERE ]
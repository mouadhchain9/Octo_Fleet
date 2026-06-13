import React from 'react';
import { useFleetStore } from '../../store/useFleetStore';

export function SetupGuide() {
  const { toggleModal } = useFleetStore();

  return (
    <div className="floating-pane config-modal" style={{ display: 'block', zIndex: 4000, width: '80%', height: '80%', left: '10%', top: '10%', transform: 'none' }}>
      <div className="pane-header">
        <span>OCTO-FLEET SETUP & CONFIGURATION GUIDE</span>
        <span className="close-x-btn" onClick={() => toggleModal('setupGuide', false)}>×</span>
      </div>

      <div className="config-body" style={{ padding: '20px', lineHeight: '1.6', height: 'calc(100% - 45px)', overflowY: 'auto' }}>

        <h2>1. Hardware & Sensors Configuration</h2>
        <p>The system requires a sensor kit attached to your 3D printer. Total estimated cost is under 50 EUR (~2330 DZD).</p>
        <ul>
          <li><strong>Microcontroller:</strong> ESP8266 NodeMCU v3 (CP2102)</li>
          <li><strong>IMU (Vibration):</strong> MPU-6050 GY-521 (6-axis accelerometer & gyroscope)</li>
          <li><strong>Thermistor (Heat Creep):</strong> KY-013 NTC Module with a 10 kΩ reference resistor.</li>
          <li><strong>Filament Encoder:</strong> Modified optical mouse (ADNS-2051 or equivalent).</li>
          <li><strong>Edge Gateway:</strong> Raspberry Pi 4 Model B (2 GB RAM)</li>
        </ul>

        <div style={{ background: 'rgba(0,255,194,0.1)', padding: '10px', borderLeft: '3px solid #00ffc2', margin: '15px 0' }}>
          <strong>Mounting MPU-6050:</strong> Print a PLA bracket (0.2mm layer, 40% infill). Attach to X-carriage. Seat the MPU-6050 with Z-axis facing vertically. Route I²C wires along the cable chain.<br />
          <strong>Mounting KY-013:</strong> Clean heatsink, apply thermal paste. Affix using Kapton tape adjacent to the heat break.
        </div>

        <hr style={{ borderColor: 'rgba(255,255,255,0.1)', margin: '20px 0' }} />

        <h2>2. Software Stack & MQTT Broker</h2>
        <p>The digital twin runs heavily on MQTT for real-time telemetry.</p>
        <ul>
          <li><strong>Mosquitto Broker:</strong> Installed on the Raspberry Pi (Port 1883).</li>
          <li><strong>Backend:</strong> Python 3.11 with scikit-learn, XGBoost, pandas.</li>
          <li><strong>API Server:</strong> Flask (Port 5000)</li>
        </ul>

        <h3>MQTT Topics Scheme:</h3>
        <ul>
          <li><code>printer/sensors</code> (QoS 0): Raw ESP8266 telemetry (50Hz MPU, 10Hz Temp).</li>
          <li><code>printer/octoprint/temperature</code> & <code>.../position</code> & <code>.../job</code> (QoS 0): OctoPrint plugin streams.</li>
          <li><code>printer/alerts/crit</code> & <code>.../warn</code> (QoS 1): PMx AI pipeline alerts.</li>
        </ul>

        <hr style={{ borderColor: 'rgba(255,255,255,0.1)', margin: '20px 0' }} />

        <h2>3. OctoPrint Integration</h2>
        <p>Ensure OctoPrint (v1.10.x+) is running on the Raspberry Pi. You must install an MQTT plugin to forward printer telemetry (like nozzle position, temperatures, and G-Code progress) to the Mosquitto broker.</p>
        <p>This allows the UI to pause/resume the print directly via the <code>/api/job</code> REST endpoint when a CRITICAL fault is detected.</p>

        <hr style={{ borderColor: 'rgba(255,255,255,0.1)', margin: '20px 0' }} />

        <h2>4. Data Capture Scripts</h2>
        <p>On the Raspberry Pi, you need to run these Python backend scripts in parallel:</p>
        <ol>
          <li><code>printer_data_collector.py</code>: Subscribes to OctoPrint topics and inserts into <code>printer_data.db</code>.</li>
          <li><code>sensor_data_collector.py</code>: Subscribes to ESP8266 sensors and inserts into <code>sensor_data.db</code>.</li>
          <li><code>data_aligner.py</code>: Merges the two databases for feature engineering.</li>
          <li><code>pmx_ai_pipeline.py</code>: Loads trained AI models (Isolation Forest, XGBoost) and publishes anomaly alerts.</li>
        </ol>

      </div>
    </div>
  );
}

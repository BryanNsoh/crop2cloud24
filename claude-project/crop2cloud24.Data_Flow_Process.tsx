import React from 'react';
import { ArrowRight, Database, FileText, Code, Cloud, BarChart, Droplet } from 'lucide-react';

const Box = ({ title, children, color, width = 'w-64', height = 'h-auto' }) => (
  <div className={`p-2 rounded-lg border-2 ${color} ${width} ${height} flex flex-col items-center justify-center text-center text-sm`}>
    <h3 className="font-bold mb-1">{title}</h3>
    {children}
  </div>
);

const Arrow = ({ label, direction = 'right' }) => (
  <div className={`flex flex-col items-center mx-2 ${direction === 'down' ? 'transform rotate-90' : ''}`}>
    <ArrowRight className="my-1" />
    <span className="text-xs">{label}</span>
  </div>
);

const DataFlow = () => (
  <div className="flex flex-col items-center space-y-6 p-4">
    <div className="flex items-start">
      <Box title="Initial Setup Script" color="border-green-500">
        <Code size={24} />
        <span>Creates BigQuery table structure</span>
      </Box>
      <Arrow label="Setup" direction="down" />
    </div>

    <div className="flex items-start">
      <Box title="Local Files" color="border-blue-500">
        <FileText size={24} />
        <span>Raw sensor data (.dat files)</span>
      </Box>
      <Arrow label="Ingest" />
      <Box title="Data Ingestion" color="border-green-500">
        <Code size={24} />
        <span>Process raw data</span>
        <span>Upload to BigQuery</span>
      </Box>
      <Arrow label="Upload" />
      <Box title="BigQuery: Raw Data" color="border-purple-500">
        <Database size={24} />
        <div>
          <div>sensor_readings</div>
          <div>weather_data</div>
        </div>
      </Box>
    </div>
    
    <div className="flex items-start">
      <Box title="BigQuery: Raw Data" color="border-purple-500">
        <Database size={24} />
        <div>
          <div>sensor_readings</div>
          <div>weather_data</div>
        </div>
      </Box>
      <Arrow label="Fetch" />
      <Box title="Log Llama Model" color="border-yellow-500">
        <Cloud size={24} />
        <span>Predict values for next 4 days</span>
        <span>Using sensor data and weather data</span>
      </Box>
      <Arrow label="Store" />
      <Box title="BigQuery: Predictions" color="border-purple-500">
        <Database size={24} />
        <div>
          <div>predicted_sensor_readings</div>
          <div>predicted_weather_data</div>
        </div>
      </Box>
    </div>
    
    <div className="flex items-start">
      <Box title="BigQuery: All Data" color="border-purple-500" width="w-72">
        <Database size={24} />
        <div>
          <div>sensor_readings (current & predicted)</div>
          <div>weather_data (current & predicted)</div>
        </div>
      </Box>
      <Arrow label="Fetch" />
      <Box title="Stress Indices Computation" color="border-red-500" width="w-72">
        <BarChart size={24} />
        <span>Compute for current and predicted data:</span>
        <div>CWSI, SWSI, ET, MDS</div>
      </Box>
      <Arrow label="Store" />
      <Box title="BigQuery: Treatment Tables" color="border-purple-500" width="w-72">
        <Database size={24} />
        <div>
          <div>treatment_X_stress_indices</div>
          <div>(current and predicted values)</div>
        </div>
      </Box>
    </div>
    
    <div className="flex items-start">
      <Box title="BigQuery: All Data" color="border-purple-500" width="w-80">
        <Database size={24} />
        <div>
          <div>sensor_readings (current & predicted)</div>
          <div>weather_data (current & predicted)</div>
          <div>treatment_X_stress_indices (current & predicted)</div>
        </div>
      </Box>
      <Arrow label="Fetch" />
      <Box title="Predictive Control Model" color="border-indigo-500" width="w-80">
        <Droplet size={24} />
        <span>Determine optimal irrigation amount</span>
        <span>to keep stress indices below threshold</span>
        <span>over the next 4 days</span>
      </Box>
      <Arrow label="Store" />
      <Box title="BigQuery: Irrigation Recommendation" color="border-purple-500">
        <Database size={24} />
        <div>
          <div>irrigation_recommendations</div>
          <div>(per treatment)</div>
        </div>
      </Box>
    </div>
    
    <div className="mt-4 text-sm">
      <div><strong>Note:</strong> All steps are orchestrated by run_pipeline.py</div>
      <div>Treatment-specific tables are created for each unique treatment</div>
    </div>
  </div>
);

export default DataFlow;
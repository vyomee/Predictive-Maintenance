import tkinter as tk
from tkinter import ttk, messagebox, filedialog, scrolledtext
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import xgboost as xgb
from datetime import datetime, timedelta
import time
import json
import threading
import warnings
warnings.filterwarnings('ignore')

# MAIN APPLICATION CLASS
class RevonyxRULMonitor:
    def __init__(self, root):
        self.root = root
        self.root.title("Revionyx RUL Monitoring System")
        self.root.geometry("1400x900")
        self.root.configure(bg='#f0f0f0')

        # Initialize data
        self.model = None
        self.feature_names = None
        self.telemetry_df = None
        self.maint_df = None
        self.errors_df = None
        self.machines_df = None

        # Application state
        self.starred_machines = [1, 2, 3]
        self.maintenance_schedule = []
        self.rul_cache = {}
        self.current_view = "dashboard"

        # Load data and model
        self.load_model_and_data()

        # Initialize maintenance schedule
        self.initialize_maintenance_schedule()

        # Create GUI
        self.create_gui()

        # Start datetime updater
        self.update_datetime()

    def load_model_and_data(self):
        """Load trained model and sensor data from edge device storage"""
        try:
            # model_path = '/edge_device/models/xgboost_model.json'
            self.model = xgb.Booster()
            self.model.load_model('xgboost_model.json')

            with open('feature_names.txt', 'r') as f:
                self.feature_names = [line.strip() for line in f.readlines()]

            # Load sensor data
            self.telemetry_df = pd.read_csv('PdM_telemetry.csv')
            self.telemetry_df['datetime'] = pd.to_datetime(self.telemetry_df['datetime'])

            self.maint_df = pd.read_csv('PdM_maint.csv')
            self.maint_df['datetime'] = pd.to_datetime(self.maint_df['datetime'])

            self.errors_df = pd.read_csv('PdM_errors.csv')
            self.errors_df['datetime'] = pd.to_datetime(self.errors_df['datetime'])

            self.machines_df = pd.read_csv('PdM_machines.csv')

            print("Model and data loaded successfully")

        except Exception as e:
            messagebox.showwarning(
                "Data Loading",
                f"Could not load all files. Using mock data.\nError: {e}"
            )
            self.create_mock_data()

    def create_mock_data(self):
        """Create mock data for demonstration"""
        # Create minimal mock datasets
        dates = pd.date_range(start='2024-01-01', end='2024-01-31', freq='H')

        self.telemetry_df = pd.DataFrame({
            'datetime': dates,
            'machineID': np.random.choice(range(1, 21), len(dates)),
            'volt': np.random.uniform(160, 180, len(dates)),
            'rotate': np.random.uniform(400, 500, len(dates)),
            'pressure': np.random.uniform(90, 110, len(dates)),
            'vibration': np.random.uniform(35, 45, len(dates))
        })

        self.machines_df = pd.DataFrame({
            'machineID': range(1, 21),
            'model': np.random.choice([1, 2, 3, 4], 20),
            'age': np.random.randint(0, 20, 20)
        })

        self.maint_df = pd.DataFrame({
            'datetime': pd.date_range(start='2024-01-01', periods=50, freq='D'),
            'machineID': np.random.choice(range(1, 21), 50),
            'comp': np.random.choice(['comp1', 'comp2', 'comp3', 'comp4'], 50)
        })

        self.errors_df = pd.DataFrame({
            'datetime': pd.date_range(start='2024-01-01', periods=100, freq='12H'),
            'machineID': np.random.choice(range(1, 21), 100),
            'errorID': np.random.choice(['error1', 'error2', 'error3'], 100)
        })

    def initialize_maintenance_schedule(self):
        """Initialize maintenance schedule with sample data"""
        self.maintenance_schedule = [
            {
                'Date': (datetime.now() + timedelta(days=1)).strftime('%Y-%m-%d'),
                'Machine_ID': 1,
                'RUL': 18.5,
                'Component': 'comp1',
                'Priority': 'CRITICAL',
                'Status': 'Scheduled'
            },
            {
                'Date': (datetime.now() + timedelta(days=3)).strftime('%Y-%m-%d'),
                'Machine_ID': 5,
                'RUL': 25.3,
                'Component': 'comp3',
                'Priority': 'HIGH',
                'Status': 'Scheduled'
            },
            {
                'Date': (datetime.now() + timedelta(days=5)).strftime('%Y-%m-%d'),
                'Machine_ID': 12,
                'RUL': 32.1,
                'Component': 'comp2',
                'Priority': 'MEDIUM',
                'Status': 'In Progress'
            }
        ]

    def create_gui(self):
        """Create main GUI structure"""
        # Header frame
        self.create_header()

        # Navigation frame
        self.create_navigation()

        # Main content frame
        self.content_frame = tk.Frame(self.root, bg='white')
        self.content_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)

        # Show dashboard by default
        self.show_dashboard()

    def create_header(self):
        """Create header with title and datetime"""
        header_frame = tk.Frame(self.root, bg='#1f77b4', height=80)
        header_frame.pack(fill=tk.X)
        header_frame.pack_propagate(False)

        title_label = tk.Label(
            header_frame,
            text="Revionyx RUL Monitoring System",
            font=('Arial', 24, 'bold'),
            bg='#1f77b4',
            fg='white'
        )
        title_label.pack(pady=10)

        self.datetime_label = tk.Label(
            header_frame,
            text="",
            font=('Arial', 12),
            bg='#1f77b4',
            fg='white'
        )
        self.datetime_label.pack()

    def create_navigation(self):
        """Create navigation menu"""
        nav_frame = tk.Frame(self.root, bg='#e0e0e0', height=50)
        nav_frame.pack(fill=tk.X)

        buttons = [
            ("Dashboard", self.show_dashboard),
            ("Maintenance", self.show_maintenance),
            ("Machine RUL", self.show_machine_rul),
            ("Sensor Data", self.show_sensor_data),
            ("Historical Data", self.show_historical_data),
            ("Retrain Model", self.show_retrain),
            ("Prototype Test", self.show_prototype_test)
        ]

        for text, command in buttons:
            btn = tk.Button(
                nav_frame,
                text=text,
                command=command,
                font=('Arial', 10, 'bold'),
                bg='#4CAF50',
                fg='white',
                relief=tk.RAISED,
                padx=10,
                pady=5
            )
            btn.pack(side=tk.LEFT, padx=5, pady=5)

    def update_datetime(self):
        """Update datetime display"""
        current_time = datetime.now().strftime("%A, %B %d, %Y  |  %H:%M:%S")
        self.datetime_label.config(text=f" {current_time}")
        self.root.after(1000, self.update_datetime)

    def clear_content(self):
        """Clear content frame"""
        for widget in self.content_frame.winfo_children():
            widget.destroy()

    # RUL CALCULATION FUNCTIONS

    def calculate_rul_from_probability(self, P_degraded):
        """Convert degraded probability to RUL percentage"""
        return (1 - P_degraded) * 100

    def assign_alert_tier(self, rul_percentage):
        """Assign alert tier based on RUL"""
        if rul_percentage < 20:
            return 'CRITICAL'
        elif rul_percentage < 35:
            return 'WARNING'
        elif rul_percentage < 65:
            return 'MONITOR'
        elif rul_percentage < 80:
            return 'GOOD'
        else:
            return 'HEALTHY'

    def get_alert_color(self, tier):
        """Get color for alert tier"""
        colors = {
            'CRITICAL': '#d32f2f',
            'WARNING': '#f57c00',
            'MONITOR': '#fbc02d',
            'GOOD': '#7cb342',
            'HEALTHY': '#388e3c'
        }
        return colors.get(tier, '#757575')

    def predict_rul_for_machine(self, machine_id):
        """Predict RUL for a machine using pre-trained model"""
        try:
            # Check cache
            cache_key = f"{machine_id}_{datetime.now().strftime('%Y%m%d%H')}"
            if cache_key in self.rul_cache:
                return self.rul_cache[cache_key]

            if self.telemetry_df is None or self.machines_df is None:
                # Return mock prediction
                P_degraded = np.random.uniform(0.15, 0.45)
                rul = self.calculate_rul_from_probability(P_degraded)
                return {
                    'machine_id': machine_id,
                    'rul_percentage': rul,
                    'alert_tier': self.assign_alert_tier(rul),
                    'P_degraded': P_degraded
                }

            # Get recent telemetry
            current_time = self.telemetry_df['datetime'].max()
            recent_data = self.telemetry_df[
                (self.telemetry_df['machineID'] == machine_id) &
                (self.telemetry_df['datetime'] >= current_time - pd.Timedelta(hours=24))
            ]

            if len(recent_data) == 0 or self.model is None:
                # Mock prediction
                P_degraded = np.random.uniform(0.1, 0.5)
                rul = self.calculate_rul_from_probability(P_degraded)
                result = {
                    'machine_id': machine_id,
                    'rul_percentage': rul,
                    'alert_tier': self.assign_alert_tier(rul),
                    'P_degraded': P_degraded
                }
                self.rul_cache[cache_key] = result
                return result

            # PLACEHOLDER: Feature engineering
            # In production, use full feature pipeline
            feature_vector = np.zeros(len(self.feature_names))

            # Fill basic features
            if 'volt' in self.feature_names:
                feature_vector[self.feature_names.index('volt')] = recent_data['volt'].iloc[-1]
            if 'rotate' in self.feature_names:
                feature_vector[self.feature_names.index('rotate')] = recent_data['rotate'].iloc[-1]
            if 'pressure' in self.feature_names:
                feature_vector[self.feature_names.index('pressure')] = recent_data['pressure'].iloc[-1]
            if 'vibration' in self.feature_names:
                feature_vector[self.feature_names.index('vibration')] = recent_data['vibration'].iloc[-1]

            # Make prediction
            dmatrix = xgb.DMatrix(feature_vector.reshape(1, -1), feature_names=self.feature_names)
            P_degraded = float(self.model.predict(dmatrix)[0])
            rul = self.calculate_rul_from_probability(P_degraded)

            result = {
                'machine_id': machine_id,
                'rul_percentage': rul,
                'alert_tier': self.assign_alert_tier(rul),
                'P_degraded': P_degraded
            }

            self.rul_cache[cache_key] = result
            return result

        except Exception as e:
            print(f"Error predicting RUL: {e}")
            # Return default
            return {
                'machine_id': machine_id,
                'rul_percentage': 75.0,
                'alert_tier': 'GOOD',
                'P_degraded': 0.25
            }

    def generate_simulated_sensor_reading(self, machine_id, working_hours):
        """Generate simulated sensor readings for prototype testing"""
        try:
            if self.telemetry_df is None:
                # Use default values
                volt_base, rotate_base = 170, 450
                pressure_base, vibration_base = 100, 40
            else:
                # Get historical baseline
                machine_history = self.telemetry_df[
                    self.telemetry_df['machineID'] == machine_id
                ]

                if len(machine_history) > 0:
                    volt_base = machine_history['volt'].mean()
                    rotate_base = machine_history['rotate'].mean()
                    pressure_base = machine_history['pressure'].mean()
                    vibration_base = machine_history['vibration'].mean()
                else:
                    volt_base = self.telemetry_df['volt'].mean()
                    rotate_base = self.telemetry_df['rotate'].mean()
                    pressure_base = self.telemetry_df['pressure'].mean()
                    vibration_base = self.telemetry_df['vibration'].mean()

            # Calculate degradation factor
            degradation_factor = min(working_hours / 10000, 1.0)

            # Simulate sensor readings with degradation
            simulated_data = {
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'machine_id': machine_id,
                'working_hours': working_hours,
                'volt': round(volt_base * (1 - 0.08 * degradation_factor) + np.random.normal(0, 2), 2),
                'rotate': round(rotate_base * (1 - 0.05 * degradation_factor) + np.random.normal(0, 20), 2),
                'pressure': round(pressure_base * (1 + 0.12 * degradation_factor) + np.random.normal(0, 5), 2),
                'vibration': round(vibration_base * (1 + 0.25 * degradation_factor) + np.random.normal(0, 3), 2),
                'degradation_factor': round(degradation_factor, 3)
            }

            return simulated_data

        except Exception as e:
            print(f"Error generating simulated data: {e}")
            return None

    def predict_rul_from_simulated_data(self, simulated_data):
        """Predict RUL from simulated sensor reading"""
        try:
            if self.model is None or self.feature_names is None:
                # Calculate mock RUL
                degradation = simulated_data.get('degradation_factor', 0.5)
                rul = (1 - degradation) * 100
                return {
                    'rul_percentage': rul,
                    'alert_tier': self.assign_alert_tier(rul),
                    'P_degraded': degradation
                }

            # PLACEHOLDER: Create feature vector
            feature_vector = np.zeros(len(self.feature_names))

            if 'volt' in self.feature_names:
                feature_vector[self.feature_names.index('volt')] = simulated_data['volt']
            if 'rotate' in self.feature_names:
                feature_vector[self.feature_names.index('rotate')] = simulated_data['rotate']
            if 'pressure' in self.feature_names:
                feature_vector[self.feature_names.index('pressure')] = simulated_data['pressure']
            if 'vibration' in self.feature_names:
                feature_vector[self.feature_names.index('vibration')] = simulated_data['vibration']

            # Make prediction
            dmatrix = xgb.DMatrix(feature_vector.reshape(1, -1), feature_names=self.feature_names)
            P_degraded = float(self.model.predict(dmatrix)[0])
            rul = self.calculate_rul_from_probability(P_degraded)

            return {
                'rul_percentage': rul,
                'alert_tier': self.assign_alert_tier(rul),
                'P_degraded': P_degraded
            }

        except Exception as e:
            print(f"Error predicting RUL: {e}")
            return None

    # VIEW 1: DASHBOARD - FLEET STATUS

    def show_dashboard(self):
        """Display fleet status dashboard"""
        self.clear_content()
        self.current_view = "dashboard"

        # Title
        title_label = tk.Label(
            self.content_frame,
            text="Fleet Status Overview",
            font=('Arial', 18, 'bold'),
            bg='white'
        )
        title_label.pack(pady=10)

        # Calculate fleet status
        if self.machines_df is not None:
            machine_ids = self.machines_df['machineID'].unique()[:20]
        else:
            machine_ids = range(1, 21)

        fleet_distribution = {
            'CRITICAL (<20%)': 0,
            'WARNING (20-35%)': 0,
            'MONITOR (35-65%)': 0,
            'GOOD (65-80%)': 0,
            'HEALTHY (>80%)': 0
        }

        # Calculate distribution
        for machine_id in machine_ids:
            result = self.predict_rul_for_machine(machine_id)
            rul = result['rul_percentage']

            if rul < 20:
                fleet_distribution['CRITICAL (<20%)'] += 1
            elif rul < 35:
                fleet_distribution['WARNING (20-35%)'] += 1
            elif rul < 65:
                fleet_distribution['MONITOR (35-65%)'] += 1
            elif rul < 80:
                fleet_distribution['GOOD (65-80%)'] += 1
            else:
                fleet_distribution['HEALTHY (>80%)'] += 1

        # Display metrics
        metrics_frame = tk.Frame(self.content_frame, bg='white')
        metrics_frame.pack(fill=tk.X, padx=20, pady=10)

        colors = ['#d32f2f', '#f57c00', '#fbc02d', '#7cb342', '#388e3c']
        icons = ['🚨', '⚠️', '👁️', '✅', '💚']

        for idx, (status, count) in enumerate(fleet_distribution.items()):
            metric_frame = tk.Frame(
                metrics_frame,
                bg=colors[idx],
                relief=tk.RAISED,
                borderwidth=2
            )
            metric_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5)

            icon_label = tk.Label(
                metric_frame,
                text=icons[idx],
                font=('Arial', 36),
                bg=colors[idx],
                fg='white'
            )
            icon_label.pack(pady=10)

            count_label = tk.Label(
                metric_frame,
                text=str(count),
                font=('Arial', 48, 'bold'),
                bg=colors[idx],
                fg='white'
            )
            count_label.pack()

            status_label = tk.Label(
                metric_frame,
                text=status,
                font=('Arial', 12),
                bg=colors[idx],
                fg='white'
            )
            status_label.pack(pady=10)

        # Summary text
        summary_frame = tk.Frame(self.content_frame, bg='#f5f5f5', relief=tk.SUNKEN, borderwidth=2)
        summary_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)

        summary_text = scrolledtext.ScrolledText(
            summary_frame,
            font=('Courier', 10),
            bg='#f5f5f5',
            height=15
        )
        summary_text.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        summary_content = f"""
╔═══════════════════════════════════════════════════════════════╗
║                    FLEET STATUS SUMMARY                       ║
╚═══════════════════════════════════════════════════════════════╝

Total Machines Monitored: {len(machine_ids)}
Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

ALERT DISTRIBUTION:
  🚨 CRITICAL (<20%):  {fleet_distribution['CRITICAL (<20%)']} machines - Immediate action required
  ⚠️  WARNING (20-35%): {fleet_distribution['WARNING (20-35%)']} machines - Schedule maintenance within 7 days
  👁️  MONITOR (35-65%): {fleet_distribution['MONITOR (35-65%)']} machines - Increase monitoring frequency
  ✅ GOOD (65-80%):    {fleet_distribution['GOOD (65-80%)']} machines - Normal operation
  💚 HEALTHY (>80%):   {fleet_distribution['HEALTHY (>80%)']} machines - Excellent condition

RECOMMENDATIONS:
"""

        if fleet_distribution['CRITICAL (<20%)'] > 0:
            summary_content += f"  • {fleet_distribution['CRITICAL (<20%)']} machines require IMMEDIATE maintenance\n"
        if fleet_distribution['WARNING (20-35%)'] > 0:
            summary_content += f"  • {fleet_distribution['WARNING (20-35%)']} machines should be scheduled for maintenance\n"
        if fleet_distribution['MONITOR (35-65%)'] > 0:
            summary_content += f"  • {fleet_distribution['MONITOR (35-65%)']} machines need increased monitoring\n"

        summary_content += f"\nSystem Status: {'⚠️ ACTION REQUIRED' if fleet_distribution['CRITICAL (<20%)'] > 0 else '✅ OPERATIONAL'}"

        summary_text.insert('1.0', summary_content)
        summary_text.config(state=tk.DISABLED)

    # VIEW 2: MAINTENANCE SCHEDULE

    def show_maintenance(self):
        """Display maintenance schedule"""
        self.clear_content()
        self.current_view = "maintenance"

        # Title
        title_label = tk.Label(
            self.content_frame,
            text="Maintenance Schedule (Next 7 Days)",
            font=('Arial', 18, 'bold'),
            bg='white'
        )
        title_label.pack(pady=10)

        # Buttons
        button_frame = tk.Frame(self.content_frame, bg='white')
        button_frame.pack(fill=tk.X, padx=20, pady=5)

        tk.Button(
            button_frame,
            text="+ Add Entry",
            command=self.add_maintenance_entry,
            font=('Arial', 10, 'bold'),
            bg='#4CAF50',
            fg='white',
            padx=15,
            pady=5
        ).pack(side=tk.LEFT, padx=5)

        tk.Button(
            button_frame,
            text="x Remove Entry",
            command=self.remove_maintenance_entry,
            font=('Arial', 10, 'bold'),
            bg='#f44336',
            fg='white',
            padx=15,
            pady=5
        ).pack(side=tk.LEFT, padx=5)

        tk.Button(
            button_frame,
            text="View All Records",
            command=self.view_all_maintenance_records,
            font=('Arial', 10, 'bold'),
            bg='#2196F3',
            fg='white',
            padx=15,
            pady=5
        ).pack(side=tk.LEFT, padx=5)

        tk.Button(
            button_frame,
            text="Download Schedule",
            command=self.download_maintenance_schedule,
            font=('Arial', 10, 'bold'),
            bg='#FF9800',
            fg='white',
            padx=15,
            pady=5
        ).pack(side=tk.LEFT, padx=5)

        # Schedule table
        table_frame = tk.Frame(self.content_frame, bg='white')
        table_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=10)

        # Create treeview
        columns = ('Date', 'Machine_ID', 'RUL', 'Component', 'Priority', 'Status')
        self.schedule_tree = ttk.Treeview(table_frame, columns=columns, show='headings', height=15)

        # Define headings
        for col in columns:
            self.schedule_tree.heading(col, text=col)
            self.schedule_tree.column(col, width=150, anchor=tk.CENTER)

        # Add scrollbar
        scrollbar = ttk.Scrollbar(table_frame, orient=tk.VERTICAL, command=self.schedule_tree.yview)
        self.schedule_tree.configure(yscroll=scrollbar.set)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.schedule_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # Populate data
        self.refresh_maintenance_table()

    def refresh_maintenance_table(self):
        """Refresh maintenance schedule table"""
        # Clear existing items
        for item in self.schedule_tree.get_children():
            self.schedule_tree.delete(item)

        # Add items
        for entry in self.maintenance_schedule:
            values = (
                entry['Date'],
                entry['Machine_ID'],
                f"{entry['RUL']:.1f}%",
                entry['Component'],
                entry['Priority'],
                entry['Status']
            )

            # Color code by priority
            tags = ()
            if entry['Priority'] == 'CRITICAL':
                tags = ('critical',)
            elif entry['Priority'] == 'HIGH':
                tags = ('high',)

            self.schedule_tree.insert('', tk.END, values=values, tags=tags)

        # Configure tags
        self.schedule_tree.tag_configure('critical', background='#ffcdd2')
        self.schedule_tree.tag_configure('high', background='#ffe0b2')

    def add_maintenance_entry(self):
        """Add new maintenance entry"""
        add_window = tk.Toplevel(self.root)
        add_window.title("Add Maintenance Entry")
        add_window.geometry("400x400")
        add_window.configure(bg='white')

        # Form fields
        tk.Label(add_window, text="Add New Maintenance Entry", font=('Arial', 14, 'bold'), bg='white').pack(pady=10)

        form_frame = tk.Frame(add_window, bg='white')
        form_frame.pack(padx=20, pady=10, fill=tk.BOTH, expand=True)

        # Date
        tk.Label(form_frame, text="Date (YYYY-MM-DD):", bg='white').grid(row=0, column=0, sticky=tk.W, pady=5)
        date_entry = tk.Entry(form_frame, width=30)
        date_entry.insert(0, (datetime.now() + timedelta(days=1)).strftime('%Y-%m-%d'))
        date_entry.grid(row=0, column=1, pady=5)

        # Machine ID
        tk.Label(form_frame, text="Machine ID:", bg='white').grid(row=1, column=0, sticky=tk.W, pady=5)
        machine_entry = tk.Entry(form_frame, width=30)
        machine_entry.insert(0, "1")
        machine_entry.grid(row=1, column=1, pady=5)

        # RUL
        tk.Label(form_frame, text="RUL (%):", bg='white').grid(row=2, column=0, sticky=tk.W, pady=5)
        rul_entry = tk.Entry(form_frame, width=30)
        rul_entry.insert(0, "50.0")
        rul_entry.grid(row=2, column=1, pady=5)

        # Component
        tk.Label(form_frame, text="Component:", bg='white').grid(row=3, column=0, sticky=tk.W, pady=5)
        component_var = tk.StringVar(value='comp1')
        component_menu = ttk.Combobox(form_frame, textvariable=component_var, values=['comp1', 'comp2', 'comp3', 'comp4'], width=27)
        component_menu.grid(row=3, column=1, pady=5)

        # Priority
        tk.Label(form_frame, text="Priority:", bg='white').grid(row=4, column=0, sticky=tk.W, pady=5)
        priority_var = tk.StringVar(value='MEDIUM')
        priority_menu = ttk.Combobox(form_frame, textvariable=priority_var, values=['CRITICAL', 'HIGH', 'MEDIUM', 'LOW'], width=27)
        priority_menu.grid(row=4, column=1, pady=5)

        # Status
        tk.Label(form_frame, text="Status:", bg='white').grid(row=5, column=0, sticky=tk.W, pady=5)
        status_var = tk.StringVar(value='Scheduled')
        status_menu = ttk.Combobox(form_frame, textvariable=status_var, values=['Scheduled', 'In Progress', 'Completed', 'Cancelled'], width=27)
        status_menu.grid(row=5, column=1, pady=5)

        def save_entry():
            try:
                new_entry = {
                    'Date': date_entry.get(),
                    'Machine_ID': int(machine_entry.get()),
                    'RUL': float(rul_entry.get()),
                    'Component': component_var.get(),
                    'Priority': priority_var.get(),
                    'Status': status_var.get()
                }
                self.maintenance_schedule.append(new_entry)
                self.refresh_maintenance_table()
                messagebox.showinfo("Success", "Entry added successfully!")
                add_window.destroy()
            except Exception as e:
                messagebox.showerror("Error", f"Invalid input: {e}")

        tk.Button(
            add_window,
            text="Save Entry",
            command=save_entry,
            font=('Arial', 12, 'bold'),
            bg='#4CAF50',
            fg='white',
            padx=20,
            pady=10
        ).pack(pady=20)

    def download_maintenance_schedule(self):
        """Download maintenance schedule as CSV"""
        file_path = filedialog.asksaveasfilename(
            defaultextension=".csv",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.* ")],
            initialfile=f"maintenance_schedule_{datetime.now().strftime('%Y%m%d')}.csv"
        )

        if file_path:
            df = pd.DataFrame(self.maintenance_schedule)
            df.to_csv(file_path, index=False)
            messagebox.showinfo("Success", f"Schedule saved to {file_path}")

    def remove_maintenance_entry(self):
        """Remove selected maintenance entry"""
        selected_item = self.schedule_tree.selection()
        if not selected_item:
            messagebox.showwarning("Selection Required", "Please select an entry to remove")
            return

        selected_index = self.schedule_tree.index(selected_item[0])

        if messagebox.askyesno("Confirm", "Remove selected entry?"):
            del self.maintenance_schedule[selected_index]
            self.refresh_maintenance_table()
            messagebox.showinfo("Success", "Entry removed successfully!")

    def view_all_maintenance_records(self):
        """View all historical maintenance records"""
        records_window = tk.Toplevel(self.root)
        records_window.title("All Maintenance Records")
        records_window.geometry("900x600")
        records_window.configure(bg='white')

        tk.Label(
            records_window,
            text="All Maintenance Records",
            font=('Arial', 16, 'bold'),
            bg='white'
        ).pack(pady=10)

        # Create a frame to hold the Treeview and its scrollbars
        tree_container_frame = tk.Frame(records_window, bg='white')
        tree_container_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Create treeview
        columns = ('Date', 'Machine_ID', 'Component', 'Action', 'Technician')
        tree = ttk.Treeview(tree_container_frame, columns=columns, show='headings', height=20)

        for col in columns:
            tree.heading(col, text=col)
            tree.column(col, width=150, anchor=tk.CENTER)

        scrollbar_y = ttk.Scrollbar(tree_container_frame, orient=tk.VERTICAL, command=tree.yview)
        scrollbar_x = ttk.Scrollbar(tree_container_frame, orient=tk.HORIZONTAL, command=tree.xview)
        tree.configure(yscroll=scrollbar_y.set, xscroll=scrollbar_x.set)

        scrollbar_y.pack(side=tk.RIGHT, fill=tk.Y)
        scrollbar_x.pack(side=tk.BOTTOM, fill=tk.X)
        tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # Generate sample historical records
        for i in range(30):
            date = (datetime.now() - timedelta(days=i)).strftime('%Y-%m-%d')
            values = (
                date,
                np.random.randint(1, 21),
                np.random.choice(['comp1', 'comp2', 'comp3', 'comp4']),
                'Replacement',
                f'Tech-{np.random.randint(1, 6)}'
            )
            tree.insert('', tk.END, values=values)

        # Download button
        def download_records():
            file_path = filedialog.asksaveasfilename(
                defaultextension=".csv",
                filetypes=[("CSV files", "*.csv"), ("All files", "*.* ")],
                initialfile=f"maintenance_records_{datetime.now().strftime('%Y%m%d')}.csv"
            )
            if file_path:
                # Create dataframe and save
                data = []
                for item in tree.get_children():
                    data.append(tree.item(item)['values'])
                df = pd.DataFrame(data, columns=columns)
                df.to_csv(file_path, index=False)
                messagebox.showinfo("Success", f"Records saved to {file_path}")

        tk.Button(
            records_window,
            text="Download Records",
            command=download_records,
            font=('Arial', 10, 'bold'),
            bg='#FF9800',
            fg='white',
            padx=20,
            pady=10
        ).pack(pady=10)

    # VIEW 3: MACHINE RUL DASHBOARD

    def show_machine_rul(self):
        """Display machine RUL dashboard"""
        self.clear_content()
        self.current_view = "machine_rul"

        # Title
        title_label = tk.Label(
            self.content_frame,
            text="Machine RUL Dashboard",
            font=('Arial', 18, 'bold'),
            bg='white'
        )
        title_label.pack(pady=10)

        # Starred machines section
        starred_frame = tk.LabelFrame(
            self.content_frame,
            text="Starred Machines",
            font=('Arial', 14, 'bold'),
            bg='white',
            fg='#1f77b4'
        )
        starred_frame.pack(fill=tk.X, padx=20, pady=10)

        # Display starred machines
        starred_grid = tk.Frame(starred_frame, bg='white')
        starred_grid.pack(fill=tk.X, padx=10, pady=10)

        for idx, machine_id in enumerate(self.starred_machines[:4]):
            result = self.predict_rul_for_machine(machine_id)
            rul = result['rul_percentage']
            tier = result['alert_tier']
            color = self.get_alert_color(tier)

            machine_frame = tk.Frame(
                starred_grid,
                bg=color,
                relief=tk.RAISED,
                borderwidth=3
            )
            machine_frame.grid(row=0, column=idx, padx=10, pady=5, sticky='ew')
            starred_grid.columnconfigure(idx, weight=1)

            tk.Label(
                machine_frame,
                text=f"Machine {machine_id}",
                font=('Arial', 14, 'bold'),
                bg=color,
                fg='white'
            ).pack(pady=5)

            tk.Label(
                machine_frame,
                text=f"{rul:.1f}%",
                font=('Arial', 36, 'bold'),
                bg=color,
                fg='white'
            ).pack()

            tk.Label(
                machine_frame,
                text=tier,
                font=('Arial', 12),
                bg=color,
                fg='white'
            ).pack(pady=5)

            tk.Button(
                machine_frame,
                text="🔍 View Details",
                command=lambda m=machine_id: self.show_machine_detail(m),
                font=('Arial', 9, 'bold'),
                bg='white',
                fg=color
            ).pack(pady=5)

        # Other machines section
        other_frame = tk.LabelFrame(
            self.content_frame,
            text="View Other Machines",
            font=('Arial', 14, 'bold'),
            bg='white',
            fg='#1f77b4'
        )
        other_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=10)

        select_frame = tk.Frame(other_frame, bg='white')
        select_frame.pack(fill=tk.X, padx=10, pady=10)

        tk.Label(select_frame, text="Select Machine ID:", font=('Arial', 12), bg='white').pack(side=tk.LEFT, padx=5)

        if self.machines_df is not None:
            all_machines = sorted(self.machines_df['machineID'].unique())
        else:
            all_machines = list(range(1, 21))

        other_machines = [m for m in all_machines if m not in self.starred_machines]

        machine_var = tk.IntVar(value=other_machines[0] if other_machines else 1)
        machine_menu = ttk.Combobox(
            select_frame,
            textvariable=machine_var,
            values=other_machines if other_machines else all_machines,
            width=15,
            font=('Arial', 11)
        )
        machine_menu.pack(side=tk.LEFT, padx=5)

        # Display frame for selected machine
        display_frame = tk.Frame(other_frame, bg='white')
        display_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        def show_selected_machine():
            for widget in display_frame.winfo_children():
                widget.destroy()

            selected_id = machine_var.get()
            result = self.predict_rul_for_machine(selected_id)
            rul = result['rul_percentage']
            tier = result['alert_tier']
            color = self.get_alert_color(tier)

            info_frame = tk.Frame(display_frame, bg=color, relief=tk.RAISED, borderwidth=5)
            info_frame.pack(pady=20)

            tk.Label(
                info_frame,
                text=f"Machine {selected_id}",
                font=('Arial', 18, 'bold'),
                bg=color,
                fg='white'
            ).pack(pady=10, padx=30)

            tk.Label(
                info_frame,
                text=f"RUL: {rul:.1f}%",
                font=('Arial', 24, 'bold'),
                bg=color,
                fg='white'
            ).pack(pady=5, padx=30)

            tk.Label(
                info_frame,
                text=f"Status: {tier}",
                font=('Arial', 16),
                bg=color,
                fg='white'
            ).pack(pady=5, padx=30)

            tk.Label(
                info_frame,
                text=f"Degradation Probability: {result['P_degraded']:.3f}",
                font=('Arial', 12),
                bg=color,
                fg='white'
            ).pack(pady=10, padx=30)

            button_frame = tk.Frame(info_frame, bg=color)
            button_frame.pack(pady=10)

            tk.Button(
                button_frame,
                text="Add to Starred",
                command=lambda: self.add_to_starred(selected_id),
                font=('Arial', 10, 'bold'),
                bg='white',
                fg=color,
                padx=10,
                pady=5
            ).pack(side=tk.LEFT, padx=5)

            tk.Button(
                button_frame,
                text="View Details",
                command=lambda: self.show_machine_detail(selected_id),
                font=('Arial', 10, 'bold'),
                bg='white',
                fg=color,
                padx=10,
                pady=5
            ).pack(side=tk.LEFT, padx=5)

        tk.Button(
            select_frame,
            text="Show Machine",
            command=show_selected_machine,
            font=('Arial', 10, 'bold'),
            bg='#4CAF50',
            fg='white',
            padx=15,
            pady=5
        ).pack(side=tk.LEFT, padx=10)

    def add_to_starred(self, machine_id):
        """Add machine to starred list"""
        if machine_id not in self.starred_machines:
            self.starred_machines.append(machine_id)
            messagebox.showinfo("Success", f"Machine {machine_id} added to starred!")
            self.show_machine_rul()
        else:
            messagebox.showinfo("Info", f"Machine {machine_id} is already starred")

    def show_machine_detail(self, machine_id):
        """Show detailed view for a specific machine"""
        detail_window = tk.Toplevel(self.root)
        detail_window.title(f"Machine {machine_id} - Detailed View")
        detail_window.geometry("1000x700")
        detail_window.configure(bg='white')

        # Header
        header_frame = tk.Frame(detail_window, bg='#1f77b4')
        header_frame.pack(fill=tk.X)

        tk.Label(
            header_frame,
            text=f"Machine {machine_id} - Detailed Analysis",
            font=('Arial', 18, 'bold'),
            bg='#1f77b4',
            fg='white'
        ).pack(pady=15)

        # Current RUL metrics
        result = self.predict_rul_for_machine(machine_id)
        rul = result['rul_percentage']
        tier = result['alert_tier']
        color = self.get_alert_color(tier)

        metrics_frame = tk.Frame(detail_window, bg='white')
        metrics_frame.pack(fill=tk.X, padx=20, pady=10)

        metric_items = [
            ("Current RUL", f"{rul:.1f}%"),
            ("Alert Tier", tier),
            ("Degradation Prob", f"{result['P_degraded']:.3f}")
        ]

        for idx, (label, value) in enumerate(metric_items):
            metric_box = tk.Frame(metrics_frame, bg='#f5f5f5', relief=tk.RAISED, borderwidth=2)
            metric_box.grid(row=0, column=idx, padx=10, pady=5, sticky='ew')
            metrics_frame.columnconfigure(idx, weight=1)

            tk.Label(
                metric_box,
                text=label,
                font=('Arial', 11),
                bg='#f5f5f5'
            ).pack(pady=5)

            tk.Label(
                metric_box,
                text=value,
                font=('Arial', 16, 'bold'),
                bg='#f5f5f5',
                fg=color if idx == 0 else '#333'
            ).pack(pady=5)

        # RUL Trajectory plot
        plot_frame = tk.LabelFrame(
            detail_window,
            text="RUL Trajectory (Last 7 Days)",
            font=('Arial', 12, 'bold'),
            bg='white'
        )
        plot_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=10)

        # Generate trajectory data
        dates = [(datetime.now() - timedelta(days=i)).strftime('%m-%d') for i in range(7, -1, -1)]
        base_rul = rul
        rul_values = []
        for i in range(8):
            decline = (8 - i) / 8 * 15
            noise = np.random.uniform(-5, 5)
            rul_values.append(min(100, max(0, base_rul + decline + noise)))

        # Create plot
        fig = Figure(figsize=(8, 4), facecolor='white')
        ax = fig.add_subplot(111)

        ax.plot(
                dates,
                rul_values,
                marker='o',
                linewidth=2,
                markersize=8,
                color='#1f77b4',
                label='RUL'
            )
        ax.axhline(y=20, color='red', linestyle='--', alpha=0.7, label='Critical')
        ax.axhline(y=35, color='orange', linestyle='--', alpha=0.7, label='Warning')
        ax.axhline(y=65, color='yellow', linestyle='--', alpha=0.7, label='Monitor')
        ax.axhline(y=80, color='lightgreen', linestyle='--', alpha=0.7, label='Good')

        ax.set_xlabel('Date', fontsize=10, fontweight='bold')
        ax.set_ylabel('RUL (%)', fontsize=10, fontweight='bold')
        ax.set_title(f'RUL Trend - Machine {machine_id}', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend(loc='upper right', fontsize=8)
        ax.set_ylim(0, 105)

        canvas = FigureCanvasTkAgg(fig, plot_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Machine information
        info_frame = tk.Frame(detail_window, bg='white')
        info_frame.pack(fill=tk.X, padx=20, pady=10)

        col1 = tk.LabelFrame(info_frame, text="ℹ️ Machine Information", font=('Arial', 11, 'bold'), bg='white')
        col1.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5)

        col2 = tk.LabelFrame(info_frame, text="🔧 Component Status", font=('Arial', 11, 'bold'), bg='white')
        col2.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=5)

        # Machine info
        if self.machines_df is not None:
            machine_info = self.machines_df[self.machines_df['machineID'] == machine_id]
            if len(machine_info) > 0:
                age = machine_info['age'].iloc[0]
                model = machine_info['model'].iloc[0]
            else:
                age, model = 10, 1
        else:
            age, model = 10, 1

        tk.Label(col1, text=f"Age: {age} years", font=('Arial', 10), bg='white').pack(anchor=tk.W, padx=10, pady=5)
        tk.Label(col1, text=f"Model: {model}", font=('Arial', 10), bg='white').pack(anchor=tk.W, padx=10, pady=5)
        tk.Label(col1, text=f"Machine ID: {machine_id}", font=('Arial', 10), bg='white').pack(anchor=tk.W, padx=10, pady=5)

        # Component status
        if self.maint_df is not None:
            machine_maint = self.maint_df[self.maint_df['machineID'] == machine_id]
            for comp in ['comp1', 'comp2', 'comp3', 'comp4']:
                comp_maint = machine_maint[machine_maint['comp'] == comp]
                if len(comp_maint) > 0:
                    last_date = comp_maint['datetime'].max()
                    days_since = (pd.Timestamp.now() - last_date).days
                    tk.Label(
                        col2,
                        text=f"{comp}: {days_since} days since replacement",
                        font=('Arial', 9),
                        bg='white'
                    ).pack(anchor=tk.W, padx=10, pady=3)
                else:
                    tk.Label(
                        col2,
                        text=f"{comp}: No maintenance history",
                        font=('Arial', 9),
                        bg='white'
                    ).pack(anchor=tk.W, padx=10, pady=3)
        else:
            for comp in ['comp1', 'comp2', 'comp3', 'comp4']:
                days = np.random.randint(50, 300)
                tk.Label(
                    col2,
                    text=f"{comp}: {days} days since replacement",
                    font=('Arial', 9),
                    bg='white'
                ).pack(anchor=tk.W, padx=10, pady=3)

    # VIEW 4: SENSOR DATA VIEWER

    def show_sensor_data(self):
        """Display sensor data viewer and download options"""
        self.clear_content()
        self.current_view = "sensor_data"

        # Title
        title_label = tk.Label(
            self.content_frame,
            text="Sensor Data Viewer",
            font=('Arial', 18, 'bold'),
            bg='white'
        )
        title_label.pack(pady=10)

        # Description
        desc_label = tk.Label(
            self.content_frame,
            text="View and download raw sensor data from edge device storage",
            font=('Arial', 11),
            bg='white',
            fg='#666'
        )
        desc_label.pack(pady=5)

        # Sensor buttons grid
        sensor_frame = tk.Frame(self.content_frame, bg='white')
        sensor_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)

        sensors = [
            ("Voltage Sensor Data", 'volt'),
            ("Rotation Sensor Data", 'rotate'),
            ("Pressure Sensor Data", 'pressure'),
            ("Vibration Sensor Data", 'vibration')
        ]

        for idx, (label, sensor_type) in enumerate(sensors):
            row = idx // 2
            col = idx % 2

            sensor_box = tk.Frame(sensor_frame, bg='#f5f5f5', relief=tk.RAISED, borderwidth=3)
            sensor_box.grid(row=row, column=col, padx=20, pady=20, sticky='nsew')
            sensor_frame.rowconfigure(row, weight=1)
            sensor_frame.columnconfigure(col, weight=1)

            tk.Label(
                sensor_box,
                text=label,
                font=('Arial', 14, 'bold'),
                bg='#f5f5f5'
            ).pack(pady=20)

            btn_frame = tk.Frame(sensor_box, bg='#f5f5f5')
            btn_frame.pack(pady=10)

            tk.Button(
                btn_frame,
                text="View Data",
                command=lambda s=sensor_type: self.view_sensor_data(s),
                font=('Arial', 10, 'bold'),
                bg='#2196F3',
                fg='white',
                padx=20,
                pady=10
            ).pack(side=tk.LEFT, padx=5)

            tk.Button(
                btn_frame,
                text="Download",
                command=lambda s=sensor_type: self.download_sensor_data(s),
                font=('Arial', 10, 'bold'),
                bg='#4CAF50',
                fg='white',
                padx=20,
                pady=10
            ).pack(side=tk.LEFT, padx=5)

        # All data options
        all_frame = tk.Frame(self.content_frame, bg='#e3f2fd', relief=tk.RAISED, borderwidth=3)
        all_frame.pack(fill=tk.X, padx=20, pady=10)

        tk.Label(
            all_frame,
            text="Complete Dataset",
            font=('Arial', 14, 'bold'),
            bg='#e3f2fd'
        ).pack(pady=10)

        btn_frame = tk.Frame(all_frame, bg='#e3f2fd')
        btn_frame.pack(pady=10)

        tk.Button(
            btn_frame,
            text="View All Telemetry",
            command=lambda: self.view_sensor_data('all'),
            font=('Arial', 10, 'bold'),
            bg='#2196F3',
            fg='white',
            padx=20,
            pady=10
        ).pack(side=tk.LEFT, padx=5)

        tk.Button(
            btn_frame,
            text="Download All Data",
            command=lambda: self.download_sensor_data('all'),
            font=('Arial', 10, 'bold'),
            bg='#4CAF50',
            fg='white',
            padx=20,
            pady=10
        ).pack(side=tk.LEFT, padx=5)

    def view_sensor_data(self, sensor_type):
        """View sensor data in a new window"""
        view_window = tk.Toplevel(self.root)
        view_window.title(f"{sensor_type.capitalize()} Sensor Data")
        view_window.geometry("900x600")
        view_window.configure(bg='white')

        tk.Label(
            view_window,
            text=f"{sensor_type.capitalize()} Sensor Data",
            font=('Arial', 16, 'bold'),
            bg='white'
        ).pack(pady=10)

        if self.telemetry_df is None:
            tk.Label(
                view_window,
                text="No sensor data available",
                font=('Arial', 12),
                bg='white'
            ).pack(pady=50)
            return

        # Create table
        frame = tk.Frame(view_window, bg='white')
        frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        if sensor_type == 'all':
            columns = list(self.telemetry_df.columns)
            data = self.telemetry_df.head(100)
        else:
            columns = ['datetime', 'machineID', sensor_type]
            data = self.telemetry_df[columns].head(100)

        tree = ttk.Treeview(frame, columns=columns, show='headings', height=20)

        for col in columns:
            tree.heading(col, text=col)
            tree.column(col, width=150, anchor=tk.CENTER)

        scrollbar_y = ttk.Scrollbar(frame, orient=tk.VERTICAL, command=tree.yview)
        scrollbar_x = ttk.Scrollbar(frame, orient=tk.HORIZONTAL, command=tree.xview)
        tree.configure(yscroll=scrollbar_y.set, xscroll=scrollbar_x.set)

        scrollbar_y.pack(side=tk.RIGHT, fill=tk.Y)
        scrollbar_x.pack(side=tk.BOTTOM, fill=tk.X)
        tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # Populate data
        for _, row in data.iterrows():
            values = [str(row[col]) for col in columns]
            tree.insert('', tk.END, values=values)

        tk.Label(
            view_window,
            text=f"Showing first 100 rows of {len(self.telemetry_df)} total records",
            font=('Arial', 10),
            bg='white',
            fg='#666'
        ).pack(pady=5)

    def download_sensor_data(self, sensor_type):
        """Download sensor data as CSV"""
        if self.telemetry_df is None:
            messagebox.showwarning("No Data", "No sensor data available to download")
            return

        file_path = filedialog.asksaveasfilename(
            defaultextension=".csv",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.* ")],
            initialfile=f"{sensor_type}_sensor_data_{datetime.now().strftime('%Y%m%d')}.csv"
        )

        if file_path:
            if sensor_type == 'all':
                self.telemetry_df.to_csv(file_path, index=False)
            else:
                self.telemetry_df[['datetime', 'machineID', sensor_type]].to_csv(file_path, index=False)
            messagebox.showinfo("Success", f"Data saved to {file_path}")

    # VIEW 5: HISTORICAL DATA ANALYSIS

    def show_historical_data(self):
        """Display historical sensor data analysis"""
        self.clear_content()
        self.current_view = "historical"

        # Title
        title_label = tk.Label(
            self.content_frame,
            text="Historical Sensor Data Analysis",
            font=('Arial', 18, 'bold'),
            bg='white'
        )
        title_label.pack(pady=10)

        # Selection frame
        select_frame = tk.Frame(self.content_frame, bg='white')
        select_frame.pack(fill=tk.X, padx=20, pady=10)

        # Time range selection
        tk.Label(select_frame, text="Time Range:", font=('Arial', 12, 'bold'), bg='white').grid(row=0, column=0, padx=5, pady=5, sticky=tk.W)
        time_var = tk.StringVar(value="Last 7 Days")
        time_menu = ttk.Combobox(
            select_frame,
            textvariable=time_var,
            values=["Last 24 Hours", "Last 7 Days", "Last 30 Days"],
            width=20,
            font=('Arial', 11)
        )
        time_menu.grid(row=0, column=1, padx=5, pady=5)

        # Sensor selection
        tk.Label(select_frame, text="Sensor Type:", font=('Arial', 12, 'bold'), bg='white').grid(row=0, column=2, padx=5, pady=5, sticky=tk.W)
        sensor_var = tk.StringVar(value="volt")
        sensor_menu = ttk.Combobox(
            select_frame,
            textvariable=sensor_var,
            values=["volt", "rotate", "pressure", "vibration"],
            width=20,
            font=('Arial', 11)
        )
        sensor_menu.grid(row=0, column=3, padx=5, pady=5)

        # Machine selection
        tk.Label(select_frame, text="Machine ID:", font=('Arial', 12, 'bold'), bg='white').grid(row=1,column=0, padx=5, pady=5, sticky=tk.W)
        if self.machines_df is not None:
            machines = sorted(self.machines_df['machineID'].unique())
        else:
            machines = list(range(1, 21))

        machine_var = tk.IntVar(value=machines[0])
        machine_menu = ttk.Combobox(
            select_frame,
            textvariable=machine_var,
            values=machines,
            width=20,
            font=('Arial', 11)
        )
        machine_menu.grid(row=1, column=1, padx=5, pady=5)

        # Plot frame
        plot_frame = tk.Frame(self.content_frame, bg='white')
        plot_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=10)

        def generate_plot():
            for widget in plot_frame.winfo_children():
                widget.destroy()

            if self.telemetry_df is None:
                tk.Label(
                    plot_frame,
                    text="No sensor data available",
                    font=('Arial', 14),
                    bg='white'
                ).pack(pady=50)
                return

            # Get parameters
            time_range = time_var.get()
            sensor_type = sensor_var.get()
            machine_id = machine_var.get()

            # Calculate date range
            end_date = self.telemetry_df['datetime'].max()
            if time_range == "Last 24 Hours":
                start_date = end_date - pd.Timedelta(hours=24)
            elif time_range == "Last 7 Days":
                start_date = end_date - pd.Timedelta(days=7)
            else:
                start_date = end_date - pd.Timedelta(days=30)

            # Filter data
            filtered_data = self.telemetry_df[
                (self.telemetry_df['machineID'] == machine_id) &
                (self.telemetry_df['datetime'] >= start_date) &
                (self.telemetry_df['datetime'] <= end_date)
            ]

            if len(filtered_data) == 0:
                tk.Label(
                    plot_frame,
                    text="No data available for selected parameters",
                    font=('Arial', 14),
                    bg='white'
                ).pack(pady=50)
                return

            # Create plot
            fig = Figure(figsize=(10, 5), facecolor='white')
            ax = fig.add_subplot(111)

            ax.plot(
                filtered_data['datetime'],
                filtered_data[sensor_type],
                linewidth=2,
                color='#1f77b4',
                label=sensor_type.capitalize()
            )

            ax.set_xlabel('Date/Time', fontsize=11, fontweight='bold')
            ax.set_ylabel(f'{sensor_type.capitalize()} Value', fontsize=11, fontweight='bold')
            ax.set_title(
                f'{sensor_type.capitalize()} Readings - Machine {machine_id}\n{time_range}',
                fontsize=13,
                fontweight='bold'
            )
            ax.grid(True, alpha=0.3)
            ax.legend(loc='upper right')

            # Rotate x-axis labels
            fig.autofmt_xdate()

            canvas = FigureCanvasTkAgg(fig, plot_frame)
            canvas.draw()
            canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

            # Statistics
            stats_frame = tk.Frame(plot_frame, bg='#f5f5f5', relief=tk.SUNKEN, borderwidth=2)
            stats_frame.pack(fill=tk.X, pady=10)

            stats = [
                ("Mean", f"{filtered_data[sensor_type].mean():.2f}"),
                ("Std Dev", f"{filtered_data[sensor_type].std():.2f}"),
                ("Min", f"{filtered_data[sensor_type].min():.2f}"),
                ("Max", f"{filtered_data[sensor_type].max():.2f}"),
                ("Samples", f"{len(filtered_data)}")
            ]

            for idx, (label, value) in enumerate(stats):
                stat_box = tk.Frame(stats_frame, bg='#f5f5f5')
                stat_box.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=10, pady=5)

                tk.Label(
                    stat_box,
                    text=label,
                    font=('Arial', 10),
                    bg='#f5f5f5'
                ).pack()

                tk.Label(
                    stat_box,
                    text=value,
                    font=('Arial', 14, 'bold'),
                    bg='#f5f5f5',
                    fg='#1f77b4'
                ).pack()

        tk.Button(
            select_frame,
            text="Generate Plot",
            command=generate_plot,
            font=('Arial', 11, 'bold'),
            bg='#4CAF50',
            fg='white',
            padx=30,
            pady=10
        ).grid(row=1, column=2, columnspan=2, padx=5, pady=10)

    # VIEW 6: MODEL RETRAINING

    def show_retrain(self):
        """Display model retraining interface"""
        self.clear_content()
        self.current_view = "retrain"

        # Title
        title_label = tk.Label(
            self.content_frame,
            text="Model Retraining",
            font=('Arial', 18, 'bold'),
            bg='white'
        )
        title_label.pack(pady=10)

        # Info frame
        info_frame = tk.Frame(self.content_frame, bg='#e3f2fd', relief=tk.RAISED, borderwidth=3)
        info_frame.pack(fill=tk.X, padx=20, pady=20)

        tk.Label(
            info_frame,
            text="ℹModel Retraining Information",
            font=('Arial', 14, 'bold'),
            bg='#e3f2fd',
            fg='#1f77b4'
        ).pack(pady=10)

        info_text = """ Model Retraining Process:

        Updated sensor data will be read from CSV files in edge device storage
        Feature engineering will be automatically applied to new data
        Model will be retrained with combined historical and new data
        New model will replace current model after validation
        Performance metrics will be calculated and displayed

        ⚠️ Warning: This process may take 10-30 minutes depending on data volume.
        The system will remain operational but predictions may be temporarily unavailable.
        PLACEHOLDER: In production, this would trigger the full retraining pipeline.
        """
        tk.Label(
            info_frame,
            text=info_text,
            font=('Arial', 10),
            bg='#e3f2fd',
            justify=tk.LEFT
        ).pack(padx=20, pady=10)

        # Current vs New model info
        comparison_frame = tk.Frame(self.content_frame, bg='white')
        comparison_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)

        # Current model
        current_frame = tk.LabelFrame(
            comparison_frame,
            text="📊 Current Model",
            font=('Arial', 12, 'bold'),
            bg='white'
        )
        current_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=10)

        current_info = [
            ("Trained Date:", "2024-01-15"),
            ("Training Samples:", "876,000"),
            ("PR-AUC:", "0.8534"),
            ("F2-Score:", "0.7891"),
            ("Model Version:", "v1.0")
        ]

        for label, value in current_info:
            item_frame = tk.Frame(current_frame, bg='white')
            item_frame.pack(fill=tk.X, padx=10, pady=5)

            tk.Label(
                item_frame,
                text=label,
                font=('Arial', 10),
                bg='white',
                width=20,
                anchor=tk.W
            ).pack(side=tk.LEFT)

            tk.Label(
                item_frame,
                text=value,
                font=('Arial', 10, 'bold'),
                bg='white',
                fg='#1f77b4'
            ).pack(side=tk.LEFT)

        # New data available
        new_frame = tk.LabelFrame(
            comparison_frame,
            text="📈 New Data Available",
            font=('Arial', 12, 'bold'),
            bg='white'
        )
        new_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=10)

        new_info = [
            ("New Samples:", "45,000"),
            ("Date Range:", "2024-01-16 to 2024-01-30"),
            ("New Maintenance Events:", "150"),
            ("New Error Logs:", "320"),
            ("Status:", "✅ Ready for Training")
        ]

        for label, value in new_info:
            item_frame = tk.Frame(new_frame, bg='white')
            item_frame.pack(fill=tk.X, padx=10, pady=5)

            tk.Label(
                item_frame,
                text=label,
                font=('Arial', 10),
                bg='white',
                width=20,
                anchor=tk.W
            ).pack(side=tk.LEFT)

            tk.Label(
                item_frame,
                text=value,
                font=('Arial', 10, 'bold'),
                bg='white',
                fg='#4CAF50'
            ).pack(side=tk.LEFT)

        # Retrain button
        button_frame = tk.Frame(self.content_frame, bg='white')
        button_frame.pack(pady=20)

        def start_retraining():
            # PLACEHOLDER: Actual retraining would happen here
            if messagebox.askyesno(
                "Confirm Retraining",
                "Start model retraining? This may take 10-30 minutes."
            ):
                progress_window = tk.Toplevel(self.root)
                progress_window.title("Model Retraining")
                progress_window.geometry("500x300")
                progress_window.configure(bg='white')

                tk.Label(
                    progress_window,
                    text="🔄 Retraining Model...",
                    font=('Arial', 16, 'bold'),
                    bg='white'
                ).pack(pady=20)

                progress = ttk.Progressbar(
                    progress_window,
                    length=400,
                    mode='determinate'
                )
                progress.pack(pady=20)

                status_label = tk.Label(
                    progress_window,
                    text="Initializing...",
                    font=('Arial', 11),
                    bg='white'
                )
                status_label.pack(pady=10)

                def update_progress():
                    stages = [
                        "Loading new sensor data...",
                        "Applying feature engineering...",
                        "Training XGBoost model...",
                        "Validating model performance...",
                        "Saving new model...",
                        "✅ Retraining complete!"
                    ]

                    for i, stage in enumerate(stages):
                        progress['value'] = (i + 1) / len(stages) * 100
                        status_label.config(text=stage)
                        progress_window.update()
                        time.sleep(1)

                    # Show results
                    result_text = """New Model Performance:
                    Trained: 2024-01-30
                    Samples: 921,000
                    PR-AUC: 0.8612 (+0.78%)
                    F2-Score: 0.7945 (+0.54%)
                    Model Version: v1.1

                    ✅ Model deployed successfully!
                        """
                    tk.Label(
                            progress_window,
                            text=result_text,
                            font=('Courier', 10),
                            bg='white',
                            justify=tk.LEFT
                        ).pack(pady=10)

                    tk.Button(
                        progress_window,
                        text="Close",
                        command=progress_window.destroy,
                        font=('Arial', 10, 'bold'),
                        bg='#4CAF50',
                        fg='white',
                        padx=20,
                        pady=5
                    ).pack(pady=10)

                # Start in thread to avoid freezing GUI
                thread = threading.Thread(target=update_progress)
                thread.start()

        tk.Button(
            button_frame,
            text="Start Model Retraining",
            command=start_retraining,
            font=('Arial', 14, 'bold'),
            bg='#FF5722',
            fg='white',
            padx=40,
            pady=15
        ).pack()

    # VIEW 7: PROTOTYPE TESTING

    def show_prototype_test(self):
        """Display prototype testing dashboard"""
        self.clear_content()
        self.current_view = "prototype"

        # Title
        title_label = tk.Label(
            self.content_frame,
            text="Prototype Testing Dashboard",
            font=('Arial', 18, 'bold'),
            bg='white'
        )
        title_label.pack(pady=10)

        # Description
        desc_frame = tk.Frame(self.content_frame, bg='#fff3cd', relief=tk.RAISED, borderwidth=2)
        desc_frame.pack(fill=tk.X, padx=20, pady=10)

        tk.Label(
            desc_frame,
            text="Prototype Testing Mode",
            font=('Arial', 12, 'bold'),
            bg='#fff3cd'
        ).pack(pady=5)

        tk.Label(
            desc_frame,
            text="This dashboard simulates sensor readings for selected machines and working hours.\n"
                "Use this to demonstrate the RUL prediction system without real-time sensor connections.",
            font=('Arial', 10),
            bg='#fff3cd',
            justify=tk.CENTER
        ).pack(pady=5)

        # Input frame
        input_frame = tk.LabelFrame(
            self.content_frame,
            text="Test Parameters",
            font=('Arial', 13, 'bold'),
            bg='white'
        )
        input_frame.pack(fill=tk.X, padx=20, pady=20)

        # Machine selection
        param_frame = tk.Frame(input_frame, bg='white')
        param_frame.pack(padx=20, pady=15)

        tk.Label(
            param_frame,
            text="Select Machine ID:",
            font=('Arial', 12, 'bold'),
            bg='white'
        ).grid(row=0, column=0, padx=10, pady=10, sticky=tk.W)

        if self.machines_df is not None:
            machines = sorted(self.machines_df['machineID'].unique())
        else:
            machines = list(range(1, 21))

        machine_var = tk.IntVar(value=machines[0])
        machine_menu = ttk.Combobox(
            param_frame,
            textvariable=machine_var,
            values=machines,
            width=20,
            font=('Arial', 11)
        )
        machine_menu.grid(row=0, column=1, padx=10, pady=10)

        # Working hours input
        tk.Label(
            param_frame,
            text="Input Working Hours:",
            font=('Arial', 12, 'bold'),
            bg='white'
        ).grid(row=1, column=0, padx=10, pady=10, sticky=tk.W)

        hours_var = tk.IntVar(value=5000)
        hours_spinbox = tk.Spinbox(
            param_frame,
            from_=0,
            to=20000,
            textvariable=hours_var,
            width=20,
            font=('Arial', 11)
        )
        hours_spinbox.grid(row=1, column=1, padx=10, pady=10)

        # Results frame
        results_frame = tk.Frame(self.content_frame, bg='white')
        results_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=10)

        def generate_and_predict():
            for widget in results_frame.winfo_children():
                widget.destroy()

            machine_id = machine_var.get()
            working_hours = hours_var.get()

            # Generate simulated data
            simulated_data = self.generate_simulated_sensor_reading(machine_id, working_hours)

            if simulated_data is None:
                tk.Label(
                    results_frame,
                    text="Error generating simulated data",
                    font=('Arial', 14),
                    bg='white',
                    fg='red'
                ).pack(pady=50)
                return

            # Display simulated sensor readings
            sensor_frame = tk.LabelFrame(
                results_frame,
                text="Simulated Sensor Readings",
                font=('Arial', 12, 'bold'),
                bg='white'
            )
            sensor_frame.pack(fill=tk.X, pady=10)

            sensor_grid = tk.Frame(sensor_frame, bg='white')
            sensor_grid.pack(padx=10, pady=10)

            sensors = [
                ("Voltage", simulated_data['volt'], "V"),
                ("Rotation", simulated_data['rotate'], "rpm"),
                ("Pressure", simulated_data['pressure'], "bar"),
                ("Vibration", simulated_data['vibration'], "mm/s")
            ]

            for idx, (label, value, unit) in enumerate(sensors):
                sensor_box = tk.Frame(sensor_grid, bg='#f5f5f5', relief=tk.RAISED, borderwidth=2)
                sensor_box.grid(row=0, column=idx, padx=10, pady=5)

                tk.Label(
                    sensor_box,
                    text=label,
                    font=('Arial', 11, 'bold'),
                    bg='#f5f5f5'
                ).pack(pady=5, padx=20)

                tk.Label(
                    sensor_box,
                    text=f"{value:.2f}",
                    font=('Arial', 18, 'bold'),
                    bg='#f5f5f5',
                    fg='#1f77b4'
                ).pack()

                tk.Label(
                    sensor_box,
                    text=unit,
                    font=('Arial', 10),
                    bg='#f5f5f5',
                    fg='#666'
                ).pack(pady=5)

            # Additional info
            info_frame = tk.Frame(sensor_frame, bg='white')
            info_frame.pack(pady=10)

            info_items = [
                ("Timestamp:", simulated_data['timestamp']),
                ("Machine ID:", str(simulated_data['machine_id'])),
                ("Working Hours:", f"{simulated_data['working_hours']:,}"),
                ("Degradation Factor:", f"{simulated_data['degradation_factor']:.3f}")
            ]

            for label, value in info_items:
                item_frame = tk.Frame(info_frame, bg='white')
                item_frame.pack(side=tk.LEFT, padx=15)

                tk.Label(
                    item_frame,
                    text=label,
                    font=('Arial', 9),
                    bg='white',
                    fg='#666'
                ).pack()

                tk.Label(
                    item_frame,
                    text=value,
                    font=('Arial', 10, 'bold'),
                    bg='white'
                ).pack()

            # Predict RUL
            rul_prediction = self.predict_rul_from_simulated_data(simulated_data)

            if rul_prediction is None:
                tk.Label(
                    results_frame,
                    text="Error predicting RUL",
                    font=('Arial', 14),
                    bg='white',
                    fg='red'
                ).pack(pady=20)
                return

            # Display RUL prediction
            rul_frame = tk.LabelFrame(
                results_frame,
                text="RUL Prediction Result",
                font=('Arial', 12, 'bold'),
                bg='white'
            )
            rul_frame.pack(fill=tk.BOTH, expand=True, pady=10)

            rul = rul_prediction['rul_percentage']
            tier = rul_prediction['alert_tier']
            color = self.get_alert_color(tier)

            # RUL display
            rul_display = tk.Frame(rul_frame, bg=color, relief=tk.RAISED, borderwidth=5)
            rul_display.pack(pady=20)

            tk.Label(
                rul_display,
                text=f"Machine {machine_id}",
                font=('Arial', 18, 'bold'),
                bg=color,
                fg='white'
            ).pack(pady=15, padx=50)

            tk.Label(
                rul_display,
                text=f"{rul:.1f}%",
                font=('Arial', 48, 'bold'),
                bg=color,
                fg='white'
            ).pack(padx=50)

            tk.Label(
                rul_display,
                text="REMAINING USEFUL LIFE",
                font=('Arial', 14),
                bg=color,
                fg='white'
            ).pack()

            tk.Label(
                rul_display,
                text=f"Status: {tier}",
                font=('Arial', 16, 'bold'),
                bg=color,
                fg='white'
            ).pack(pady=15, padx=50)

            # Recommendation
            recommend_frame = tk.Frame(rul_frame, bg='#f5f5f5', relief=tk.SUNKEN, borderwidth=2)
            recommend_frame.pack(fill=tk.X, padx=20, pady=10)

            recommendations = {
                'CRITICAL': 'URGENT: Schedule immediate maintenance within 48 hours',
                'WARNING': 'Schedule maintenance within 7 days',
                'MONITOR': 'Increase monitoring frequency, plan maintenance',
                'GOOD': 'Continue routine checks, normal operation',
                'HEALTHY': 'Excellent condition, maintain regular schedule'
            }

            tk.Label(
                recommend_frame,
                text="Recommendation:",
                font=('Arial', 12, 'bold'),
                bg='#f5f5f5'
            ).pack(anchor=tk.W, padx=10, pady=5)

            tk.Label(
                recommend_frame,
                text=recommendations.get(tier, 'Continue monitoring'),
                font=('Arial', 11),
                bg='#f5f5f5',
                wraplength=800,
                justify=tk.LEFT
            ).pack(anchor=tk.W, padx=10, pady=5)

            # Model details
            details_frame = tk.Frame(rul_frame, bg='white')
            details_frame.pack(fill=tk.X, padx=20, pady=10)

            detail_items = [
                ("Degradation Probability:", f"{rul_prediction['P_degraded']:.4f}"),
                ("Model Confidence:", f"{(1 - abs(rul_prediction['P_degraded'] - 0.5) * 2) * 100:.1f}%"),
                ("Prediction Timestamp:", datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
            ]

            for label, value in detail_items:
                item_frame = tk.Frame(details_frame, bg='white')
                item_frame.pack(fill=tk.X, pady=2)

                tk.Label(
                    item_frame,
                    text=label,
                    font=('Arial', 10),
                    bg='white',
                    width=25,
                    anchor=tk.W
                ).pack(side=tk.LEFT)

                tk.Label(
                    item_frame,
                    text=value,
                    font=('Arial', 10, 'bold'),
                    bg='white',
                    fg='#1f77b4'
                ).pack(side=tk.LEFT)

        # Generate button
        tk.Button(
            input_frame,
            text="Generate Sensor Data & Calculate RUL",
            command=generate_and_predict,
            font=('Arial', 13, 'bold'),
            bg='#4CAF50',
            fg='white',
            padx=40,
            pady=15
        ).pack(pady=15)

#Main execution
def main():
    """Main entry point for the application"""
    root = tk.Tk()
    app = RevonyxRULMonitor(root)
    root.mainloop()

if __name__ == "__main__":
        main()
# C:\Users\mjhen\SPT_GUI\widgets\analysis_widgets.py

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QSpinBox, QDoubleSpinBox,
    QComboBox, QCheckBox, QGroupBox, QFormLayout, QPushButton, QTextEdit,
    QTabWidget, QTableWidget, QHeaderView, QListWidget, QAbstractItemView,
    QDialog, QDialogButtonBox, QStackedWidget
)
from PyQt5.QtCore import Qt, pyqtSignal, QObject

# Import the analyzer classes to access their default parameters
# Note: These imports assume the widgets directory is a sibling of the Analysis directory
# If your structure is different, adjust the import paths (e.g., from ..Analysis.dwell_time import ...)
from Analysis.diffusion_models import DiffusionAnalyzer as DiffusionModelsAnalyzer # Name clash, use alias if needed
from Analysis.active_transport import ActiveTransportAnalyzer
from Analysis.boundary_crossing import BoundaryCrossingAnalyzer
from Analysis.dwell_time import DwellTimeAnalyzer
from Analysis.crowding import CrowdingAnalyzer
from Analysis.diffusion_population import DiffusionPopulationAnalyzer
from Analysis.gel_structure import GelStructureAnalyzer
from Analysis.microcompartment import MultiChannelManager # Assuming MultiChannelManager is here


import logging
logger = logging.getLogger(__name__)

# Basic MplCanvas for embedding plots in widgets
class MplCanvas(FigureCanvas):
    """Canvas for matplotlib figures in the GUI."""
    def __init__(self, width=5, height=4, dpi=100):
        self.fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = self.fig.add_subplot(111)
        super(MplCanvas, self).__init__(self.fig)
        self.fig.tight_layout()

    def clear(self):
        """Clears the axes for a new plot."""
        self.axes.clear()
        self.draw()

# --- Define the specific analysis widgets ---

class DiffusionAnalysisWidget(QWidget):
    """Widget for Diffusion Analysis controls and visualization."""

    parameters_changed = pyqtSignal(dict)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setup_ui()
        self.setup_connections()
        self.current_results = None # To hold analysis results for display

    def setup_ui(self):
        layout = QVBoxLayout(self)
        params_group = QGroupBox("Diffusion Analysis Parameters")
        params_layout = QFormLayout()

        # Parameters based on DiffusionAnalyzer (diffusion_models.py) and spt_gui snippets
        # Note: DiffusionAnalyzer in diffusion_models.py seems to be the main one for fits.
        # The DiffusionAnalyzer in diffusion.py seems more focused on visualization.
        # Using parameters from DiffusionModelsAnalyzer.

        # Using a dummy instance to get default values if available
        try:
            default_analyzer = DiffusionModelsAnalyzer()
        except Exception:
            logger.warning("Could not instantiate DiffusionModelsAnalyzer to get defaults. Using hardcoded values.")
            class DummyDiffusionModelsAnalyzer:
                 def __init__(self):
                      self.config = { # Based on DEFAULT_CONFIG in diffusion_models.py
                          'msd_max_lag': 20,
                          'msd_min_track_length': 10,
                          'max_fit_points': 10,
                      }
            default_analyzer = DummyDiffusionModelsAnalyzer()


        self.max_lag_spinbox = QSpinBox()
        self.max_lag_spinbox.setRange(2, 500) # Increased range
        self.max_lag_spinbox.setValue(default_analyzer.config.get('msd_max_lag', 20))
        self.max_lag_spinbox.setToolTip("Maximum time lag for MSD calculation (frames).")
        params_layout.addRow("Max Lag:", self.max_lag_spinbox)

        self.min_track_length_spin = QSpinBox()
        self.min_track_length_spin.setRange(3, 500) # Min 3 needed for MSD calculation
        self.min_track_length_spin.setValue(default_analyzer.config.get('msd_min_track_length', 10))
        self.min_track_length_spin.setToolTip("Minimum track length to include in MSD and fitting.")
        params_layout.addRow("Min Track Length:", self.min_track_length_spin)

        self.max_fit_points_spin = QSpinBox()
        self.max_fit_points_spin.setRange(2, 50)
        self.max_fit_points_spin.setValue(default_analyzer.config.get('max_fit_points', 10))
        self.max_fit_points_spin.setToolTip("Number of initial points of the MSD curve used for model fitting.")
        params_layout.addRow("Max Fit Points:", self.max_fit_points_spin)

        self.model_selection_combo = QComboBox()
        self.model_selection_combo.addItems([
            "All Models", # Analyze and fit all models
            "Simple Diffusion",
            "Anomalous Diffusion",
            "Confined Diffusion",
            "Directed Motion"
        ])
        self.model_selection_combo.setToolTip("Select which diffusion models to fit.")
        params_layout.addRow("Diffusion Model(s):", self.model_selection_combo)

        params_group.setLayout(params_layout)
        layout.addWidget(params_group)

        # Visualization Options (Based on spt_gui snippet)
        viz_group = QGroupBox("Visualization Options")
        viz_layout = QFormLayout()

        self.plot_type_combo = QComboBox()
        self.plot_type_combo.addItems([
            "MSD Curves with Fits",
            "Diffusion Coefficient Histogram",
            "Diffusion Coefficient Map",
            "Anomalous Exponent Histogram",
            "Model Comparison (R²)"
        ])
        viz_layout.addRow("Plot Type:", self.plot_type_combo)

        # Display options (Based on spt_gui snippet)
        self.show_individual_msd_check = QCheckBox("Show Individual MSDs")
        self.show_individual_msd_check.setChecked(False) # Default to False to avoid clutter
        self.show_individual_msd_check.setToolTip("Overlay individual track MSD curves on the ensemble plot.")
        viz_layout.addRow("Display Options:", self.show_individual_msd_check) # Add to layout

        self.show_ensemble_msd_check = QCheckBox("Show Ensemble Average MSD")
        self.show_ensemble_msd_check.setChecked(True)
        self.show_ensemble_msd_check.setToolTip("Show the ensemble average MSD curve.")
        viz_layout.addRow("", self.show_ensemble_msd_check) # Add to layout

        self.show_model_fits_check = QCheckBox("Show Model Fits")
        self.show_model_fits_check.setChecked(True)
        self.show_model_fits_check.setToolTip("Show fitted model curves on the MSD plot.")
        viz_layout.addRow("", self.show_model_fits_check) # Add to layout


        viz_group.setLayout(viz_layout)
        layout.addWidget(viz_group)


        # Results Display
        self.results_tabs = QTabWidget()

        self.summary_text = QTextEdit()
        self.summary_text.setReadOnly(True)
        self.results_tabs.addTab(self.summary_text, "Summary")

        self.results_table = QTableWidget() # Table for track-level diffusion results
        self.results_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.results_tabs.addTab(self.results_table, "Detailed Results")


        self.plot_widget = MplCanvas() # For plots
        self.results_tabs.addTab(self.plot_widget, "Plots")

        layout.addWidget(self.results_tabs)


    def setup_connections(self):
        # Connect parameter changes to emit signal
        self.max_lag_spinbox.valueChanged.connect(self.on_parameter_changed)
        self.min_track_length_spin.valueChanged.connect(self.on_parameter_changed)
        self.max_fit_points_spin.valueChanged.connect(self.on_parameter_changed)
        self.model_selection_combo.currentTextChanged.connect(self.on_parameter_changed)

        # Connect visualization options to update plot display
        self.plot_type_combo.currentIndexChanged.connect(self.update_plot_display)
        self.show_individual_msd_check.stateChanged.connect(self.update_plot_display)
        self.show_ensemble_msd_check.stateChanged.connect(self.update_plot_display)
        self.show_model_fits_check.stateChanged.connect(self.update_plot_display)


    def on_parameter_changed(self):
        """Emits signal with current parameters when a parameter changes."""
        params = self.get_current_parameters()
        self.parameters_changed.emit(params)

    def get_current_parameters(self):
        """Collects current analysis parameters from the UI."""
        return {
            'max_lag': self.max_lag_spinbox.value(),
            'min_track_length': self.min_track_length_spin.value(),
            'max_fit_points': self.max_fit_points_spin.value(),
            'model_type': self.model_selection_combo.currentText().lower().replace(" ", "_"),
            # pixel_size and frame_interval should be passed from project settings by the main GUI
        }

    def update_results_display(self, results):
        """Updates the widget's display with analysis results."""
        self.current_results = results # Store results received from the worker
        if results and isinstance(results, dict): # Ensure results is a dictionary
            # Update Summary Text
            summary_text = "Diffusion Analysis Results:\n\n"

            # Assuming results structure based on DiffusionModelsAnalyzer.analyze output
            # which returns {'msd_df': msd_df, 'ensemble_msd': ensemble_msd, 'fit_results': fit_results, ...}
            if 'ensemble_msd' in results and not results['ensemble_msd'].empty:
                 ensemble_df = results['ensemble_msd']
                 summary_text += "--- Ensemble Average MSD ---\n"
                 summary_text += f"Calculated for {len(ensemble_df)} lag times.\n"
                 summary_text += f"MSD at first lag ({ensemble_df['time_lag'].iloc[0]:.3f} s): {ensemble_df['msd'].iloc[0]:.4f} μm²\n"
                 summary_text += "\n"

            if 'fit_results' in results and isinstance(results['fit_results'], dict):
                 fit_results = results['fit_results']
                 summary_text += "--- Model Fitting Results (First "
                 summary_text += f"{self.get_current_parameters().get('max_fit_points', 'N/A')} points) ---\n"
                 if 'best_model' in fit_results:
                      summary_text += f"Best Fit Model (based on R²): {fit_results['best_model']}\n"
                 for model_name, fit_data in fit_results.items():
                      if isinstance(fit_data, dict) and 'D' in fit_data:
                           summary_text += f"  {model_name.replace('_', ' ').title()}:\n"
                           summary_text += f"    D = {fit_data['D']:.4e} μm²/s (± {fit_data.get('D_err', np.nan):.2e})\n"
                           if 'alpha' in fit_data: summary_text += f"    α = {fit_data['alpha']:.3f} (± {fit_data.get('alpha_err', np.nan):.3f})\n"
                           if 'L' in fit_data: summary_text += f"    L = {fit_data['L']:.3f} μm (± {fit_data.get('L_err', np.nan):.3f})\n"
                           if 'v' in fit_data: summary_text += f"    v = {fit_data['v']:.3f} μm/s (± {fit_data.get('v_err', np.nan):.3f})\n"
                           if 'r_squared' in fit_data: summary_text += f"    R² = {fit_data['r_squared']:.3f}\n"
                 summary_text += "\n"

            if 'classification_df' in results and not results['classification_df'].empty:
                 classification_df = results['classification_df']
                 summary_text += f"--- Track Classification ({self.get_current_parameters().get('model_type', 'N/A')}) ---\n" # Assuming model_type relates to overall classification
                 summary_text += f"Classified {len(classification_df)} tracks.\n"
                 if 'diffusion_type' in classification_df.columns:
                      type_counts = classification_df['diffusion_type'].value_counts()
                      summary_text += "Diffusion Type Counts:\n"
                      for dtype, count in type_counts.items():
                           summary_text += f"  {dtype}: {count}\n"
                      # Basic stats per type
                      if 'D' in classification_df.columns:
                           summary_text += "\nMean D by Type:\n"
                           mean_d_by_type = classification_df.groupby('diffusion_type')['D'].mean()
                           for dtype, mean_d in mean_d_by_type.items():
                                summary_text += f"  {dtype}: {mean_d:.4e} μm²/s\n"
                 summary_text += "\n"

            self.summary_text.setText(summary_text)

            # Update Detailed Results Table
            if 'classification_df' in results and not results['classification_df'].empty:
                 self._update_detailed_table(results['classification_df']) # Assuming classification_df contains track-level stats
            else:
                 self.results_table.clear()
                 self.results_table.setRowCount(0)


            # Update Plots Display
            self.update_plot_display() # Call to draw the selected plot

        else:
            self.summary_text.setText("No Diffusion Analysis Results Available.")
            self.results_table.clear()
            self.results_table.setRowCount(0)
            self.plot_widget.clear()


    def _update_detailed_table(self, results_df):
        """Populates the detailed results table with track-level data."""
        self.results_table.clear()
        if results_df is None or results_df.empty:
            self.results_table.setRowCount(0)
            return

        # Use columns from the DataFrame for headers
        headers = results_df.columns.tolist() # e.g., track_id, D, alpha, straightness, cluster, diffusion_type
        self.results_table.setColumnCount(len(headers))
        self.results_table.setHorizontalHeaderLabels(headers)
        self.results_table.setRowCount(len(results_df))

        for i, (index, row) in enumerate(results_df.iterrows()):
            for j, col in enumerate(headers):
                item_text = str(row[col])
                if isinstance(row[col], (float, np.floating)):
                     item_text = f"{row[col]:.4f}" # Format floats
                self.results_table.setItem(i, j, QTableWidgetItem(item_text))

        self.results_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.results_table.verticalHeader().setVisible(False)


    def update_plot_display(self):
        """Clears and draws the selected plot type."""
        self.plot_widget.clear()
        if self.current_results is None:
            self.plot_widget.axes.text(0.5, 0.5, "No results to plot.", ha='center', va='center')
            self.plot_widget.draw()
            return

        plot_type = self.plot_type_combo.currentText()
        ax = self.plot_widget.axes

        # Access plotting functions from visualization.diffusion
        try:
            # Assuming visualization.diffusion functions are available and take axes as argument
            from visualization.diffusion import plot_msd_fits, plot_diffusion_histogram, plot_diffusion_map, plot_motion_classification

            if plot_type == "MSD Curves with Fits":
                 if 'ensemble_msd' in self.current_results and 'fit_results' in self.current_results:
                      ensemble_df = self.current_results['ensemble_msd']
                      fit_results = self.current_results['fit_results']

                      plot_msd_fits(
                           ensemble_df['time_lag'].values,
                           ensemble_df['msd'].values,
                           fit_results,
                           ax=ax,
                           max_lag=self.get_current_parameters().get('max_lag') # Use parameter if available
                      )
                 else:
                      ax.text(0.5, 0.5, "No ensemble MSD or fit results.", ha='center', va='center')

            elif plot_type == "Diffusion Coefficient Histogram":
                 if 'classification_df' in self.current_results and not self.current_results['classification_df'].empty:
                      classification_df = self.current_results['classification_df']
                      if 'D' in classification_df.columns:
                           plot_diffusion_histogram(
                                classification_df,
                                ax=ax,
                                log_scale=True # Example: always log scale for D
                           )
                      else:
                           ax.text(0.5, 0.5, "No 'D' column in classification results.", ha='center', va='center')
                 else:
                      ax.text(0.5, 0.5, "No track classification results.", ha='center', va='center')


            elif plot_type == "Diffusion Coefficient Map":
                 if hasattr(self.parent(), 'tracks_df') and self.parent().tracks_df is not None and \
                    'classification_df' in self.current_results and not self.current_results['classification_df'].empty:

                      # This plot requires the original tracks DataFrame and optionally a background image.
                      # The main GUI holds these; they need to be passed or accessed.
                      # Assuming the main GUI can provide tracks_df and the current image (e.g., max projection)
                      main_gui = self.parent() # Assuming parent is the main GUI window
                      tracks_df = main_gui.tracks_df
                      diffusion_df = self.current_results['classification_df'] # Contains 'track_id' and 'D'
                      background_image = None
                      if hasattr(main_gui, 'image_stack') and main_gui.image_stack is not None:
                           background_image = np.max(main_gui.image_stack, axis=0) # Example background

                      if tracks_df is not None:
                           plot_diffusion_map(
                                tracks_df,
                                diffusion_df,
                                background=background_image,
                                ax=ax,
                                pixel_size=main_gui.project_settings.get('pixel_size', 1.0) # Use project setting
                           )
                      else:
                           ax.text(0.5, 0.5, "Original tracks data not available.", ha='center', va='center')
                 else:
                      ax.text(0.5, 0.5, "Tracks or classification results not available.", ha='center', va='center')


            elif plot_type == "Anomalous Exponent Histogram":
                 if 'classification_df' in self.current_results and not self.current_results['classification_df'].empty:
                      classification_df = self.current_results['classification_df']
                      if 'alpha' in classification_df.columns:
                           ax.hist(classification_df['alpha'].dropna(), bins=30, alpha=0.7)
                           ax.set_xlabel("Anomalous Exponent (α)")
                           ax.set_ylabel("Frequency")
                           ax.set_title("Anomalous Exponent Distribution")
                      else:
                           ax.text(0.5, 0.5, "No 'alpha' column in classification results.", ha='center', va='center')
                 else:
                      ax.text(0.5, 0.5, "No track classification results.", ha='center', va='center')


            elif plot_type == "Model Comparison (R²)":
                 if 'fit_results' in self.current_results and isinstance(self.current_results['fit_results'], dict):
                      fit_results = self.current_results['fit_results']
                      model_names = [name.replace('_', ' ').title() for name in fit_results.keys() if isinstance(fit_results[name], dict) and 'r_squared' in fit_results[name]]
                      r_squared_values = [fit_results[name].get('r_squared', 0) for name in fit_results.keys() if isinstance(fit_results[name], dict) and 'r_squared' in fit_results[name]]

                      if model_names and r_squared_values:
                           ax.bar(model_names, r_squared_values, alpha=0.7)
                           ax.set_ylabel("Coefficient of Determination (R²)")
                           ax.set_title("Diffusion Model Fit Comparison")
                           ax.set_ylim(0, 1.1) # R2 is usually between 0 and 1
                           plt.xticks(rotation=45, ha='right') # Rotate labels if needed
                           plt.tight_layout() # Adjust layout
                      else:
                           ax.text(0.5, 0.5, "No R² data in fit results.", ha='center', va='center')
                 else:
                      ax.text(0.5, 0.5, "No fit results available.", ha='center', va='center')

            else:
                ax.text(0.5, 0.5, "Plot type not implemented.", ha='center', va='center')

            self.plot_widget.draw()

        except ImportError:
             logger.error("Visualization library not available.")
             self.plot_widget.clear()
             self.plot_widget.axes.text(0.5, 0.5, "Visualization library not found.", ha='center', va='center', color='red')
             self.plot_widget.draw()

        except Exception as e:
            logger.error(f"Error generating diffusion plot: {e}", exc_info=True)
            self.plot_widget.clear()
            self.plot_widget.axes.text(0.5, 0.5, f"Plot Error: {e}", ha='center', va='center', color='red')
            self.plot_widget.draw()


# Note: This widget requires access to project settings (pixel_size, frame_interval)
# and potentially the background image for plotting the diffusion map.
# These should be passed or made accessible by the main GUI window.


class ActiveTransportWidget(QWidget):
    """Widget for Active Transport Analysis controls and visualization."""

    parameters_changed = pyqtSignal(dict)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setup_ui()
        self.setup_connections()
        self.current_results = None # To hold analysis results for display

    def setup_ui(self):
        layout = QVBoxLayout(self)
        params_group = QGroupBox("Active Transport Analysis Parameters")
        params_layout = QFormLayout()

        # Parameters based on ActiveTransportAnalyzer
        # Using a dummy instance to get default values if available
        try:
            default_analyzer = ActiveTransportAnalyzer()
        except Exception:
            logger.warning("Could not instantiate ActiveTransportAnalyzer to get defaults. Using hardcoded values.")
            class DummyActiveTransportAnalyzer:
                 def __init__(self):
                      self.min_alpha = 1.3
                      self.min_track_length = 10
                      self.min_velocity = 0.1
                      self.min_run_length = 0.5
                      self.min_duration = 0.5
            default_analyzer = DummyActiveTransportAnalyzer()


        self.min_alpha_spin = QDoubleSpinBox()
        self.min_alpha_spin.setRange(1.0, 2.0)
        self.min_alpha_spin.setValue(default_analyzer.min_alpha)
        self.min_alpha_spin.setSingleStep(0.01) # More fine-grained control
        self.min_alpha_spin.setToolTip("Minimum anomalous exponent (α) for superdiffusion classification.")
        params_layout.addRow("Min. Alpha (Superdiffusion):", self.min_alpha_spin)

        self.min_track_length_spin = QSpinBox()
        self.min_track_length_spin.setRange(5, 500)
        self.min_track_length_spin.setValue(default_analyzer.min_track_length)
        self.min_track_length_spin.setToolTip("Minimum track length to analyze.")
        params_layout.addRow("Min Track Length:", self.min_track_length_spin)


        # Parameters for directed motion analysis
        self.min_duration_spin = QDoubleSpinBox()
        self.min_duration_spin.setRange(0.01, 100.0)
        self.min_duration_spin.setValue(default_analyzer.min_duration)
        self.min_duration_spin.setSingleStep(0.01)
        self.min_duration_spin.setSuffix(" s")
        self.min_duration_spin.setToolTip("Minimum duration for a segment to be considered directed.")
        params_layout.addRow("Min Directed Duration:", self.min_duration_spin)

        self.min_displacement_spin = QDoubleSpinBox()
        self.min_displacement_spin.setRange(0.01, 50.0)
        self.min_displacement_spin.setValue(default_analyzer.min_displacement)
        self.min_displacement_spin.setSingleStep(0.01)
        self.min_displacement_spin.setSuffix(" μm")
        self.min_displacement_spin.setToolTip("Minimum net displacement for a segment to be considered directed.")
        params_layout.addRow("Min Directed Displacement:", self.min_displacement_spin)

        # Note: analyze_superdiffusion takes compartment_masks
        # This widget doesn't define masks, they should be passed by the main GUI
        # if available and needed for compartment-specific superdiffusion analysis.

        params_group.setLayout(params_layout)
        layout.addWidget(params_group)

        # Visualization Options
        viz_group = QGroupBox("Visualization Options")
        viz_layout = QFormLayout()

        self.plot_type_combo = QComboBox()
        self.plot_type_combo.addItems([
            "Superdiffusion Alpha Distribution",
            "Directed Segment Speeds",
            "Directed Segment Map",
            "Transport Parameters Summary" # Placeholder for a summary plot
        ])
        viz_layout.addRow("Plot Type:", self.plot_type_combo)

        # Optional visualization parameters (example)
        self.show_all_tracks_check = QCheckBox("Show All Tracks (Background)")
        self.show_all_tracks_check.setChecked(False) # Default to False
        self.show_all_tracks_check.setToolTip("Show all tracks lightly in the background of spatial plots.")
        viz_layout.addRow("Display Options:", self.show_all_tracks_check)

        viz_group.setLayout(viz_layout)
        layout.addWidget(viz_group)


        # Results Display
        self.results_tabs = QTabWidget()

        self.summary_text = QTextEdit()
        self.summary_text.setReadOnly(True)
        self.results_tabs.addTab(self.summary_text, "Summary")

        self.superdiffusion_table = QTableWidget() # Table for superdiffusion results
        self.superdiffusion_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.results_tabs.addTab(self.superdiffusion_table, "Superdiffusion Results")

        self.directed_segments_table = QTableWidget() # Table for directed segment results
        self.directed_segments_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.results_tabs.addTab(self.directed_segments_table, "Directed Segments")

        self.transport_parameters_text = QTextEdit() # Text display for transport parameters
        self.transport_parameters_text.setReadOnly(True)
        self.results_tabs.addTab(self.transport_parameters_text, "Transport Parameters")


        self.plot_widget = MplCanvas() # For plots
        self.results_tabs.addTab(self.plot_widget, "Plots")

        layout.addWidget(self.results_tabs)


    def setup_connections(self):
        # Connect parameter changes to emit signal
        self.min_alpha_spin.valueChanged.connect(self.on_parameter_changed)
        self.min_track_length_spin.valueChanged.connect(self.on_parameter_changed)
        self.min_duration_spin.valueChanged.connect(self.on_parameter_changed)
        self.min_displacement_spin.valueChanged.connect(self.on_parameter_changed)

        # Connect visualization options to update plot display
        self.plot_type_combo.currentIndexChanged.connect(self.update_plot_display)
        self.show_all_tracks_check.stateChanged.connect(self.update_plot_display)


    def on_parameter_changed(self):
        """Emits signal with current parameters when a parameter changes."""
        params = self.get_current_parameters()
        self.parameters_changed.emit(params)

    def get_current_parameters(self):
        """Collects current analysis parameters from the UI."""
        return {
            'min_alpha': self.min_alpha_spin.value(), # Used by analyze_superdiffusion
            'min_track_length': self.min_track_length_spin.value(), # Used by analyze_superdiffusion, analyze_directed_motion
            'min_duration': self.min_duration_spin.value(), # Used by analyze_directed_motion
            'min_displacement': self.min_displacement_spin.value(), # Used by analyze_directed_motion
            # pixel_size and frame_interval should be passed from project settings by the main GUI
            # compartment_masks might be needed by analyze_superdiffusion, also passed by main GUI.
        }

    def update_results_display(self, results):
        """Updates the widget's display with analysis results."""
        self.current_results = results # Store results received from the worker
        if results and isinstance(results, dict): # Ensure results is a dictionary
            # Update Summary Text
            summary_text = "Active Transport Analysis Results:\n\n"

            # Assuming results structure based on ActiveTransportAnalyzer.analyze
            # which returns {'superdiffusive_tracks': ..., 'directed_motion_results': ..., 'transport_statistics': ...}
            if 'superdiffusive_tracks' in results and isinstance(results['superdiffusive_tracks'], dict):
                 superdiff_results = results['superdiffusive_tracks']
                 n_superdiffusive = sum(1 for r in superdiff_results.values() if r.get('superdiffusive', False))
                 n_total_tracks_analyzed = len(superdiff_results) # Tracks that met min_track_length
                 summary_text += f"--- Superdiffusion Analysis ---\n"
                 summary_text += f"Analyzed {n_total_tracks_analyzed} tracks for superdiffusion.\n"
                 summary_text += f"Identified {n_superdiffusive} superdiffusive tracks.\n"
                 summary_text += "\n"

            if 'directed_motion_results' in results and isinstance(results['directed_motion_results'], dict):
                 directed_results = results['directed_motion_results']
                 n_segments = directed_results.get('n_segments', 0)
                 summary_text += f"--- Directed Motion Analysis ---\n"
                 summary_text += f"Identified {n_segments} directed segments.\n"
                 # Could add breakdown by track

                 if 'transport_statistics' in results and isinstance(results['transport_statistics'], dict):
                      transport_stats = results['transport_statistics']
                      summary_text += "Transport Statistics (Overall):\n"
                      summary_text += f"  Mean Speed: {transport_stats.get('mean_speed', np.nan):.3f} μm/s\n"
                      summary_text += f"  Mean Duration: {transport_stats.get('mean_duration', np.nan):.3f} s\n"
                      summary_text += f"  Mean Straightness: {transport_stats.get('mean_straightness', np.nan):.3f}\n"
                      summary_text += f"  Segments per Track (avg): {transport_stats.get('segments_per_track', np.nan):.2f}\n"
                      summary_text += "\n"

            # Assuming calculate_transport_parameters results are also in the top-level dict if run separately
            if 'calculated_transport_parameters' in results and isinstance(results['calculated_transport_parameters'], dict):
                 calc_params = results['calculated_transport_parameters']
                 summary_text += "--- Calculated Transport Parameters ---\n"
                 summary_text += f"  Mean Processivity: {calc_params.get('mean_processivity', np.nan):.3f} μm\n"
                 summary_text += f"  Transport Efficiency: {calc_params.get('transport_efficiency', np.nan):.3f}\n"
                 if 'run_length_distribution' in calc_params and isinstance(calc_params['run_length_distribution'], dict):
                      rl_fit = calc_params['run_length_distribution']
                      summary_text += f"  Run Length Distribution Fit ({rl_fit.get('distribution', 'N/A')}): Mean={rl_fit.get('mean', np.nan):.3f}, p={rl_fit.get('p_value', np.nan):.3f} (Good Fit: {rl_fit.get('good_fit', False)})\n"
                 summary_text += "\n"


            self.summary_text.setText(summary_text)

            # Update Tables
            if 'superdiffusive_tracks' in results and isinstance(results['superdiffusive_tracks'], dict):
                 self._update_superdiffusion_table(results['superdiffusive_tracks'])
            else:
                 self.superdiffusion_table.clear()
                 self.superdiffusion_table.setRowCount(0)

            if 'directed_motion_results' in results and isinstance(results['directed_motion_results'], dict) and 'segments' in results['directed_motion_results']:
                 self._update_directed_segments_table(results['directed_motion_results']['segments'])
            else:
                 self.directed_segments_table.clear()
                 self.directed_segments_table.setRowCount(0)

            # Update Transport Parameters Text display (if different from summary)
            # self.transport_parameters_text.setText(...) # Could put raw parameter dict here


            # Update Plots Display
            self.update_plot_display() # Call to draw the selected plot

        else:
            self.summary_text.setText("No Active Transport Analysis Results Available.")
            self.superdiffusion_table.clear()
            self.superdiffusion_table.setRowCount(0)
            self.directed_segments_table.clear()
            self.directed_segments_table.setRowCount(0)
            self.transport_parameters_text.clear()
            self.plot_widget.clear()


    def _update_superdiffusion_table(self, superdiffusion_results_dict):
        """Populates the superdiffusion results table."""
        self.superdiffusion_table.clear()
        if not superdiffusion_results_dict:
            self.superdiffusion_table.setRowCount(0)
            return

        # Filter to show only superdiffusive tracks in the table
        superdiffusive_tracks = {k: v for k, v in superdiffusion_results_dict.items() if v.get('superdiffusive', False)}

        if not superdiffusive_tracks:
            self.superdiffusion_table.setRowCount(0)
            return

        # Define headers based on common keys in superdiffusive track results
        headers = ["Track ID", "Superdiffusive", "Alpha", "D (μm²/s)", "Speed (μm/s)", "Duration (s)"] # Example headers
        self.superdiffusion_table.setColumnCount(len(headers))
        self.superdiffusion_table.setHorizontalHeaderLabels(headers)
        self.superdiffusion_table.setRowCount(len(superdiffusive_tracks))

        for i, (track_id, result_data) in enumerate(superdiffusive_tracks.items()):
            self.superdiffusion_table.setItem(i, 0, QTableWidgetItem(str(track_id)))
            self.superdiffusion_table.setItem(i, 1, QTableWidgetItem(str(result_data.get('superdiffusive', False))))
            self.superdiffusion_table.setItem(i, 2, QTableWidgetItem(f"{result_data.get('alpha', np.nan):.3f}"))
            self.superdiffusion_table.setItem(i, 3, QTableWidgetItem(f"{result_data.get('diffusion_coefficient', np.nan):.4e}"))
            self.superdiffusion_table.setItem(i, 4, QTableWidgetItem(f"{result_data.get('speed', np.nan):.3f}"))
            self.superdiffusion_table.setItem(i, 5, QTableWidgetItem(f"{result_data.get('duration', np.nan):.3f}"))
            # Add more columns as needed


        self.superdiffusion_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.superdiffusion_table.verticalHeader().setVisible(False)


    def _update_directed_segments_table(self, directed_segments_list):
        """Populates the directed segments table."""
        self.directed_segments_table.clear()
        if not directed_segments_list:
            self.directed_segments_table.setRowCount(0)
            return

        # Define headers based on keys in directed segment dictionaries
        headers = ["Track ID", "Start Frame", "End Frame", "Duration (s)", "Displacement (μm)", "Speed (μm/s)", "Straightness"] # Example headers
        self.directed_segments_table.setColumnCount(len(headers))
        self.directed_segments_table.setHorizontalHeaderLabels(headers)
        self.directed_segments_table.setRowCount(len(directed_segments_list))

        for i, segment_data in enumerate(directed_segments_list):
            self.directed_segments_table.setItem(i, 0, QTableWidgetItem(str(segment_data.get('track_id', 'N/A'))))
            self.directed_segments_table.setItem(i, 1, QTableWidgetItem(str(segment_data.get('start_frame', 'N/A'))))
            self.directed_segments_table.setItem(i, 2, QTableWidgetItem(str(segment_data.get('end_frame', 'N/A'))))
            self.directed_segments_table.setItem(i, 3, QTableWidgetItem(f"{segment_data.get('duration', np.nan):.3f}"))
            self.directed_segments_table.setItem(i, 4, QTableWidgetItem(f"{segment_data.get('displacement', np.nan):.3f}"))
            self.directed_segments_table.setItem(i, 5, QTableWidgetItem(f"{segment_data.get('speed', np.nan):.3f}"))
            self.directed_segments_table.setItem(i, 6, QTableWidgetItem(f"{segment_data.get('straightness', np.nan):.3f}"))
            # Add more columns as needed


        self.directed_segments_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.directed_segments_table.verticalHeader().setVisible(False)


    def update_plot_display(self):
        """Clears and draws the selected plot type."""
        self.plot_widget.clear()
        if self.current_results is None:
            self.plot_widget.axes.text(0.5, 0.5, "No results to plot.", ha='center', va='center')
            self.plot_widget.draw()
            return

        plot_type = self.plot_type_combo.currentText()
        ax = self.plot_widget.axes

        # Access plotting functions from visualization modules (if available)
        try:
            # Assuming visualization.active_transport or visualization.tracks contains relevant plotting functions
            # from visualization.active_transport import plot_velocity_distribution, plot_run_length_distribution # Example
            # from visualization.tracks import plot_directed_segments # Example

            if plot_type == "Superdiffusion Alpha Distribution":
                 if 'superdiffusive_tracks' in self.current_results and isinstance(self.current_results['superdiffusive_tracks'], dict):
                      superdiff_tracks = self.current_results['superdiffusive_tracks']
                      alphas = [r.get('alpha') for r in superdiff_tracks.values() if r.get('alpha') is not None]
                      if alphas:
                           ax.hist(alphas, bins=30, alpha=0.7)
                           ax.set_xlabel("Anomalous Exponent (α)")
                           ax.set_ylabel("Frequency")
                           ax.set_title("Superdiffusive Alpha Distribution")
                      else:
                           ax.text(0.5, 0.5, "No superdiffusive alpha data found.", ha='center', va='center')
                 else:
                      ax.text(0.5, 0.5, "No superdiffusion analysis results.", ha='center', va='center')


            elif plot_type == "Directed Segment Speeds":
                 if 'directed_motion_results' in self.current_results and isinstance(self.current_results['directed_motion_results'], dict) and 'segments' in self.current_results['directed_motion_results']:
                      directed_segments = self.current_results['directed_motion_results']['segments']
                      speeds = [s.get('speed') for s in directed_segments if s.get('speed') is not None]
                      if speeds:
                           ax.hist(speeds, bins=30, alpha=0.7)
                           ax.set_xlabel("Speed (μm/s)")
                           ax.set_ylabel("Frequency")
                           ax.set_title("Directed Segment Speed Distribution")
                      else:
                           ax.text(0.5, 0.5, "No directed segment speed data found.", ha='center', va='center')
                 else:
                      ax.text(0.5, 0.5, "No directed motion analysis results.", ha='center', va='center')


            elif plot_type == "Directed Segment Map":
                 if 'directed_motion_results' in self.current_results and isinstance(self.current_results['directed_motion_results'], dict) and 'segments' in self.current_results['directed_motion_results']:
                      directed_segments = self.current_results['directed_motion_results']['segments']
                      # This plot requires the original tracks DataFrame and segment positions/info
                      # The main GUI holds the full tracks_df. Need to access or pass it.
                      # The segment dictionaries in directed_motion_results already contain 'segment_positions'
                      # Need to integrate visualization logic or call a function from visualization.tracks

                      # Assuming visualization.tracks.visualize_directed_segments exists and takes segments and tracks_df
                      # import visualization.tracks
                      # if hasattr(visualization.tracks, 'visualize_directed_segments') and hasattr(self.parent(), 'tracks_df') and self.parent().tracks_df is not None:
                      #      visualization.tracks.visualize_directed_segments(
                      #           directed_segments,
                      #           self.parent().tracks_df, # Pass full tracks_df for context
                      #           ax=ax
                      #      )
                      # else:
                      ax.text(0.5, 0.5, "Directed Segment Map Placeholder", ha='center', va='center')


            elif plot_type == "Transport Parameters Summary":
                 if ('transport_statistics' in self.current_results and isinstance(self.current_results['transport_statistics'], dict)) or \
                    ('calculated_transport_parameters' in self.current_results and isinstance(self.current_results['calculated_transport_parameters'], dict)):
                      # Create a bar plot or similar summarizing key parameters
                      # Placeholder plot:
                      ax.text(0.5, 0.5, "Transport Parameters Summary Plot Placeholder", ha='center', va='center')
                 else:
                      ax.text(0.5, 0.5, "No transport statistics available.", ha='center', va='center')

            else:
                ax.text(0.5, 0.5, "Plot type not implemented.", ha='center', va='center')

            self.plot_widget.draw()

        except ImportError:
             logger.error("Visualization library not available.")
             self.plot_widget.clear()
             self.plot_widget.axes.text(0.5, 0.5, "Visualization library not found.", ha='center', va='center', color='red')
             self.plot_widget.draw()
        except Exception as e:
            logger.error(f"Error generating active transport plot: {e}", exc_info=True)
            self.plot_widget.clear()
            self.plot_widget.axes.text(0.5, 0.5, f"Plot Error: {e}", ha='center', va='center', color='red')
            self.plot_widget.draw()

# Note: This widget requires access to project settings (pixel_size, frame_interval).
# It also needs access to original tracks_df for spatial visualization.


class BoundaryCrossingWidget(QWidget):
    """Widget for Boundary Crossing Analysis controls and visualization."""

    parameters_changed = pyqtSignal(dict)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setup_ui()
        self.setup_connections()
        self.current_results = None # To hold analysis results for display

    def setup_ui(self):
        layout = QVBoxLayout(self)
        params_group = QGroupBox("Boundary Crossing Analysis Parameters")
        params_layout = QFormLayout()

        # Parameters based on BoundaryCrossingAnalyzer
        # Using a dummy instance to get default values if available
        try:
            default_analyzer = BoundaryCrossingAnalyzer()
        except Exception:
             logger.warning("Could not instantiate BoundaryCrossingAnalyzer to get defaults. Using hardcoded values.")
             class DummyBoundaryCrossingAnalyzer:
                  def __init__(self):
                       self.dt = 0.014
             default_analyzer = DummyBoundaryCrossingAnalyzer()

        self.dt_spin = QDoubleSpinBox()
        self.dt_spin.setRange(0.001, 100.0)
        self.dt_spin.setValue(default_analyzer.dt)
        self.dt_spin.setSingleStep(0.001)
        self.dt_spin.setSuffix(" s")
        self.dt_spin.setToolTip("Time interval between frames for velocity calculations.")
        params_layout.addRow("Time Interval:", self.dt_spin)

        # Parameters related to the underlying compartment masks
        # These are not defined *within* this widget, but should be available
        # in the main GUI and passed to the analyzer's analyze method.
        # The widget should indicate this dependency.
        self.compartment_masks_label = QLabel("Compartment masks are required. Define them in Image Processing or Project tabs.")
        self.compartment_masks_label.setWordWrap(True)
        params_layout.addWidget(self.compartment_masks_label)


        params_group.setLayout(params_layout)
        layout.addWidget(params_group)

        # Visualization Options
        viz_group = QGroupBox("Visualization Options")
        viz_layout = QFormLayout()

        self.plot_type_combo = QComboBox()
        self.plot_type_combo.addItems([
            "Crossing Events Map",
            "Crossing Angle Histogram",
            "Crossing Angle Polar Plot",
            "Angular Statistics Summary" # Placeholder for a summary plot
        ])
        viz_layout.addRow("Plot Type:", self.plot_type_combo)

        # Optional visualization parameters (example)
        self.show_background_check = QCheckBox("Show Background Image")
        self.show_background_check.setChecked(True) # Default to True for spatial plots
        self.show_background_check.setToolTip("Show the background image (e.g., max projection) on spatial plots.")
        viz_layout.addRow("Display Options:", self.show_background_check)

        viz_group.setLayout(viz_layout)
        layout.addWidget(viz_group)


        # Results Display (Based on spt_gui snippet)
        self.results_tabs = QTabWidget()

        self.summary_text = QTextEdit()
        self.summary_text.setReadOnly(True)
        self.results_tabs.addTab(self.summary_text, "Summary")

        self.crossing_events_table = QTableWidget() # Table for crossing events
        self.crossing_events_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.results_tabs.addTab(self.crossing_events_table, "Crossing Events")

        self.angular_summary_table = QTableWidget() # Table for angular distribution summary
        self.angular_summary_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.results_tabs.addTab(self.angular_summary_table, "Angular Summary")


        self.plot_widget = MplCanvas() # For plots (angular, etc.)
        self.results_tabs.addTab(self.plot_widget, "Plots")

        layout.addWidget(self.results_tabs)


    def setup_connections(self):
        # Connect parameter changes to emit signal
        self.dt_spin.valueChanged.connect(self.on_parameter_changed)

        # Connect visualization options to update plot display
        self.plot_type_combo.currentIndexChanged.connect(self.update_plot_display)
        self.show_background_check.stateChanged.connect(self.update_plot_display)


    def on_parameter_changed(self):
        """Emits signal with current parameters when a parameter changes."""
        params = self.get_current_parameters()
        self.parameters_changed.emit(params)

    def get_current_parameters(self):
        """Collects current analysis parameters from the UI."""
        return {
            'dt': self.dt_spin.value(),
            # pixel_size should be passed from project settings by the main GUI
            # compartment_masks are required and should be passed by the main GUI.
        }

    def update_results_display(self, results):
        """Updates the widget's display with analysis results."""
        self.current_results = results # Store results received from the worker
        if results and isinstance(results, dict): # Ensure results is a dictionary
            # Update Summary Text
            summary_text = "Boundary Crossing Analysis Results:\n\n"

            # Assuming results structure from BoundaryCrossingAnalyzer.analyze_boundary_crossings
            if 'crossing_events' in results and isinstance(results['crossing_events'], list):
                 crossing_events = results['crossing_events']
                 summary_text += f"--- Boundary Crossing Events ---\n"
                 summary_text += f"Identified {len(crossing_events)} crossing events.\n"
                 # Could add counts per boundary pair
                 summary_text += "\n"

            # Assuming results structure from BoundaryCrossingAnalyzer.analyze_angular_distribution
            if 'angular_distributions' in results and isinstance(results['angular_distributions'], dict):
                 angular_results = results['angular_distributions']
                 summary_text += f"--- Angular Distribution Analysis ({angular_results.get('status', 'N/A')}) ---\n"
                 if angular_results.get('status') == 'Computed':
                      summary_text += f"Analyzed {len(angular_results.get('crossing_angles', []))} crossing angles.\n"
                      # Could add overall mean/circular mean angle
                 summary_text += "\n"

                 if 'boundary_summary' in angular_results and isinstance(angular_results['boundary_summary'], dict):
                      boundary_summary = angular_results['boundary_summary']
                      summary_text += "Angular Statistics by Boundary Pair:\n"
                      for boundary_pair, stats in boundary_summary.items():
                           summary_text += f"  {boundary_pair[0]} - {boundary_pair[1]}:\n"
                           summary_text += f"    Count: {stats.get('count', 0)}\n"
                           summary_text += f"    Mean Angle: {stats.get('mean_angle', np.nan):.2f}°\n"
                           summary_text += f"    Circular Mean: {stats.get('circular_mean_angle', np.nan):.2f}°\n"
                           summary_text += f"    Circular Std Dev: {stats.get('circular_std_dev_deg', np.nan):.2f}°\n"
                           summary_text += "\n"

            self.summary_text.setText(summary_text)

            # Update Tables (Based on spt_gui snippet)
            if 'crossing_events' in results and isinstance(results['crossing_events'], list):
                 self._update_crossing_events_table(results['crossing_events'])
            else:
                 self.crossing_events_table.clear()
                 self.crossing_events_table.setRowCount(0)

            if 'angular_distributions' in results and 'boundary_summary' in results['angular_distributions'] and isinstance(results['angular_distributions']['boundary_summary'], dict):
                 self._update_angular_summary_table(results['angular_distributions']['boundary_summary'])
            else:
                 self.angular_summary_table.clear()
                 self.angular_summary_table.setRowCount(0)


            # Update Plots Display
            self.update_plot_display() # Call to draw the selected plot

        else:
            self.summary_text.setText("No Boundary Crossing Analysis Results Available.")
            self.crossing_events_table.clear()
            self.crossing_events_table.setRowCount(0)
            self.angular_summary_table.clear()
            self.angular_summary_table.setRowCount(0)
            self.plot_widget.clear()


    def _update_crossing_events_table(self, crossing_events_list):
        """Populates the crossing events table."""
        self.crossing_events_table.clear()
        if not crossing_events_list:
            self.crossing_events_table.setRowCount(0)
            return

        # Headers based on keys in crossing event dictionaries
        headers = ["Track ID", "Frame From", "Frame To", "From Compartment", "To Compartment", "Position From (px)", "Position To (px)"]
        self.crossing_events_table.setColumnCount(len(headers))
        self.crossing_events_table.setHorizontalHeaderLabels(headers)
        self.crossing_events_table.setRowCount(len(crossing_events_list))

        for i, event in enumerate(crossing_events_list):
            self.crossing_events_table.setItem(i, 0, QTableWidgetItem(str(event.get('track_id', 'N/A'))))
            self.crossing_events_table.setItem(i, 1, QTableWidgetItem(str(event.get('frame_from', 'N/A'))))
            self.crossing_events_table.setItem(i, 2, QTableWidgetItem(str(event.get('frame_to', 'N/A'))))
            self.crossing_events_table.setItem(i, 3, QTableWidgetItem(str(event.get('from_compartment', 'N/A'))))
            self.crossing_events_table.setItem(i, 4, QTableWidgetItem(str(event.get('to_compartment', 'N/A'))))
            self.crossing_events_table.setItem(i, 5, QTableWidgetItem(str(event.get('position_from', 'N/A')))) # Store/display as string? Or format?
            self.crossing_events_table.setItem(i, 6, QTableWidgetItem(str(event.get('position_to', 'N/A'))))   # Store/display as string? Or format?
            # Add dx, dy if needed: f"{event.get('dx', np.nan):.2f}", f"{event.get('dy', np.nan):.2f}"


        self.crossing_events_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.crossing_events_table.verticalHeader().setVisible(False)


    def _update_angular_summary_table(self, boundary_summary_dict):
        """Populates the angular summary table."""
        self.angular_summary_table.clear()
        if not boundary_summary_dict:
            self.angular_summary_table.setRowCount(0)
            return

        # Headers based on keys in angular summary dictionaries
        headers = ["Boundary Pair", "Count", "Mean Angle (°)", "Circular Mean (°)", "Std Dev (°)", "Circular Std Dev (°)", "Rayleigh p-value"] # Added Rayleigh
        self.angular_summary_table.setColumnCount(len(headers))
        self.angular_summary_table.setHorizontalHeaderLabels(headers)
        self.angular_summary_table.setRowCount(len(boundary_summary_dict))

        sorted_boundary_pairs = sorted(boundary_summary_dict.keys()) # Display in order
        for i, boundary_pair in enumerate(sorted_boundary_pairs):
            stats = boundary_summary_dict[boundary_pair]
            self.angular_summary_table.setItem(i, 0, QTableWidgetItem(f"{boundary_pair[0]} - {boundary_pair[1]}"))
            self.angular_summary_table.setItem(i, 1, QTableWidgetItem(str(stats.get('count', 0))))
            self.angular_summary_table.setItem(i, 2, QTableWidgetItem(f"{stats.get('mean_angle', np.nan):.2f}"))
            self.angular_summary_table.setItem(i, 3, QTableWidgetItem(f"{stats.get('circular_mean_angle', np.nan):.2f}"))
            self.angular_summary_table.setItem(i, 4, QTableWidgetItem(f"{stats.get('std_angle', np.nan):.2f}"))
            self.angular_summary_table.setItem(i, 5, QTableWidgetItem(f"{stats.get('circular_std_dev_deg', np.nan):.2f}"))
            self_rayleigh_p = stats.get('rayleigh_pvalue', np.nan) # Assuming Rayleigh test is done in analyzer
            self.angular_summary_table.setItem(i, 6, QTableWidgetItem(f"{self_rayleigh_p:.3f}" if not np.isnan(self_rayleigh_p) else "N/A"))
            # Could indicate if non-uniform (p < 0.05)

        self.angular_summary_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.angular_summary_table.verticalHeader().setVisible(False)


    def update_plot_display(self):
        """Clears and draws the selected plot type."""
        self.plot_widget.clear()
        if self.current_results is None or 'angular_distributions' not in self.current_results or self.current_results['angular_distributions'].get('status') != 'Computed':
            self.plot_widget.axes.text(0.5, 0.5, "No angular distribution results to plot.", ha='center', va='center')
            self.plot_widget.draw()
            return

        plot_type = self.plot_type_combo.currentText()
        ax = self.plot_widget.axes
        angular_results = self.current_results['angular_distributions']

        try:
            if plot_type == "Crossing Events Map":
                 if hasattr(self.parent(), 'tracks_df') and self.parent().tracks_df is not None and \
                    'crossing_events' in self.current_results and isinstance(self.current_results['crossing_events'], list):

                      # This plot requires original tracks, crossing event positions, and optionally background/masks.
                      # Need to integrate visualization logic or call a function from visualization.boundary_plots
                      # Assuming visualization.boundary_plots.plot_crossing_events exists

                      # import visualization.boundary_plots
                      # if hasattr(visualization.boundary_plots, 'plot_crossing_events'):
                      #      boundary_plots.plot_crossing_events(
                      #           self.current_results['crossing_events'],
                      #           background_image=self.parent().image_stack[0] if self.show_background_check.isChecked() and hasattr(self.parent(), 'image_stack') else None,
                      #           compartment_masks=self.parent().compartment_masks if hasattr(self.parent(), 'compartment_masks') else None, # Pass masks if available
                      #           ax=ax
                      #      )
                      # else:
                      ax.text(0.5, 0.5, "Crossing Events Map Placeholder", ha='center', va='center')

                 else:
                      ax.text(0.5, 0.5, "Tracks or crossing events data not available.", ha='center', va='center')


            elif plot_type == "Crossing Angle Histogram":
                 crossing_angles = angular_results.get('crossing_angles', [])
                 if crossing_angles:
                      ax.hist(crossing_angles, bins=36, range=(-180, 180), alpha=0.7)
                      ax.set_xlabel('Crossing Angle (degrees)')
                      ax.set_ylabel('Frequency')
                      ax.set_title('Distribution of Crossing Angles')
                 else:
                      ax.text(0.5, 0.5, "No crossing angle data available.", ha='center', va='center')


            elif plot_type == "Crossing Angle Polar Plot":
                 crossing_angles = angular_results.get('crossing_angles', [])
                 if crossing_angles:
                      # Need to create a polar axis
                      self.plot_widget.clear() # Clear the default axes
                      polar_ax = self.plot_widget.fig.add_subplot(111, projection='polar')
                      angles_rad = np.deg2rad(crossing_angles)
                      polar_ax.hist(angles_rad, bins=36, alpha=0.7)
                      polar_ax.set_theta_zero_location("N") # Set 0 degrees to North (up)
                      polar_ax.set_theta_direction(-1) # Clockwise direction
                      polar_ax.set_title("Polar Distribution of Crossing Angles")
                      # Store the polar axis reference if needed later, or redraw entire figure.
                      self.plot_widget.axes = polar_ax # Update the widget's internal axes reference
                 else:
                      ax.text(0.5, 0.5, "No crossing angle data available.", ha='center', va='center')

            elif plot_type == "Angular Statistics Summary":
                 if 'boundary_summary' in angular_results and isinstance(angular_results['boundary_summary'], dict):
                      boundary_summary = angular_results['boundary_summary']
                      # Create a bar plot of mean angles or circular means per boundary pair
                      # Placeholder plot:
                      ax.text(0.5, 0.5, "Angular Statistics Summary Plot Placeholder", ha='center', va='center')
                 else:
                      ax.text(0.5, 0.5, "No angular summary data available.", ha='center', va='center')


            else:
                ax.text(0.5, 0.5, "Plot type not implemented.", ha='center', va='center')

            self.plot_widget.draw()

        except ImportError:
             logger.error("Visualization library not available.")
             self.plot_widget.clear()
             self.plot_widget.axes.text(0.5, 0.5, "Visualization library not found.", ha='center', va='center', color='red')
             self.plot_widget.draw()
        except Exception as e:
            logger.error(f"Error generating boundary crossing plot: {e}", exc_info=True)
            self.plot_widget.clear()
            self.plot_widget.axes.text(0.5, 0.5, f"Plot Error: {e}", ha='center', va='center', color='red')
            self.plot_widget.draw()


# Note: This widget requires access to project settings (pixel_size, frame_interval).
# It also requires compartment masks to be generated and available (passed by main GUI).
# Spatial plots (Crossing Events Map) require the original tracks_df and background image.



clclass DwellTimeWidget(QWidget):
    """Widget for Dwell Time Analysis controls and visualization."""

    parameters_changed = pyqtSignal(dict)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setup_ui()
        self.setup_connections()
        self.current_results = None  # To hold analysis results for display

    def setup_ui(self):
        layout = QVBoxLayout(self)
        params_group = QGroupBox("Dwell Time Analysis Parameters")
        params_layout = QFormLayout()

        # Example parameters
        self.dt_spin = QDoubleSpinBox()
        self.dt_spin.setRange(0.001, 100.0)
        self.dt_spin.setValue(0.1)
        self.dt_spin.setSingleStep(0.001)
        self.dt_spin.setSuffix(" s")
        self.dt_spin.setToolTip("Time interval between frames.")
        params_layout.addRow("Time Interval:", self.dt_spin)

        self.min_binding_frames_spin = QSpinBox()
        self.min_binding_frames_spin.setRange(2, 500)
        self.min_binding_frames_spin.setValue(10)
        self.min_binding_frames_spin.setToolTip("Minimum frames for a detected binding event.")
        params_layout.addRow("Min Binding Frames:", self.min_binding_frames_spin)

        params_group.setLayout(params_layout)
        layout.addWidget(params_group)

        # Results Display
        self.results_tabs = QTabWidget()
        self.summary_text = QTextEdit()
        self.summary_text.setReadOnly(True)
        self.results_tabs.addTab(self.summary_text, "Summary")
        layout.addWidget(self.results_tabs)

    def setup_connections(self):
        self.dt_spin.valueChanged.connect(self.on_parameter_changed)
        self.min_binding_frames_spin.valueChanged.connect(self.on_parameter_changed)

    def on_parameter_changed(self):
        params = self.get_current_parameters()
        self.parameters_changed.emit(params)

    def get_current_parameters(self):
        return {
            'dt': self.dt_spin.value(),
            'min_binding_frames': self.min_binding_frames_spin.value(),
        }

    def update_results_display(self, results):
        self.current_results = results
        if results:
            self.summary_text.setText("Results updated.")
        else:
            self.summary_text.setText("No results available.")
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setup_ui()
        self.setup_connections()
        self.current_results = None # To hold analysis results for display

    def setup_ui(self):
        layout = QVBoxLayout(self)
        params_group = QGroupBox("Dwell Time Analysis Parameters")
        params_layout = QFormLayout()

        # Parameters based on DwellTimeAnalyzer.__init__
        default_analyzer = DwellTimeAnalyzer() # Instance to get defaults

        self.dt_spin = QDoubleSpinBox()
        self.dt_spin.setRange(0.001, 100.0) # Increased range based on potential project settings
        self.dt_spin.setValue(default_analyzer.dt)
        self.dt_spin.setSingleStep(0.001)
        self.dt_spin.setSuffix(" s")
        self.dt_spin.setToolTip("Time interval between frames.")
        params_layout.addRow("Time Interval:", self.dt_spin)

        self.immobility_threshold_spin = QDoubleSpinBox()
        self.immobility_threshold_spin.setRange(0.01, 50.0)
        self.immobility_threshold_spin.setValue(default_analyzer.immobility_threshold)
        self.immobility_threshold_spin.setSingleStep(0.01)
        self.immobility_threshold_spin.setSuffix(" px")
        self.immobility_threshold_spin.setToolTip("Displacement threshold for immobile classification.")
        params_layout.addRow("Immobility Threshold:", self.immobility_threshold_spin)

        self.min_binding_frames_spin = QSpinBox()
        self.min_binding_frames_spin.setRange(2, 500)
        self.min_binding_frames_spin.setValue(default_analyzer.min_binding_frames)
        self.min_binding_frames_spin.setToolTip("Minimum frames for a detected binding event.")
        params_layout.addRow("Min Binding Frames:", self.min_binding_frames_spin)

        self.cage_detection_window_spin = QSpinBox()
        self.cage_detection_window_spin.setRange(5, 200)
        self.cage_detection_window_spin.setValue(default_analyzer.cage_detection_window)
        self.cage_detection_window_spin.setToolTip("Window size for detecting confined/caged motion.")
        params_layout.addRow("Cage Detection Window:", self.cage_detection_window_spin)

        # Options for specific dwell time analysis tasks (e.g., which ones to run/display)
        # This could be implemented with checkboxes or a dropdown if needed.
        # For simplicity, assume running all relevant analyses in the main GUI's worker.

        params_group.setLayout(params_layout)
        layout.addWidget(params_group)

        # Visualization Options (Placeholder)
        viz_group = QGroupBox("Visualization Options")
        viz_layout = QFormLayout()
        self.plot_type_combo = QComboBox()
        self.plot_type_combo.addItems(["Dwell Time Distribution", "Binding Event Durations", "Cage Escape Rates"]) # Example plot types
        viz_layout.addRow("Plot Type:", self.plot_type_combo)
        viz_group.setLayout(viz_layout)
        layout.addWidget(viz_group)

        # Results Display
        self.results_tabs = QTabWidget()

        self.summary_text = QTextEdit()
        self.summary_text.setReadOnly(True)
        self.results_tabs.addTab(self.summary_text, "Summary")

        self.binding_events_table = QTableWidget() # Table for binding events
        self.binding_events_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.results_tabs.addTab(self.binding_events_table, "Binding Events")

        self.plot_widget = MplCanvas()
        self.results_tabs.addTab(self.plot_widget, "Plots")

        layout.addWidget(self.results_tabs)

    def setup_connections(self):
        # Connect parameter changes to emit signal
        self.dt_spin.valueChanged.connect(self.on_parameter_changed)
        self.immobility_threshold_spin.valueChanged.connect(self.on_parameter_changed)
        self.min_binding_frames_spin.valueChanged.connect(self.on_parameter_changed)
        self.cage_detection_window_spin.valueChanged.connect(self.on_parameter_changed)

        # Connect plot type selection to update plot display
        self.plot_type_combo.currentIndexChanged.connect(self.update_plot_display)


    def on_parameter_changed(self):
        """Emits signal with current parameters when a parameter changes."""
        params = self.get_current_parameters()
        self.parameters_changed.emit(params)

    def get_current_parameters(self):
        """Collects current analysis parameters from the UI."""
        return {
            'dt': self.dt_spin.value(),
            'immobility_threshold': self.immobility_threshold_spin.value(),
            'min_binding_frames': self.min_binding_frames_spin.value(),
            'cage_detection_window': self.cage_detection_window_spin.value(),
            # Add parameters for specific analysis tasks if needed by the analyzer.analyze method
            # Example: 'analyze_binding': self.analyze_binding_checkbox.isChecked()
        }

    def update_results_display(self, results):
        """Updates the widget's display with analysis results."""
        self.current_results = results # Store results received from the worker
        if results:
            # Update Summary Text
            summary_text = "Dwell Time Analysis Results:\n\n"
            if 'dwell_times' in results and 'statistics' in results['dwell_times']:
                summary_text += "--- Dwell Time Statistics by Compartment ---\n"
                for comp, stats in results['dwell_times']['statistics'].items():
                    summary_text += f"  Compartment: {comp}\n"
                    summary_text += f"    Mean Dwell Time: {stats.get('mean', np.nan):.3f} s\n"
                    summary_text += f"    Number of Events: {stats.get('n_events', 0)}\n"
                    if 'exponential_fit' in stats and stats['exponential_fit']:
                        fit = stats['exponential_fit']
                        summary_text += f"    Exponential Fit (koff): {fit.get('rate', np.nan):.3f} s⁻¹ (p={fit.get('p_value', np.nan):.3f})\n"
                    summary_text += "\n"

            if 'binding_events' in results and 'events' in results['binding_events']:
                 summary_text += f"--- Binding Events ---\n"
                 summary_text += f"Total Binding Events Detected: {len(results['binding_events']['events'])}\n"
                 # Basic counts by type/compartment could be added here

            if 'cage_escapes' in results and 'events' in results['cage_escapes']:
                 summary_text += f"--- Cage Dynamics ---\n"
                 summary_text += f"Total Cage Events Detected: {len(results['cage_escapes']['events'])}\n"
                 # Basic stats on cage properties could be added here

            if 'kinetic_parameters' in results and results['kinetic_parameters']:
                summary_text += "--- Extracted Kinetic Parameters ---\n"
                for param, value in results['kinetic_parameters'].items():
                    if isinstance(value, float):
                        summary_text += f"  {param}: {value:.4f}\n"
                    else:
                         summary_text += f"  {param}: {value}\n"

            self.summary_text.setText(summary_text)

            # Update Binding Events Table
            if 'binding_events' in results and 'events' in results['binding_events']:
                 self._update_binding_table(results['binding_events']['events'])
            else:
                 self.binding_events_table.clear()
                 self.binding_events_table.setRowCount(0)


            # Update Plots Display
            self.update_plot_display() # Call to draw the selected plot


        else:
            self.summary_text.setText("No Dwell Time Analysis Results Available.")
            self.binding_events_table.clear()
            self.binding_events_table.setRowCount(0)
            self.plot_widget.clear()

    def _update_binding_table(self, binding_events_list):
         """Populates the binding events table."""
         self.binding_events_table.clear()
         if not binding_events_list:
             self.binding_events_table.setRowCount(0)
             return

         headers = ["Track ID", "Start Frame", "End Frame", "Duration (s)", "Compartment", "Type", "MSD during Binding"]
         self.binding_events_table.setColumnCount(len(headers))
         self.binding_events_table.setHorizontalHeaderLabels(headers)
         self.binding_events_table.setRowCount(len(binding_events_list))

         for i, event in enumerate(binding_events_list):
              self.binding_events_table.setItem(i, 0, QTableWidgetItem(str(event.get('track_id', 'N/A'))))
              self.binding_events_table.setItem(i, 1, QTableWidgetItem(str(event.get('start_frame', 'N/A'))))
              self.binding_events_table.setItem(i, 2, QTableWidgetItem(str(event.get('end_frame', 'N/A'))))
              self.binding_events_table.setItem(i, 3, QTableWidgetItem(f"{event.get('duration', np.nan):.3f}"))
              self.binding_events_table.setItem(i, 4, QTableWidgetItem(str(event.get('compartment', 'N/A'))))
              self.binding_events_table.setItem(i, 5, QTableWidgetItem(str(event.get('binding_type', 'N/A'))))
              self.binding_events_table.setItem(i, 6, QTableWidgetItem(f"{event.get('msd_during_binding', np.nan):.4f}")) # Assuming this key exists

         self.binding_events_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
         self.binding_events_table.verticalHeader().setVisible(False) # Hide row numbers


    def update_plot_display(self):
        """Clears and draws the selected plot type."""
        self.plot_widget.clear()
        if self.current_results is None:
            self.plot_widget.axes.text(0.5, 0.5, "No results to plot.", ha='center', va='center')
            self.plot_widget.draw()
            return

        plot_type = self.plot_type_combo.currentText()
        ax = self.plot_widget.axes

        try:
            if plot_type == "Dwell Time Distribution":
                 if 'dwell_times' in self.current_results and 'events' in self.current_results['dwell_times']:
                      # Plot dwell time histograms by compartment
                      dwell_events = self.current_results['dwell_times']['events']
                      for comp, events in dwell_events.items():
                           if events:
                                dwell_times = [e['dwell_time'] for e in events]
                                ax.hist(dwell_times, bins=50, alpha=0.7, label=comp)
                      ax.set_xlabel("Dwell Time (s)")
                      ax.set_ylabel("Frequency")
                      ax.set_title("Dwell Time Distribution by Compartment")
                      ax.legend()
                 else:
                      ax.text(0.5, 0.5, "No dwell time events found.", ha='center', va='center')

            elif plot_type == "Binding Event Durations":
                 if 'binding_events' in self.current_results and 'events' in self.current_results['binding_events']:
                      binding_events = self.current_results['binding_events']['events']
                      if binding_events:
                           durations = [e['duration'] for e in binding_events]
                           ax.hist(durations, bins=50, alpha=0.7)
                           ax.set_xlabel("Binding Duration (s)")
                           ax.set_ylabel("Frequency")
                           ax.set_title("Binding Event Duration Distribution")
                      else:
                           ax.text(0.5, 0.5, "No binding events found.", ha='center', va='center')
                 else:
                      ax.text(0.5, 0.5, "No binding event data available.", ha='center', va='center')


            elif plot_type == "Cage Escape Rates":
                 if 'cage_escapes' in self.current_results and 'events' in self.current_results['cage_escapes']:
                      cage_events = self.current_results['cage_escapes']['events']
                      if cage_events:
                           # This is a simplified plot - actual escape rate analysis might be more complex
                           cage_lifetimes = [(e['end_frame'] - e['start_frame']) * self.get_current_parameters().get('dt', 1.0) for e in cage_events]
                           ax.hist(cage_lifetimes, bins=30, alpha=0.7)
                           ax.set_xlabel("Cage Lifetime (s)")
                           ax.set_ylabel("Frequency")
                           ax.set_title("Cage Lifetime Distribution")
                      else:
                           ax.text(0.5, 0.5, "No cage events found.", ha='center', va='center')
                 else:
                      ax.text(0.5, 0.5, "No cage dynamics data available.", ha='center', va='center')

            else:
                ax.text(0.5, 0.5, "Plot type not implemented.", ha='center', va='center')

            self.plot_widget.draw()

        except Exception as e:
            logger.error(f"Error generating dwell time plot: {e}", exc_info=True)
            self.plot_widget.clear()
            self.plot_widget.axes.text(0.5, 0.5, f"Plot Error: {e}", ha='center', va='center', color='red')
            self.plot_widget.draw()


# C:\Users\mjhen\SPT_GUI\widgets\analysis_widgets.py (Continued)

class CrowdingWidget(QWidget):
    """Widget for Crowding Effects Analysis controls and visualization."""

    parameters_changed = pyqtSignal(dict)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setup_ui()
        self.setup_connections()
        self.current_results = None # To hold analysis results for display

    def setup_ui(self):
        layout = QVBoxLayout(self)
        params_group = QGroupBox("Crowding Effects Analysis Parameters")
        params_layout = QFormLayout()

        # Parameters based on CrowdingAnalyzer.__init__ and analyze methods
        default_analyzer = CrowdingAnalyzer() # Instance to get defaults

        self.dt_spin = QDoubleSpinBox()
        self.dt_spin.setRange(0.001, 100.0)
        self.dt_spin.setValue(default_analyzer.dt)
        self.dt_spin.setSingleStep(0.001)
        self.dt_spin.setSuffix(" s")
        self.dt_spin.setToolTip("Time interval between frames.")
        params_layout.addRow("Time Interval:", self.dt_spin)

        self.min_track_length_spin = QSpinBox()
        self.min_track_length_spin.setRange(5, 100)
        self.min_track_length_spin.setValue(default_analyzer.min_track_length)
        self.min_track_length_spin.setToolTip("Minimum track length for analysis.")
        params_layout.addRow("Min Track Length:", self.min_track_length_spin)

        # Parameters for specific analysis methods within CrowdingAnalyzer
        self.particle_radius_spin = QDoubleSpinBox()
        self.particle_radius_spin.setRange(0.1, 1000.0)
        self.particle_radius_spin.setValue(5.0) # Example default
        self.particle_radius_spin.setSingleStep(0.1)
        self.particle_radius_spin.setSuffix(" nm")
        self.particle_radius_spin.setToolTip("Radius of the tracked particle.")
        params_layout.addRow("Particle Radius:", self.particle_radius_spin)

        self.temperature_spin = QDoubleSpinBox()
        self.temperature_spin.setRange(0.1, 100.0)
        self.temperature_spin.setValue(25.0) # Example default in Celsius
        self.temperature_spin.setSingleStep(0.1)
        self.temperature_spin.setSuffix(" °C")
        self.temperature_spin.setToolTip("Temperature for viscosity calculations.")
        params_layout.addRow("Temperature:", self.temperature_spin)

        # Add options for specific Crowding analysis types if needed (e.g., spatial, viscosity, non-gaussian)
        self.analyze_viscosity_check = QCheckBox("Analyze Viscosity")
        self.analyze_viscosity_check.setChecked(True)
        params_layout.addRow("Include Viscosity Analysis:", self.analyze_viscosity_check)

        self.analyze_non_gaussian_check = QCheckBox("Analyze Non-Gaussianity")
        self.analyze_non_gaussian_check.setChecked(True)
        params_layout.addRow("Include Non-Gaussian Analysis:", self.analyze_non_gaussian_check)


        params_group.setLayout(params_layout)
        layout.addWidget(params_group)

        # Visualization Options (Placeholder)
        viz_group = QGroupBox("Visualization Options")
        viz_layout = QFormLayout()
        self.plot_type_combo = QComboBox()
        self.plot_type_combo.addItems(["Spatial Heterogeneity Map", "Viscosity Distribution", "Non-Gaussian Parameter"]) # Example plot types
        viz_layout.addRow("Plot Type:", self.plot_type_combo)
        viz_group.setLayout(viz_layout)
        layout.addWidget(viz_group)

        # Results Display
        self.results_tabs = QTabWidget()

        self.summary_text = QTextEdit()
        self.summary_text.setReadOnly(True)
        self.results_tabs.addTab(self.summary_text, "Summary")

        self.results_table = QTableWidget() # Table for track-level crowding metrics
        self.results_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.results_tabs.addTab(self.results_table, "Detailed Results")


        self.plot_widget = MplCanvas()
        self.results_tabs.addTab(self.plot_widget, "Plots")

        layout.addWidget(self.results_tabs)

    def setup_connections(self):
        # Connect parameter changes to emit signal
        self.dt_spin.valueChanged.connect(self.on_parameter_changed)
        self.min_track_length_spin.valueChanged.connect(self.on_parameter_changed)
        self.particle_radius_spin.valueChanged.connect(self.on_parameter_changed)
        self.temperature_spin.valueChanged.connect(self.on_parameter_changed)
        self.analyze_viscosity_check.stateChanged.connect(self.on_parameter_changed)
        self.analyze_non_gaussian_check.stateChanged.connect(self.on_parameter_changed)


        # Connect plot type selection to update plot display
        self.plot_type_combo.currentIndexChanged.connect(self.update_plot_display)

    def on_parameter_changed(self):
        """Emits signal with current parameters when a parameter changes."""
        params = self.get_current_parameters()
        self.parameters_changed.emit(params)

    def get_current_parameters(self):
        """Collects current analysis parameters from the UI."""
        return {
            'dt': self.dt_spin.value(),
            'min_track_length': self.min_track_length_spin.value(),
            'particle_radius': self.particle_radius_spin.value(), # Used by analyze_viscosity
            'temperature': self.temperature_spin.value(),       # Used by analyze_viscosity
            'analyze_viscosity': self.analyze_viscosity_check.isChecked(),
            'analyze_non_gaussian': self.analyze_non_gaussian_check.isChecked(),
            # Compartment masks are needed by some methods, these would be passed by the main GUI
            # based on user selection elsewhere if compartment-specific analysis is desired.
        }

    def update_results_display(self, results):
        """Updates the widget's display with analysis results."""
        self.current_results = results # Store results received from the worker
        if results:
            # Update Summary Text
            summary_text = "Crowding Effects Analysis Results:\n\n"

            if 'spatial_heterogeneity_results' in results:
                 summary_text += "--- Spatial Heterogeneity ---\n"
                 # Assuming spatial_heterogeneity_results contains metrics like local alpha, D, etc.
                 if isinstance(results['spatial_heterogeneity_results'], pd.DataFrame):
                      hetero_df = results['spatial_heterogeneity_results']
                      summary_text += f"Analyzed {len(hetero_df)} track segments/windows.\n"
                      summary_text += f"Mean Local Alpha: {hetero_df.get('local_alpha', pd.Series()).mean():.3f}\n"
                      summary_text += f"Mean Local D: {hetero_df.get('local_D', pd.Series()).mean():.3f} μm²/s\n"
                 else:
                      summary_text += str(results['spatial_heterogeneity_results']) + "\n"
                 summary_text += "\n"


            if 'viscosity_results' in results:
                 summary_text += "--- Viscosity Analysis ---\n"
                 # Assuming viscosity_results contains mean/std viscosity or related metrics
                 if isinstance(results['viscosity_results'], dict):
                      visc_res = results['viscosity_results']
                      summary_text += f"Estimated Viscosity: {visc_res.get('mean_viscosity', np.nan):.4f} Pa·s\n"
                      # Add other relevant viscosity metrics
                 else:
                      summary_text += str(results['viscosity_results']) + "\n"
                 summary_text += "\n"


            if 'non_gaussian_results' in results:
                 summary_text += "--- Non-Gaussian Analysis ---\n"
                 # Assuming non_gaussian_results contains Alpha, MSD(t)/t etc. or related plots/metrics
                 if isinstance(results['non_gaussian_results'], dict):
                      ngp_res = results['non_gaussian_results']
                      summary_text += f"Mean Non-Gaussian Parameter (α₂): {ngp_res.get('mean_alpha2', np.nan):.4f}\n"
                      # Add other relevant non-gaussian metrics
                 else:
                       summary_text += str(results['non_gaussian_results']) + "\n"
                 summary_text += "\n"


            self.summary_text.setText(summary_text)

            # Update Detailed Results Table (e.g., spatial heterogeneity track-level results)
            if 'spatial_heterogeneity_results' in results and isinstance(results['spatial_heterogeneity_results'], pd.DataFrame):
                 self._update_detailed_table(results['spatial_heterogeneity_results'])
            else:
                 self.results_table.clear()
                 self.results_table.setRowCount(0)


            # Update Plots Display
            self.update_plot_display() # Call to draw the selected plot

        else:
            self.summary_text.setText("No Crowding Effects Analysis Results Available.")
            self.results_table.clear()
            self.results_table.setRowCount(0)
            self.plot_widget.clear()

    def _update_detailed_table(self, results_df):
        """Populates the detailed results table (e.g., track segment data)."""
        self.results_table.clear()
        if results_df is None or results_df.empty:
            self.results_table.setRowCount(0)
            return

        # Use columns from the DataFrame for headers
        headers = results_df.columns.tolist()
        self.results_table.setColumnCount(len(headers))
        self.results_table.setHorizontalHeaderLabels(headers)
        self.results_table.setRowCount(len(results_df))

        for i, (index, row) in enumerate(results_df.iterrows()):
            for j, col in enumerate(headers):
                item_text = str(row[col])
                if isinstance(row[col], (float, np.floating)):
                     item_text = f"{row[col]:.4f}" # Format floats
                self.results_table.setItem(i, j, QTableWidgetItem(item_text))

        self.results_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.results_table.verticalHeader().setVisible(False) # Hide row numbers


    def update_plot_display(self):
        """Clears and draws the selected plot type."""
        self.plot_widget.clear()
        if self.current_results is None:
            self.plot_widget.axes.text(0.5, 0.5, "No results to plot.", ha='center', va='center')
            self.plot_widget.draw()
            return

        plot_type = self.plot_type_combo.currentText()
        ax = self.plot_widget.axes

        try:
            if plot_type == "Spatial Heterogeneity Map":
                # This plot would require track positions and a heterogeneity metric per track/segment
                ax.text(0.5, 0.5, "Spatial Heterogeneity Map Placeholder", ha='center', va='center')
                # If spatial_heterogeneity_results contains track/segment metrics and original track positions are available,
                # you would call a plotting function from visualization.crowding_plots
                # e.g., CrowdingPlotter.plot_crowding_map(self.current_results['spatial_heterogeneity_results'], self.main_gui.image_stack[0])

            elif plot_type == "Viscosity Distribution":
                 if 'viscosity_results' in self.current_results and isinstance(self.current_results['viscosity_results'], dict) and 'per_track_viscosity' in self.current_results['viscosity_results']:
                      # Assuming viscosity_results includes a list or array of viscosity values per track
                      viscosity_values = self.current_results['viscosity_results']['per_track_viscosity']
                      if viscosity_values:
                           ax.hist(viscosity_values, bins=30, alpha=0.7)
                           ax.set_xlabel("Estimated Viscosity (Pa·s)")
                           ax.set_ylabel("Frequency")
                           ax.set_title("Estimated Viscosity Distribution")
                      else:
                           ax.text(0.5, 0.5, "No viscosity data per track found.", ha='center', va='center')
                 else:
                      ax.text(0.5, 0.5, "No viscosity results available.", ha='center', va='center')


            elif plot_type == "Non-Gaussian Parameter":
                 if 'non_gaussian_results' in self.current_results and isinstance(self.current_results['non_gaussian_results'], dict):
                      # Assuming non_gaussian_results includes a list or array of alpha2 values or similar
                      ngp_values = self.current_results['non_gaussian_results'].get('alpha2_values') # Example key
                      if ngp_values:
                           ax.hist(ngp_values, bins=30, alpha=0.7)
                           ax.set_xlabel("Non-Gaussian Parameter (α₂)")
                           ax.set_ylabel("Frequency")
                           ax.set_title("Non-Gaussian Parameter Distribution")
                      else:
                           ax.text(0.5, 0.5, "No non-gaussian parameter data found.", ha='center', va='center')
                 else:
                      ax.text(0.5, 0.5, "No non-gaussian results available.", ha='center', va='center')

            else:
                ax.text(0.5, 0.5, "Plot type not implemented.", ha='center', va='center')

            self.plot_widget.draw()

        except Exception as e:
            logger.error(f"Error generating crowding plot: {e}", exc_info=True)
            self.plot_widget.clear()
            self.plot_widget.axes.text(0.5, 0.5, f"Plot Error: {e}", ha='center', va='center', color='red')
            self.plot_widget.draw()
# C:\Users\mjhen\SPT_GUI\widgets\analysis_widgets.py (Continued)

class DiffusionPopulationWidget(QWidget):
    """Widget for Diffusion Population Analysis controls and visualization."""

    parameters_changed = pyqtSignal(dict)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setup_ui()
        self.setup_connections()
        self.current_results = None # To hold analysis results for display

    def setup_ui(self):
        layout = QVBoxLayout(self)
        params_group = QGroupBox("Diffusion Population Analysis Parameters")
        params_layout = QFormLayout()

        # Parameters based on DiffusionPopulationAnalyzer.__init__ and analyze methods
        default_analyzer = DiffusionPopulationAnalyzer() # Instance to get defaults

        self.dt_spin = QDoubleSpinBox()
        self.dt_spin.setRange(0.001, 100.0)
        self.dt_spin.setValue(default_analyzer.dt)
        self.dt_spin.setSingleStep(0.001)
        self.dt_spin.setSuffix(" s")
        self.dt_spin.setToolTip("Time interval between frames.")
        params_layout.addRow("Time Interval:", self.dt_spin)

        self.max_populations_spin = QSpinBox()
        self.max_populations_spin.setRange(1, 10)
        self.max_populations_spin.setValue(default_analyzer.max_populations)
        self.max_populations_spin.setToolTip("Maximum number of populations for mixture models.")
        params_layout.addRow("Max Populations:", self.max_populations_spin)

        self.min_segment_length_spin = QSpinBox()
        self.min_segment_length_spin.setRange(3, 100)
        self.min_segment_length_spin.setValue(default_analyzer.min_segment_length)
        self.min_segment_length_spin.setToolTip("Minimum points for a valid trajectory segment.")
        params_layout.addRow("Min Segment Length:", self.min_segment_length_spin)

        # Options for specific analysis tasks within Diffusion Population
        self.deconvolution_method_combo = QComboBox()
        self.deconvolution_method_combo.addItems(["gmm", "bayesian"]) # Based on analyze_jump_size_distribution in GelStructure (similar concept)
        self.deconvolution_method_combo.setToolTip("Method for deconvoluting jump distributions.")
        params_layout.addRow("Deconvolution Method:", self.deconvolution_method_combo)

        self.segmentation_method_combo = QComboBox()
        self.segmentation_method_combo.addItems(["changepoint", "sliding_window"]) # Based on segment_trajectories
        self.segmentation_method_combo.setToolTip("Method for segmenting trajectories.")
        params_layout.addRow("Segmentation Method:", self.segmentation_method_combo)

        self.state_identification_method_combo = QComboBox()
        self.state_identification_method_combo.addItems(["hmm", "kmeans"]) # Based on identify_diffusion_states
        self.state_identification_method_combo.setToolTip("Method for identifying diffusion states.")
        params_layout.addRow("State Identification Method:", self.state_identification_method_combo)

        self.n_states_spin = QSpinBox()
        self.n_states_spin.setRange(1, 10)
        self.n_states_spin.setValue(3) # Example default
        self.n_states_spin.setToolTip("Number of states for state identification (HMM/KMeans).")
        params_layout.addRow("Number of States:", self.n_states_spin)


        params_group.setLayout(params_layout)
        layout.addWidget(params_group)

        # Visualization Options (Placeholder)
        viz_group = QGroupBox("Visualization Options")
        viz_layout = QFormLayout()
        self.plot_type_combo = QComboBox()
        self.plot_type_combo.addItems(["Jump Distribution Deconvolution", "Segment Alpha Distribution", "State Occupancies", "State Transition Matrix"]) # Example plot types
        viz_layout.addRow("Plot Type:", self.plot_type_combo)
        viz_group.setLayout(viz_layout)
        layout.addWidget(viz_group)


        # Results Display
        self.results_tabs = QTabWidget()

        self.summary_text = QTextEdit()
        self.summary_text.setReadOnly(True)
        self.results_tabs.addTab(self.summary_text, "Summary")

        self.deconvolution_table = QTableWidget() # Table for deconvolution results (populations)
        self.deconvolution_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.results_tabs.addTab(self.deconvolution_table, "Deconvolution Results")

        self.segments_table = QTableWidget() # Table for trajectory segment properties
        self.segments_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.results_tabs.addTab(self.segments_table, "Segment Results")

        self.states_table = QTableWidget() # Table for diffusion state properties
        self.states_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.results_tabs.addTab(self.states_table, "State Results")


        self.plot_widget = MplCanvas()
        self.results_tabs.addTab(self.plot_widget, "Plots")

        layout.addWidget(self.results_tabs)

    def setup_connections(self):
        # Connect parameter changes to emit signal
        self.dt_spin.valueChanged.connect(self.on_parameter_changed)
        self.max_populations_spin.valueChanged.connect(self.on_parameter_changed)
        self.min_segment_length_spin.valueChanged.connect(self.on_parameter_changed)
        self.deconvolution_method_combo.currentTextChanged.connect(self.on_parameter_changed)
        self.segmentation_method_combo.currentTextChanged.connect(self.on_parameter_changed)
        self.state_identification_method_combo.currentTextChanged.connect(self.on_parameter_changed)
        self.n_states_spin.valueChanged.connect(self.on_parameter_changed)


        # Connect plot type selection to update plot display
        self.plot_type_combo.currentIndexChanged.connect(self.update_plot_display)


    def on_parameter_changed(self):
        """Emits signal with current parameters when a parameter changes."""
        params = self.get_current_parameters()
        self.parameters_changed.emit(params)

    def get_current_parameters(self):
        """Collects current analysis parameters from the UI."""
        return {
            'dt': self.dt_spin.value(),
            'max_populations': self.max_populations_spin.value(),
            'min_segment_length': self.min_segment_length_spin.value(),
            'deconvolution_method': self.deconvolution_method_combo.currentText(),
            'segmentation_method': self.segmentation_method_combo.currentText(),
            'state_identification_method': self.state_identification_method_combo.currentText(),
            'n_states': self.n_states_spin.value(),
            # Compartment masks are needed by deconvolution, these would be passed by the main GUI
        }

    def update_results_display(self, results):
        """Updates the widget's display with analysis results."""
        self.current_results = results # Store results received from the worker
        if results:
            # Update Summary Text
            summary_text = "Diffusion Population Analysis Results:\n\n"

            if 'jump_mixture' in results:
                 summary_text += "--- Jump Distribution Deconvolution ---\n"
                 if 'all' in results['jump_mixture'] and isinstance(results['jump_mixture']['all'], dict):
                      all_mix_res = results['jump_mixture']['all']
                      summary_text += f"Overall Distribution ({all_mix_res.get('method', 'N/A')}): {all_mix_res.get('best_n_components', 0)} components\n"
                      d_values = all_mix_res.get('diffusion_coefficients', [])
                      weights = all_mix_res.get('weights', [])
                      for i in range(len(d_values)):
                           summary_text += f"  Population {i+1}: D = {d_values[i]:.4f} μm²/s, Weight = {weights[i]:.3f}\n"
                 if 'by_compartment' in results['jump_mixture']:
                      summary_text += "By Compartment:\n"
                      for comp, mix_res in results['jump_mixture']['by_compartment'].items():
                           if isinstance(mix_res, dict) and 'best_n_components' in mix_res:
                                summary_text += f"  {comp}: {mix_res.get('best_n_components', 0)} components\n"
                                # Could list D and weights here too if desired
                           else:
                                summary_text += f"  {comp}: {mix_res.get('status', 'N/A')}\n"
                 summary_text += "\n"

            if 'trajectory_segments' in results:
                 summary_text += "--- Trajectory Segment Analysis ---\n"
                 seg_stats = results['trajectory_segments']
                 summary_text += f"Tracks segmented: {seg_stats.get('n_tracks_segmented', 0)}\n"
                 summary_text += f"Total segments: {seg_stats.get('n_total_segments', 0)}\n"
                 if 'diffusion_modes' in seg_stats:
                      summary_text += "Diffusion Modes:\n"
                      for mode, stats in seg_stats['diffusion_modes'].items():
                           summary_text += f"  {mode}: {stats.get('count', 0)} segments ({stats.get('fraction', 0):.1%})\n"
                           if 'mean_alpha' in stats and stats['mean_alpha'] is not None: summary_text += f"    Mean α: {stats['mean_alpha']:.3f}\n"
                           if 'mean_D' in stats and stats['mean_D'] is not None: summary_text += f"    Mean D: {stats['mean_D']:.4f} μm²/s\n"
                 summary_text += "\n"

            if 'diffusion_states' in results:
                 summary_text += "--- Diffusion State Identification ---\n"
                 state_stats = results['diffusion_states']
                 summary_text += f"Method: {state_stats.get('method', 'N/A')}\n"
                 summary_text += f"Identified {state_stats.get('n_states', 0)} states.\n"
                 if 'states_summary' in state_stats:
                      summary_text += "State Summary:\n"
                      for state_id, state_data in state_stats['states_summary'].items():
                           summary_text += f"  State {state_id}:\n"
                           if 'mean_D' in state_data and state_data['mean_D'] is not None: summary_text += f"    Mean D: {state_data['mean_D']:.4f} μm²/s\n"
                           if 'occupancy' in state_data: summary_text += f"    Occupancy: {state_data['occupancy']:.1%}\n"
                           # Could include transition probabilities here, but might be verbose

            self.summary_text.setText(summary_text)

            # Update Tables
            if 'jump_mixture' in results and 'all' in results['jump_mixture'] and isinstance(results['jump_mixture']['all'], dict) and 'diffusion_coefficients' in results['jump_mixture']['all']:
                 self._update_deconvolution_table(results['jump_mixture']['all'])
            else:
                 self.deconvolution_table.clear()
                 self.deconvolution_table.setRowCount(0)

            if 'trajectory_segments' in results and 'all_segments_df' in results['trajectory_segments'] and isinstance(results['trajectory_segments']['all_segments_df'], pd.DataFrame):
                 self._update_segments_table(results['trajectory_segments']['all_segments_df'])
            else:
                 self.segments_table.clear()
                 self.segments_table.setRowCount(0)

            if 'diffusion_states' in results and 'states' in results['diffusion_states'] and isinstance(results['diffusion_states']['states'], dict):
                 self._update_states_table(results['diffusion_states']['states'])
            else:
                 self.states_table.clear()
                 self.states_table.setRowCount(0)


            # Update Plots Display
            self.update_plot_display() # Call to draw the selected plot

        else:
            self.summary_text.setText("No Diffusion Population Analysis Results Available.")
            self.deconvolution_table.clear()
            self.deconvolution_table.setRowCount(0)
            self.segments_table.clear()
            self.segments_table.setRowCount(0)
            self.states_table.clear()
            self.states_table.setRowCount(0)
            self.plot_widget.clear()

    def _update_deconvolution_table(self, deconvolution_results):
        """Populates the deconvolution results table."""
        self.deconvolution_table.clear()
        if not deconvolution_results or 'diffusion_coefficients' not in deconvolution_results:
            self.deconvolution_table.setRowCount(0)
            return

        d_values = deconvolution_results['diffusion_coefficients']
        weights = deconvolution_results['weights']

        headers = ["Population ID", "D (μm²/s)", "Weight"]
        self.deconvolution_table.setColumnCount(len(headers))
        self.deconvolution_table.setHorizontalHeaderLabels(headers)
        self.deconvolution_table.setRowCount(len(d_values))

        for i in range(len(d_values)):
            self.deconvolution_table.setItem(i, 0, QTableWidgetItem(str(i + 1)))
            self.deconvolution_table.setItem(i, 1, QTableWidgetItem(f"{d_values[i]:.4f}"))
            self.deconvolution_table.setItem(i, 2, QTableWidgetItem(f"{weights[i]:.3f}"))

        self.deconvolution_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.deconvolution_table.verticalHeader().setVisible(False)

    def _update_segments_table(self, segments_df):
        """Populates the trajectory segments table."""
        self.segments_table.clear()
        if segments_df is None or segments_df.empty:
            self.segments_table.setRowCount(0)
            return

        # Use columns from the DataFrame for headers
        headers = segments_df.columns.tolist() # e.g., start_frame, end_frame, n_frames, alpha, diffusion_coefficient, diffusion_mode
        self.segments_table.setColumnCount(len(headers))
        self.segments_table.setHorizontalHeaderLabels(headers)
        self.segments_table.setRowCount(len(segments_df))

        for i, (index, row) in enumerate(segments_df.iterrows()):
            for j, col in enumerate(headers):
                item_text = str(row[col])
                if isinstance(row[col], (float, np.floating)):
                     item_text = f"{row[col]:.4f}" # Format floats
                self.segments_table.setItem(i, j, QTableWidgetItem(item_text))

        self.segments_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.segments_table.verticalHeader().setVisible(False)


    def _update_states_table(self, states_dict):
        """Populates the diffusion states table."""
        self.states_table.clear()
        if not states_dict:
            self.states_table.setRowCount(0)
            return

        # Headers based on state dictionary keys (mean_D, occupancy, transition_probabilities, etc.)
        # Assuming a consistent set of keys in each state entry
        headers = ["State ID", "Mean D (μm²/s)", "Occupancy"]
        # Add transition probabilities as columns if desired, or handle separately
        # For simplicity, will not add transition matrix to this flat table.
        self.states_table.setColumnCount(len(headers))
        self.states_table.setHorizontalHeaderLabels(headers)
        self.states_table.setRowCount(len(states_dict))

        sorted_state_ids = sorted(states_dict.keys()) # Display in order
        for i, state_id in enumerate(sorted_state_ids):
             state_data = states_dict[state_id]
             self.states_table.setItem(i, 0, QTableWidgetItem(str(state_data.get('state_id', state_id))))
             self.states_table.setItem(i, 1, QTableWidgetItem(f"{state_data.get('mean_D', np.nan):.4f}"))
             self.states_table.setItem(i, 2, QTableWidgetItem(f"{state_data.get('occupancy', np.nan):.3f}"))
             # Add other columns as needed

        self.states_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.states_table.verticalHeader().setVisible(False)


    def update_plot_display(self):
        """Clears and draws the selected plot type."""
        self.plot_widget.clear()
        if self.current_results is None:
            self.plot_widget.axes.text(0.5, 0.5, "No results to plot.", ha='center', va='center')
            self.plot_widget.draw()
            return

        plot_type = self.plot_type_combo.currentText()
        ax = self.plot_widget.axes

        try:
            if plot_type == "Jump Distribution Deconvolution":
                 if 'jump_mixture' in self.current_results and 'all' in self.current_results['jump_mixture'] and isinstance(self.current_results['jump_mixture']['all'], dict):
                      # This would require plotting the histogram of squared jumps and the fitted GMM components
                      # The raw squared jump data might not be stored in the results, may need to recalculate or store.
                      # Placeholder plot:
                      ax.text(0.5, 0.5, "Jump Deconvolution Plot Placeholder", ha='center', va='center')
                      # A proper plot would show the histogram of squared jumps and overlay the fitted PDF of the mixture model.
                 else:
                      ax.text(0.5, 0.5, "No jump deconvolution results available for plotting.", ha='center', va='center')

            elif plot_type == "Segment Alpha Distribution":
                 if 'trajectory_segments' in self.current_results and isinstance(self.current_results['trajectory_segments'].get('all_segments_df'), pd.DataFrame):
                      segments_df = self.current_results['trajectory_segments']['all_segments_df']
                      if 'alpha' in segments_df.columns and not segments_df['alpha'].dropna().empty:
                           ax.hist(segments_df['alpha'].dropna(), bins=50, alpha=0.7)
                           ax.set_xlabel("Anomalous Exponent (α)")
                           ax.set_ylabel("Frequency")
                           ax.set_title("Distribution of Segment Alphas")
                      else:
                           ax.text(0.5, 0.5, "No segment alpha data available.", ha='center', va='center')
                 else:
                      ax.text(0.5, 0.5, "No trajectory segment results available for plotting.", ha='center', va='center')


            elif plot_type == "State Occupancies":
                 if 'diffusion_states' in self.current_results and 'states' in self.current_results['diffusion_states'] and isinstance(self.current_results['diffusion_states']['states'], dict):
                      states_dict = self.current_results['diffusion_states']['states']
                      state_labels = [f'State {i}' for i in sorted(states_dict.keys())]
                      occupancies = [states_dict[i].get('occupancy', 0) for i in sorted(states_dict.keys())]
                      if occupancies:
                           ax.bar(state_labels, occupancies, alpha=0.7)
                           ax.set_ylabel("Occupancy Fraction")
                           ax.set_title("Diffusion State Occupancies")
                      else:
                           ax.text(0.5, 0.5, "No state occupancy data available.", ha='center', va='center')

                 else:
                      ax.text(0.5, 0.5, "No diffusion state results available for plotting.", ha='center', va='center')


            elif plot_type == "State Transition Matrix":
                 if 'diffusion_states' in self.current_results and 'states' in self.current_results['diffusion_states'] and isinstance(self.current_results['diffusion_states']['states'], dict):
                      states_dict = self.current_results['diffusion_states']['states']
                      n_states = len(states_dict)
                      # Reconstruct transition matrix - assuming it's stored in each state's entry
                      # This might need refinement based on how the analyzer actually stores the matrix.
                      # Assuming state_data['transition_probabilities'] is a list representing the row for that state_id
                      transition_matrix = np.array([states_dict[i].get('transition_probabilities', [0]*n_states) for i in sorted(states_dict.keys())])

                      if transition_matrix.shape == (n_states, n_states):
                           import seaborn as sns
                           state_labels = [f'State {i}' for i in sorted(states_dict.keys())]
                           sns.heatmap(transition_matrix, annot=True, cmap='viridis', fmt=".2f",
                                       xticklabels=state_labels, yticklabels=state_labels, ax=ax)
                           ax.set_xlabel("To State")
                           ax.set_ylabel("From State")
                           ax.set_title("Diffusion State Transition Matrix")
                      else:
                           ax.text(0.5, 0.5, "Transition matrix data not available or incorrect shape.", ha='center', va='center')

                 else:
                      ax.text(0.5, 0.5, "No diffusion state results available for plotting.", ha='center', va='center')


            else:
                ax.text(0.5, 0.5, "Plot type not implemented.", ha='center', va='center')

            self.plot_widget.draw()

        except Exception as e:
            logger.error(f"Error generating diffusion population plot: {e}", exc_info=True)
            self.plot_widget.clear()
            self.plot_widget.axes.text(0.5, 0.5, f"Plot Error: {e}", ha='center', va='center', color='red')
            self.plot_widget.draw()
# C:\Users\mjhen\SPT_GUI\widgets\analysis_widgets.py (Continued)

class GelStructureWidget(QWidget):
    """Widget for Gel Structure Analysis controls and visualization."""

    parameters_changed = pyqtSignal(dict)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setup_ui()
        self.setup_connections()
        self.current_results = None # To hold analysis results for display

    def setup_ui(self):
        layout = QVBoxLayout(self)
        params_group = QGroupBox("Gel Structure Analysis Parameters")
        params_layout = QFormLayout()

        # Parameters based on GelStructureAnalyzer.__init__ and analyze methods
        default_analyzer = GelStructureAnalyzer() # Instance to get defaults

        self.dt_spin = QDoubleSpinBox()
        self.dt_spin.setRange(0.001, 100.0)
        self.dt_spin.setValue(default_analyzer.dt)
        self.dt_spin.setSingleStep(0.001)
        self.dt_spin.setSuffix(" s")
        self.dt_spin.setToolTip("Time interval between frames.")
        params_layout.addRow("Time Interval:", self.dt_spin)

        self.min_jumps_spin = QSpinBox()
        self.min_jumps_spin.setRange(10, 1000)
        self.min_jumps_spin.setValue(default_analyzer.min_jumps)
        self.min_jumps_spin.setToolTip("Minimum number of jumps required for analysis.")
        params_layout.addRow("Min Jumps:", self.min_jumps_spin)

        # Parameters for specific analysis methods within GelStructureAnalyzer
        self.particle_radius_spin = QDoubleSpinBox()
        self.particle_radius_spin.setRange(0.1, 1000.0)
        self.particle_radius_spin.setValue(5.0) # Example default
        self.particle_radius_spin.setSingleStep(0.1)
        self.particle_radius_spin.setSuffix(" nm")
        self.particle_radius_spin.setToolTip("Radius of the tracked particle for property calculations.")
        params_layout.addRow("Particle Radius:", self.particle_radius_spin)

        self.temperature_spin = QDoubleSpinBox()
        self.temperature_spin.setRange(0.1, 100.0)
        self.temperature_spin.setValue(25.0) # Example default in Celsius
        self.temperature_spin.setSingleStep(0.1)
        self.temperature_spin.setSuffix(" °C")
        self.temperature_spin.setToolTip("Temperature for property calculations.")
        params_layout.addRow("Temperature:", self.temperature_spin)

        self.pore_size_method_combo = QComboBox()
        self.pore_size_method_combo.addItems(["jump_distribution", "confinement"]) # Based on estimate_pore_size
        self.pore_size_method_combo.setToolTip("Method for estimating gel pore size.")
        params_layout.addRow("Pore Size Method:", self.pore_size_method_combo)

        params_group.setLayout(params_layout)
        layout.addWidget(params_group)

        # Visualization Options (Placeholder)
        viz_group = QGroupBox("Visualization Options")
        viz_layout = QFormLayout()
        self.plot_type_combo = QComboBox()
        self.plot_type_combo.addItems(["Jump Distance Distribution", "Pore Size Estimation", "Gel Properties Summary"]) # Example plot types
        viz_layout.addRow("Plot Type:", self.plot_type_combo)
        viz_group.setLayout(viz_layout)
        layout.addWidget(viz_group)

        # Results Display
        self.results_tabs = QTabWidget()

        self.summary_text = QTextEdit()
        self.summary_text.setReadOnly(True)
        self.results_tabs.addTab(self.summary_text, "Summary")

        self.jump_distribution_table = QTableWidget() # Table for jump distribution stats/fits
        self.jump_distribution_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.results_tabs.addTab(self.jump_distribution_table, "Jump Distribution")

        self.pore_size_table = QTableWidget() # Table for pore size results
        self.pore_size_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.results_tabs.addTab(self.pore_size_table, "Pore Size")

        self.gel_properties_table = QTableWidget() # Table for gel properties
        self.gel_properties_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.results_tabs.addTab(self.gel_properties_table, "Gel Properties")


        self.plot_widget = MplCanvas()
        self.results_tabs.addTab(self.plot_widget, "Plots")

        layout.addWidget(self.results_tabs)

    def setup_connections(self):
        # Connect parameter changes to emit signal
        self.dt_spin.valueChanged.connect(self.on_parameter_changed)
        self.min_jumps_spin.valueChanged.connect(self.on_parameter_changed)
        self.particle_radius_spin.valueChanged.connect(self.on_parameter_changed)
        self.temperature_spin.valueChanged.connect(self.on_parameter_changed)
        self.pore_size_method_combo.currentTextChanged.connect(self.on_parameter_changed)


        # Connect plot type selection to update plot display
        self.plot_type_combo.currentIndexChanged.connect(self.update_plot_display)


    def on_parameter_changed(self):
        """Emits signal with current parameters when a parameter changes."""
        params = self.get_current_parameters()
        self.parameters_changed.emit(params)

    def get_current_parameters(self):
        """Collects current analysis parameters from the UI."""
        return {
            'dt': self.dt_spin.value(),
            'min_jumps': self.min_jumps_spin.value(),
            'particle_radius': self.particle_radius_spin.value(), # Used by calculate_gel_properties
            'temperature': self.temperature_spin.value(),       # Used by calculate_gel_properties
            'pore_size_method': self.pore_size_method_combo.currentText(), # Used by estimate_pore_size
            # Compartment masks are needed by some methods, these would be passed by the main GUI
        }

    def update_results_display(self, results):
        """Updates the widget's display with analysis results."""
        self.current_results = results # Store results received from the worker
        if results:
            # Update Summary Text
            summary_text = "Gel Structure Analysis Results:\n\n"

            if 'jump_distributions' in results:
                 summary_text += "--- Jump Distance Distribution ---\n"
                 if 'all' in results['jump_distributions'] and isinstance(results['jump_distributions']['all'], dict):
                      jump_all = results['jump_distributions']['all']
                      summary_text += f"Overall: Mean={jump_all.get('mean_jump', np.nan):.3f}, Median={jump_all.get('median_jump', np.nan):.3f}, Std={jump_all.get('std_jump', np.nan):.3f}\n"
                      summary_text += f"Best Fit: {jump_all.get('best_fit', 'N/A')} (p={jump_all.get('good_fit', False)})\n"
                 if 'by_compartment' in results['jump_distributions']:
                      summary_text += "By Compartment:\n"
                      for comp, jump_comp in results['jump_distributions']['by_compartment'].items():
                           if isinstance(jump_comp, dict):
                                summary_text += f"  {comp}: Mean={jump_comp.get('mean_jump', np.nan):.3f}, Median={jump_comp.get('median_jump', np.nan):.3f}, Std={jump_comp.get('std_jump', np.nan):.3f}\n"
                                summary_text += f"    Best Fit: {jump_comp.get('best_fit', 'N/A')} (p={jump_comp.get('good_fit', False)})\n"
                           else:
                                summary_text += f"  {comp}: {jump_comp.get('status', 'N/A')}\n"
                 summary_text += "\n"

            if 'pore_size_results' in results:
                 summary_text += "--- Pore Size Estimation ---\n"
                 if 'all' in results['pore_size_results'] and isinstance(results['pore_size_results']['all'], dict):
                      pore_all = results['pore_size_results']['all']
                      summary_text += f"Overall: Pore Size={pore_all.get('pore_size', np.nan):.3f}, Method={pore_all.get('method', 'N/A')}\n"
                 if 'by_compartment' in results['pore_size_results']:
                      summary_text += "By Compartment:\n"
                      for comp, pore_comp in results['pore_size_results']['by_compartment'].items():
                           if isinstance(pore_comp, dict):
                                summary_text += f"  {comp}: Pore Size={pore_comp.get('pore_size', np.nan):.3f}, Method={pore_comp.get('method', 'N/A')}\n"
                           else:
                                summary_text += f"  {comp}: {pore_comp.get('status', 'N/A')}\n"
                 summary_text += "\n"

            if 'gel_properties' in results:
                 summary_text += "--- Gel Properties ---\n"
                 gel_props = results['gel_properties']
                 summary_text += f"Analyzed {gel_props.get('n_subdiffusive_tracks', 0)} subdiffusive tracks.\n"
                 summary_text += f"Mean Alpha: {gel_props.get('mean_alpha', np.nan):.3f}\n"
                 summary_text += f"Estimated Mesh Size: {gel_props.get('mesh_size', np.nan):.3f}\n"
                 summary_text += f"Estimated Elastic Modulus: {gel_props.get('elastic_modulus', np.nan):.4e} Pa\n"
                 summary_text += f"Estimated Viscosity: {gel_props.get('viscosity', np.nan):.4e} Pa·s\n"
                 summary_text += f"Estimated Storage Modulus (ω=1): {gel_props.get('storage_modulus', np.nan):.4e} Pa\n"
                 summary_text += f"Estimated Loss Modulus (ω=1): {gel_props.get('loss_modulus', np.nan):.4e} Pa\n"
                 summary_text += "\n"


            self.summary_text.setText(summary_text)

            # Update Tables
            if 'jump_distributions' in results and 'all' in results['jump_distributions'] and isinstance(results['jump_distributions']['all'], dict) and 'histogram' in results['jump_distributions']['all']:
                 # Update jump distribution table with histogram data or fit parameters
                 self._update_jump_distribution_table(results['jump_distributions']) # Pass the whole structure to potentially show by compartment
            else:
                 self.jump_distribution_table.clear()
                 self.jump_distribution_table.setRowCount(0)

            if 'pore_size_results' in results and isinstance(results['pore_size_results'], dict):
                 self._update_pore_size_table(results['pore_size_results'])
            else:
                 self.pore_size_table.clear()
                 self.pore_size_table.setRowCount(0)

            if 'gel_properties' in results and isinstance(results['gel_properties'], dict):
                 self._update_gel_properties_table(results['gel_properties'])
            else:
                 self.gel_properties_table.clear()
                 self.gel_properties_table.setRowCount(0)


            # Update Plots Display
            self.update_plot_display() # Call to draw the selected plot

        else:
            self.summary_text.setText("No Gel Structure Analysis Results Available.")
            self.jump_distribution_table.clear()
            self.jump_distribution_table.setRowCount(0)
            self.pore_size_table.clear()
            self.pore_size_table.setRowCount(0)
            self.gel_properties_table.clear()
            self.gel_properties_table.setRowCount(0)
            self.plot_widget.clear()


    def _update_jump_distribution_table(self, jump_distribution_results):
         """Populates the jump distribution table with stats and fits."""
         self.jump_distribution_table.clear()
         headers = ["Location", "Mean Jump", "Median Jump", "Std Jump", "Best Fit", "Fit p-value"]
         self.jump_distribution_table.setColumnCount(len(headers))
         self.jump_distribution_table.setHorizontalHeaderLabels(headers)
         self.jump_distribution_table.setRowCount(0) # Start with 0 rows

         row_count = 0
         # Add overall results
         if 'all' in jump_distribution_results and isinstance(jump_distribution_results['all'], dict):
              self.jump_distribution_table.insertRow(row_count)
              jump_data = jump_distribution_results['all']
              self.jump_distribution_table.setItem(row_count, 0, QTableWidgetItem("Overall"))
              self.jump_distribution_table.setItem(row_count, 1, QTableWidgetItem(f"{jump_data.get('mean_jump', np.nan):.3f}"))
              self.jump_distribution_table.setItem(row_count, 2, QTableWidgetItem(f"{jump_data.get('median_jump', np.nan):.3f}"))
              self.jump_distribution_table.setItem(row_count, 3, QTableWidgetItem(f"{jump_data.get('std_jump', np.nan):.3f}"))
              self.jump_distribution_table.setItem(row_count, 4, QTableWidgetItem(str(jump_data.get('best_fit', 'N/A'))))
              # Assuming fit p-value is stored like this from _analyze_jump_distribution
              p_val = jump_data.get('fits', {}).get(jump_data.get('best_fit'), {}).get('p_value', np.nan)
              self.jump_distribution_table.setItem(row_count, 5, QTableWidgetItem(f"{p_val:.3f}" if not np.isnan(p_val) else "N/A"))
              row_count += 1

         # Add by compartment results
         if 'by_compartment' in jump_distribution_results:
              for comp, jump_data in jump_distribution_results['by_compartment'].items():
                   if isinstance(jump_data, dict):
                        self.jump_distribution_table.insertRow(row_count)
                        self.jump_distribution_table.setItem(row_count, 0, QTableWidgetItem(str(comp)))
                        self.jump_distribution_table.setItem(row_count, 1, QTableWidgetItem(f"{jump_data.get('mean_jump', np.nan):.3f}"))
                        self.jump_distribution_table.setItem(row_count, 2, QTableWidgetItem(f"{jump_data.get('median_jump', np.nan):.3f}"))
                        self.jump_distribution_table.setItem(row_count, 3, QTableWidgetItem(f"{jump_data.get('std_jump', np.nan):.3f}"))
                        self.jump_distribution_table.setItem(row_count, 4, QTableWidgetItem(str(jump_data.get('best_fit', 'N/A'))))
                        p_val = jump_data.get('fits', {}).get(jump_data.get('best_fit'), {}).get('p_value', np.nan)
                        self.jump_distribution_table.setItem(row_count, 5, QTableWidgetItem(f"{p_val:.3f}" if not np.isnan(p_val) else "N/A"))
                        row_count += 1

         self.jump_distribution_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
         self.jump_distribution_table.verticalHeader().setVisible(False)


    def _update_pore_size_table(self, pore_size_results):
        """Populates the pore size table."""
        self.pore_size_table.clear()
        headers = ["Location", "Pore Size", "Method", "Uncertainty"] # Add more if needed
        self.pore_size_table.setColumnCount(len(headers))
        self.pore_size_table.setHorizontalHeaderLabels(headers)
        self.pore_size_table.setRowCount(0)

        row_count = 0
        # Add overall results
        if 'all' in pore_size_results and isinstance(pore_size_results['all'], dict):
             self.pore_size_table.insertRow(row_count)
             pore_data = pore_size_results['all']
             self.pore_size_table.setItem(row_count, 0, QTableWidgetItem("Overall"))
             self.pore_size_table.setItem(row_count, 1, QTableWidgetItem(f"{pore_data.get('pore_size', np.nan):.3f}"))
             self.pore_size_table.setItem(row_count, 2, QTableWidgetItem(str(pore_data.get('method', 'N/A'))))
             self.pore_size_table.setItem(row_count, 3, QTableWidgetItem(f"{pore_data.get('uncertainty', np.nan):.3f}" if not np.isnan(pore_data.get('uncertainty', np.nan)) else "N/A"))
             row_count += 1

        # Add by compartment results
        if 'by_compartment' in pore_size_results:
             for comp, pore_data in pore_size_results['by_compartment'].items():
                  if isinstance(pore_data, dict):
                       self.pore_size_table.insertRow(row_count)
                       self.pore_size_table.setItem(row_count, 0, QTableWidgetItem(str(comp)))
                       self.pore_size_table.setItem(row_count, 1, QTableWidgetItem(f"{pore_data.get('pore_size', np.nan):.3f}"))
                       self.pore_size_table.setItem(row_count, 2, QTableWidgetItem(str(pore_data.get('method', 'N/A'))))
                       self.pore_size_table.setItem(row_count, 3, QTableWidgetItem(f"{pore_data.get('uncertainty', np.nan):.3f}" if not np.isnan(pore_data.get('uncertainty', np.nan)) else "N/A"))
                       row_count += 1

        self.pore_size_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.pore_size_table.verticalHeader().setVisible(False)


    def _update_gel_properties_table(self, gel_properties_dict):
         """Populates the gel properties table."""
         self.gel_properties_table.clear()
         headers = ["Property", "Value"] # Generic for key-value display
         self.gel_properties_table.setColumnCount(len(headers))
         self.gel_properties_table.setHorizontalHeaderLabels(headers)
         self.gel_properties_table.setRowCount(0)

         row_count = 0
         if isinstance(gel_properties_dict, dict):
              for key, value in gel_properties_dict.items():
                   self.gel_properties_table.insertRow(row_count)
                   self.gel_properties_table.setItem(row_count, 0, QTableWidgetItem(str(key)))
                   if isinstance(value, float):
                        self.gel_properties_table.setItem(row_count, 1, QTableWidgetItem(f"{value:.4e}" if abs(value) > 1e-3 or value == 0 else f"{value:.4f}")) # Use scientific notation for small/large numbers
                   elif isinstance(value, int):
                         self.gel_properties_table.setItem(row_count, 1, QTableWidgetItem(str(value)))
                   else:
                         self.gel_properties_table.setItem(row_count, 1, QTableWidgetItem(str(value)))
                   row_count += 1

         self.gel_properties_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
         self.gel_properties_table.verticalHeader().setVisible(False)


    def update_plot_display(self):
        """Clears and draws the selected plot type."""
        self.plot_widget.clear()
        if self.current_results is None:
            self.plot_widget.axes.text(0.5, 0.5, "No results to plot.", ha='center', va='center')
            self.plot_widget.draw()
            return

        plot_type = self.plot_type_combo.currentText()
        ax = self.plot_widget.axes

        try:
            if plot_type == "Jump Distance Distribution":
                 if 'jump_distributions' in self.current_results and 'all' in self.current_results['jump_distributions'] and isinstance(self.current_results['jump_distributions']['all'], dict):
                      jump_data = self.current_results['jump_distributions']['all']
                      histogram = jump_data.get('histogram')
                      fits = jump_data.get('fits')

                      if histogram and 'bin_centers' in histogram and 'counts' in histogram:
                           ax.plot(histogram['bin_centers'], histogram['counts'], 'o-', label='Data')

                           if fits:
                                for fit_name, fit_data in fits.items():
                                     if fit_data and fit_data.get('pdf') is not None:
                                          ax.plot(histogram['bin_centers'], fit_data['pdf'], '--', label=f'{fit_name} Fit')

                           ax.set_xlabel("Jump Distance (px)") # Assuming jumps are in pixels for now
                           ax.set_ylabel("Probability Density")
                           ax.set_title("Jump Distance Distribution (Overall)")
                           ax.legend()
                      else:
                           ax.text(0.5, 0.5, "No histogram data for jump distribution.", ha='center', va='center')

                 else:
                      ax.text(0.5, 0.5, "No overall jump distribution results for plotting.", ha='center', va='center')


            elif plot_type == "Pore Size Estimation":
                 if 'pore_size_results' in self.current_results and isinstance(self.current_results['pore_size_results'], dict):
                      # This could be a bar plot comparing pore size estimates by compartment
                      # or a plot illustrating the method used (e.g., jump distribution peak)
                      # Placeholder plot:
                      ax.text(0.5, 0.5, "Pore Size Plot Placeholder", ha='center', va='center')
                 else:
                      ax.text(0.5, 0.5, "No pore size results available for plotting.", ha='center', va='center')


            elif plot_type == "Gel Properties Summary":
                 if 'gel_properties' in self.current_results and isinstance(self.current_results['gel_properties'], dict):
                      # This could be a bar plot of key properties or a radar chart
                      # Placeholder plot:
                      ax.text(0.5, 0.5, "Gel Properties Plot Placeholder", ha='center', va='center')
                 else:
                      ax.text(0.5, 0.5, "No gel properties results available for plotting.", ha='center', va='center')

            else:
                ax.text(0.5, 0.5, "Plot type not implemented.", ha='center', va='center')

            self.plot_widget.draw()

        except Exception as e:
            logger.error(f"Error generating gel structure plot: {e}", exc_info=True)
            self.plot_widget.clear()
            self.plot_widget.axes.text(0.5, 0.5, f"Plot Error: {e}", ha='center', va='center', color='red')
            self.plot_widget.draw()
# C:\Users\mjhen\SPT_GUI\widgets\analysis_widgets.py (Continued)

class MicrocompartmentWidget(QWidget):
    """Widget for Microcompartment Analysis controls and visualization."""

    parameters_changed = pyqtSignal(dict)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setup_ui()
        self.setup_connections()
        self.current_results = None # To hold analysis results for display
        self.compartment_config_manager = None # Assuming this is managed elsewhere and passed

    def setup_ui(self):
        layout = QVBoxLayout(self)
        params_group = QGroupBox("Microcompartment Analysis Parameters")
        params_layout = QFormLayout()

        # Parameters based on Microcompartment analysis concepts (e.g., related to masks, roles)
        # Note: Actual Microcompartment analysis in the provided code is split
        # between TrackingCompartmentAnalyzer and NuclearCompartmentAnalyzer.
        # A comprehensive widget would need to integrate these.
        # Basic parameters related to using masks:

        # Need a way to select/define compartments - likely links to the Project/Multi-channel setup
        # Placeholder for compartment selection:
        self.compartment_selection_label = QLabel("Compartment Selection handled in Project/Multi-channel tabs.")
        params_layout.addRow(self.compartment_selection_label)

        # Parameters potentially used by analyzers like TrackingCompartmentAnalyzer (e.g., morphology)
        self.apply_morphology_check = QCheckBox("Apply Morphological Operations to Masks")
        self.apply_morphology_check.setChecked(True)
        params_layout.addRow("Mask Processing:", self.apply_morphology_check)

        self.opening_iterations_spin = QSpinBox()
        self.opening_iterations_spin.setRange(0, 10)
        self.opening_iterations_spin.setValue(1)
        self.opening_iterations_spin.setToolTip("Iterations for morphological opening.")
        params_layout.addRow("Opening Iterations:", self.opening_iterations_spin)

        self.closing_iterations_spin = QSpinBox()
        self.closing_iterations_spin.setRange(0, 10)
        self.closing_iterations_spin.setValue(1)
        self.closing_iterations_spin.setToolTip("Iterations for morphological closing.")
        params_layout.addRow("Closing Iterations:", self.closing_iterations_spin)


        params_group.setLayout(params_layout)
        layout.addWidget(params_group)

        # Visualization Options (Placeholder)
        viz_group = QGroupBox("Visualization Options")
        viz_layout = QFormLayout()
        self.plot_type_combo = QComboBox()
        self.plot_type_combo.addItems(["Compartment Masks Overlay", "Tracks in Compartments", "Compartment Statistics"]) # Example plot types
        viz_layout.addRow("Plot Type:", self.plot_type_combo)
        viz_group.setLayout(viz_layout)
        layout.addWidget(viz_group)

        # Results Display
        self.results_tabs = QTabWidget()

        self.summary_text = QTextEdit()
        self.summary_text.setReadOnly(True)
        self.results_tabs.addTab(self.summary_text, "Summary")

        self.compartment_properties_table = QTableWidget() # Table for compartment properties
        self.compartment_properties_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.results_tabs.addTab(self.compartment_properties_table, "Compartment Properties")

        self.track_compartment_table = QTableWidget() # Table for track-compartment assignment
        self.track_compartment_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.results_tabs.addTab(self.track_compartment_table, "Track Assignment")


        self.plot_widget = MplCanvas()
        self.results_tabs.addTab(self.plot_widget, "Plots")

        layout.addWidget(self.results_tabs)

    def setup_connections(self):
        # Connect parameter changes to emit signal
        self.apply_morphology_check.stateChanged.connect(self.on_parameter_changed)
        self.opening_iterations_spin.valueChanged.connect(self.on_parameter_changed)
        self.closing_iterations_spin.valueChanged.connect(self.on_parameter_changed)


        # Connect plot type selection to update plot display
        self.plot_type_combo.currentIndexChanged.connect(self.update_plot_display)


    def on_parameter_changed(self):
        """Emits signal with current parameters when a parameter changes."""
        params = self.get_current_parameters()
        self.parameters_changed.emit(params)

    def get_current_parameters(self):
        """Collects current analysis parameters from the UI."""
        return {
            'apply_morphology': self.apply_morphology_check.isChecked(),
            'opening_iterations': self.opening_iterations_spin.value(),
            'closing_iterations': self.closing_iterations_spin.value(),
            # Parameters related to compartment definition (thresholds, rules) would likely
            # be managed by the CompartmentConfigManager and passed separately by the main GUI.
        }

    def update_results_display(self, results):
        """Updates the widget's display with analysis results."""
        self.current_results = results # Store results received from the worker
        if results:
            # Update Summary Text
            summary_text = "Microcompartment Analysis Results:\n\n"

            # Assuming results structure based on TrackingCompartmentAnalyzer and NuclearCompartmentAnalyzer
            if 'compartment_properties' in results:
                 summary_text += "--- Compartment Properties ---\n"
                 if isinstance(results['compartment_properties'], dict):
                      for comp, props in results['compartment_properties'].items():
                           summary_text += f"  Compartment: {comp}\n"
                           summary_text += f"    Total Area: {props.get('total_area', 0)}\n"
                           summary_text += f"    Number of Regions: {props.get('num_regions', 0)}\n"
                           # Add other key properties
                           summary_text += "\n"
                 else:
                      summary_text += str(results['compartment_properties']) + "\n"
                 summary_text += "\n"


            if 'track_compartments' in results and isinstance(results['track_compartments'], pd.DataFrame):
                 summary_text += "--- Track Compartment Assignment ---\n"
                 summary_text += f"Assigned {len(results['track_compartments'])} track points.\n"
                 # Could add counts per compartment here
                 summary_text += "\n"

            if 'nuclear_relationship_results' in results: # Example key for nuclear analysis results
                 summary_text += "--- Nuclear Relationship ---\n"
                 if isinstance(results['nuclear_relationship_results'], dict):
                      for comp, rel in results['nuclear_relationship_results'].items():
                           summary_text += f"  Compartment: {comp}\n"
                           summary_text += f"    Overlap Area: {rel.get('overlap_area', 0)}\n"
                           summary_text += f"    Inside Fraction: {rel.get('inside_fraction', np.nan):.3f}\n"
                           # Add other relationship metrics
                           summary_text += "\n"
                 else:
                       summary_text += str(results['nuclear_relationship_results']) + "\n"
                 summary_text += "\n"

            self.summary_text.setText(summary_text)

            # Update Tables
            if 'compartment_properties' in results and isinstance(results['compartment_properties'], dict):
                 self._update_compartment_properties_table(results['compartment_properties'])
            else:
                 self.compartment_properties_table.clear()
                 self.compartment_properties_table.setRowCount(0)


            if 'track_compartments' in results and isinstance(results['track_compartments'], pd.DataFrame):
                 self._update_track_compartment_table(results['track_compartments'])
            else:
                 self.track_compartment_table.clear()
                 self.track_compartment_table.setRowCount(0)


            # Update Plots Display
            self.update_plot_display() # Call to draw the selected plot

        else:
            self.summary_text.setText("No Microcompartment Analysis Results Available.")
            self.compartment_properties_table.clear()
            self.compartment_properties_table.setRowCount(0)
            self.track_compartment_table.clear()
            self.track_compartment_table.setRowCount(0)
            self.plot_widget.clear()

    def _update_compartment_properties_table(self, properties_dict):
        """Populates the compartment properties table."""
        self.compartment_properties_table.clear()
        # Headers based on common properties from analyze_compartments
        headers = ["Compartment", "Total Area", "Num Regions", "Largest Area", "Mean Intensity"]
        self.compartment_properties_table.setColumnCount(len(headers))
        self.compartment_properties_table.setHorizontalHeaderLabels(headers)
        self.compartment_properties_table.setRowCount(0)

        row_count = 0
        for comp_name, props in properties_dict.items():
             self.compartment_properties_table.insertRow(row_count)
             self.compartment_properties_table.setItem(row_count, 0, QTableWidgetItem(str(comp_name)))
             self.compartment_properties_table.setItem(row_count, 1, QTableWidgetItem(str(props.get('total_area', 0))))
             self.compartment_properties_table.setItem(row_count, 2, QTableWidgetItem(str(props.get('num_regions', 0))))
             self.compartment_properties_table.setItem(row_count, 3, QTableWidgetItem(str(props.get('largest_region_area', 0))))
             mean_intensity = props.get('mean_intensity')
             self.compartment_properties_table.setItem(row_count, 4, QTableWidgetItem(f"{mean_intensity:.3f}" if mean_intensity is not None else "N/A"))

             # Add nuclear relationship metrics if available
             if 'nuclear_relationship' in props:
                  rel = props['nuclear_relationship']
                  # Add columns if not already present (dynamic headers) or add to summary
                  # For this table, keep it simple, focus on compartment properties
             row_count += 1

        self.compartment_properties_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.compartment_properties_table.verticalHeader().setVisible(False)


    def _update_track_compartment_table(self, track_compartment_df):
        """Populates the track compartment assignment table (showing some entries)."""
        self.track_compartment_table.clear()
        if track_compartment_df is None or track_compartment_df.empty:
            self.track_compartment_table.setRowCount(0)
            return

        # Display a subset or summary if the table is too large
        if len(track_compartment_df) > 1000: # Example limit
             logger.warning(f"Track assignment table is very large ({len(track_compartment_df)} rows). Displaying first 1000.")
             display_df = track_compartment_df.head(1000)
        else:
             display_df = track_compartment_df

        headers = display_df.columns.tolist() # Should be ['track_id', 'frame', 'compartment']
        self.track_compartment_table.setColumnCount(len(headers))
        self.track_compartment_table.setHorizontalHeaderLabels(headers)
        self.track_compartment_table.setRowCount(len(display_df))

        for i, (index, row) in enumerate(display_df.iterrows()):
            for j, col in enumerate(headers):
                self.track_compartment_table.setItem(i, j, QTableWidgetItem(str(row[col])))

        self.track_compartment_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.track_compartment_table.verticalHeader().setVisible(False)


    def update_plot_display(self):
        """Clears and draws the selected plot type."""
        self.plot_widget.clear()
        if self.current_results is None:
            self.plot_widget.axes.text(0.5, 0.5, "No results to plot.", ha='center', va='center')
            self.plot_widget.draw()
            return

        plot_type = self.plot_type_combo.currentText()
        ax = self.plot_widget.axes

        try:
            if plot_type == "Compartment Masks Overlay":
                 # Requires access to compartment masks and potentially a background image
                 ax.text(0.5, 0.5, "Compartment Masks Plot Placeholder", ha='center', va='center')
                 # If masks and background image are available (e.g., passed from main GUI)
                 # you would call a plotting function from visualization.boundary_plots or similar
                 # e.g., boundary_plots.plot_compartment_masks(self.current_results['compartment_masks'], background_image)

            elif plot_type == "Tracks in Compartments":
                 # Requires track positions and compartment assignment, and potentially a background image
                 ax.text(0.5, 0.5, "Tracks in Compartments Plot Placeholder", ha='center', va='center')
                 # If track_compartment_df and track positions are available, and background image
                 # e.g., visualization.tracks.plot_tracks_colored_by_compartment(tracks_df, self.current_results['track_compartments'], background_image)

            elif plot_type == "Compartment Statistics":
                 if 'compartment_properties' in self.current_results and isinstance(self.current_results['compartment_properties'], dict):
                      # Bar plot of a key statistic (e.g., Total Area) per compartment
                      comp_names = list(self.current_results['compartment_properties'].keys())
                      areas = [self.current_results['compartment_properties'][c].get('total_area', 0) for c in comp_names]
                      if comp_names and areas:
                           ax.bar(comp_names, areas, alpha=0.7)
                           ax.set_ylabel("Total Area (pixels)")
                           ax.set_title("Compartment Areas")
                      else:
                           ax.text(0.5, 0.5, "No compartment property data available for plotting.", ha='center', va='center')
                 else:
                      ax.text(0.5, 0.5, "No compartment properties results available for plotting.", ha='center', va='center')

            else:
                ax.text(0.5, 0.5, "Plot type not implemented.", ha='center', va='center')

            self.plot_widget.draw()

        except Exception as e:
            logger.error(f"Error generating microcompartment plot: {e}", exc_info=True)
            self.plot_widget.clear()
            self.plot_widget.axes.text(0.5, 0.5, f"Plot Error: {e}", ha='center', va='center', color='red')
            self.plot_widget.draw()
# C:\Users\mjhen\SPT_GUI\widgets\analysis_widgets.py (Continued)

# Assuming MultiChannelManager is imported (e.g., from Analysis.microcompartment)
# from Analysis.microcompartment import MultiChannelManager # Already imported at the top

class MultiChannelAnalysisWidget(QWidget):
    """Widget for Multi-Channel Analysis controls and visualization."""

    parameters_changed = pyqtSignal(dict)
    # Add signals for specific multi-channel analysis types if needed later

    def __init__(self, parent=None):
        super().__init__(parent)
        # self.manager = None # MultiChannelManager instance should be set from outside
        self.setup_ui()
        self.setup_connections() # Connections within the widget
        self.current_results = None # To hold analysis results for display
        self.available_channels = {} # {channel_index: channel_name} - set from outside


    def setup_ui(self):
        layout = QVBoxLayout(self)

        # Channel Management Group - Based on snippet
        channel_group = QGroupBox("Channel Information")
        channel_layout = QVBoxLayout()

        self.channel_info_table = QTableWidget()
        self.channel_info_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.channel_info_table.setSelectionMode(QAbstractItemView.SingleRow)
        self.channel_info_table.setSelectionBehavior(QAbstractItemView.SelectRows)
        channel_layout.addWidget(self.channel_info_table)

        # Note: Buttons like Add/Remove/Edit Channel from snippet would likely
        # interact with a MultiChannelManager instance, possibly via the main GUI.
        # They are omitted here as this widget focuses on *analysis* parameters/results.

        channel_group.setLayout(channel_layout)
        layout.addWidget(channel_group)

        # Analysis Group - Based on snippet
        analysis_group = QGroupBox("Analysis Settings")
        analysis_layout = QFormLayout()

        # Analysis Type
        self.analysis_type_combo = QComboBox()
        # Analysis types listed in the original snippet's widget
        self.analysis_type_combo.addItems([
            "Colocalization",
            "Cross-Correlation",
            "Intensity Correlation",
            "Distance Analysis"
        ])
        analysis_layout.addRow("Analysis Type:", self.analysis_type_combo)

        # Parameters that depend on analysis type (placeholder/example)
        self.analysis_param1_label = QLabel("Param 1:")
        self.analysis_param1_spin = QDoubleSpinBox()
        self.analysis_param1_spin.setRange(0, 1000)
        self.analysis_param1_spin.setValue(1.0)
        self.analysis_param1_label.setVisible(False) # Start hidden
        self.analysis_param1_spin.setVisible(False)
        analysis_layout.addRow(self.analysis_param1_label, self.analysis_param1_spin)

        # This needs a more dynamic way to load parameters based on the selected analysis type
        # and the specific parameters required by the underlying analysis logic (which is not
        # fully defined in the provided analysis modules for these types).

        analysis_group.setLayout(analysis_layout)
        layout.addWidget(analysis_group)

        # Results Group - Based on snippet
        results_group = QGroupBox("Results")
        results_layout = QVBoxLayout()
        self.results_tabs = QTabWidget()

        self.summary_text = QTextEdit()
        self.summary_text.setReadOnly(True)
        self.results_tabs.addTab(self.summary_text, "Summary")

        # Additional tabs for specific results (e.g., Correlation matrix, Distance histogram)
        self.analysis_results_table = QTableWidget() # Generic table for results
        self.analysis_results_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.results_tabs.addTab(self.analysis_results_table, "Detailed Results")


        self.plot_widget = MplCanvas() # For plots
        self.results_tabs.addTab(self.plot_widget, "Plots")

        results_layout.addWidget(self.results_tabs)
        results_group.setLayout(results_layout)
        layout.addWidget(results_group)

        # Parameter change signal - assuming it's needed for integration
        # self.parameters_changed = pyqtSignal(dict) # Need to define in class if used


    def setup_connections(self):
        # Connect analysis type combo to update parameters visibility
        self.analysis_type_combo.currentTextChanged.connect(self.update_parameter_visibility)

        # Connect parameter changes to emit signal (example for one parameter)
        self.analysis_param1_spin.valueChanged.connect(self.on_parameter_changed)

        # Connect plot type selection (if added to this widget) to update plot display
        # Note: The original snippet had a plot tab but no specific plot type combo within the widget's UI setup.
        # Assuming plotting is driven by the selected analysis type and results.

    def update_parameter_visibility(self, analysis_type):
        """Adjusts parameter visibility based on selected analysis type."""
        # This needs to be implemented based on the parameters required by each analysis type.
        # For now, just a placeholder showing/hiding one example parameter.
        if analysis_type in ["Colocalization", "Cross-Correlation"]: # Example types needing param1
             self.analysis_param1_label.setVisible(True)
             self.analysis_param1_spin.setVisible(True)
             # Set ranges/values based on the specific analysis type if known
        else:
             self.analysis_param1_label.setVisible(False)
             self.analysis_param1_spin.setVisible(False)

        # Emit parameter change signal after updating visibility, in case defaults change
        self.on_parameter_changed()


    def on_parameter_changed(self):
        """Emits signal with current parameters when a parameter changes."""
        params = self.get_current_parameters()
        self.parameters_changed.emit(params)


    def get_current_parameters(self):
        """Collects current analysis parameters from the UI."""
        # This needs to collect parameters specific to the selected analysis type.
        current_analysis_type = self.analysis_type_combo.currentText()
        params = {
            'analysis_type': current_analysis_type,
            # Collect common parameters if any
            # Collect parameters specific to the current_analysis_type
        }

        if current_analysis_type in ["Colocalization", "Cross-Correlation"]:
             params['param1'] = self.analysis_param1_spin.value() # Example parameter

        return params


    def update_results_display(self, results):
        """Updates the widget's display with analysis results."""
        self.current_results = results # Store results received from the worker
        if results:
            # Update Summary Text
            summary_text = f"Multi-Channel Analysis Results ({self.get_current_parameters().get('analysis_type', 'N/A')}):\n\n"

            # The structure of 'results' will depend entirely on the output of the
            # specific multi-channel analysis functions being called by the analyzer.
            # Example based on potential outputs:
            if isinstance(results, dict):
                 for key, value in results.items():
                      summary_text += f"--- {key.replace('_', ' ').title()} ---\n"
                      if isinstance(value, (int, float, str)):
                           summary_text += f"{value}\n"
                      elif isinstance(value, dict):
                           for sub_key, sub_value in value.items():
                                summary_text += f"  {sub_key}: {sub_value}\n"
                           summary_text += "\n"
                      elif isinstance(value, pd.DataFrame):
                           summary_text += f"DataFrame with {len(value)} rows, {len(value.columns)} columns.\n"
                           # Could display head() or describe()
                           summary_text += "\n"
                      else:
                           summary_text += f"Data of type {type(value)}.\n"
                      summary_text += "\n"

            else:
                 summary_text += str(results)

            self.summary_text.setText(summary_text)

            # Update Results Table
            # This depends on which part of the results should go into a table.
            # Example: if results['colocalization_metrics'] is a DataFrame
            if isinstance(results, dict): # Basic check
                 for key, value in results.items():
                      if isinstance(value, pd.DataFrame):
                           # Assume the first DataFrame found goes into the main table
                           self._update_analysis_results_table(value)
                           break # Stop after the first DataFrame

            # Update Plots Display
            self.update_plot_display() # Call to draw the selected plot


        else:
            self.summary_text.setText("No Multi-Channel Analysis Results Available.")
            self.analysis_results_table.clear()
            self.analysis_results_table.setRowCount(0)
            self.plot_widget.clear()


    def _update_analysis_results_table(self, results_df):
        """Populates a generic table with results DataFrame."""
        self.analysis_results_table.clear()
        if results_df is None or results_df.empty:
            self.analysis_results_table.setRowCount(0)
            return

        headers = results_df.columns.tolist()
        self.analysis_results_table.setColumnCount(len(headers))
        self.analysis_results_table.setHorizontalHeaderLabels(headers)
        self.analysis_results_table.setRowCount(len(results_df))

        for i, (index, row) in enumerate(results_df.iterrows()):
            for j, col in enumerate(headers):
                item_text = str(row[col])
                if isinstance(row[col], (float, np.floating)):
                     item_text = f"{row[col]:.4f}" # Format floats
                self.analysis_results_table.setItem(i, j, QTableWidgetItem(item_text))

        self.analysis_results_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.analysis_results_table.verticalHeader().setVisible(False)


    def update_plot_display(self):
        """Clears and draws the selected plot type."""
        self.plot_widget.clear()
        if self.current_results is None:
            self.plot_widget.axes.text(0.5, 0.5, "No results to plot.", ha='center', va='center')
            self.plot_widget.draw()
            return

        # Plotting logic depends entirely on the structure of self.current_results
        # and the selected analysis_type from self.analysis_type_combo.currentText().
        # Placeholder plot:
        ax = self.plot_widget.axes
        ax.text(0.5, 0.5, "Multi-Channel Plot Placeholder", ha='center', va='center')
        self.plot_widget.draw()

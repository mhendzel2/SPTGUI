"""
SPT Analyzer - Main GUI Application

This file implements a PyQt5-based graphical user interface for the SPT Analysis framework,
providing tools for single particle tracking and analysis.
"""

# Standard library imports
import os
import sys
import logging
import traceback
import MplCanvas
from typing import Dict, List, Tuple
from dataclasses import dataclass

# Third-party imports
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.animation import FuncAnimation
import matplotlib.colors as mcolors
from matplotlib.gridspec import GridSpec
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from PyQt5.QtWidgets import (
    # ... existing Qt imports ...
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QSettings, QSize
from PyQt5.QtGui import QIcon, QPixmap, QFont

# Local imports
from analysis.diffusion import DiffusionAnalyzer
from analysis.diffusion import DiffusionAnalyzer
from analysis.active_transport import ActiveTransportAnalyzer
from analysis.boundary_crossing import BoundaryCrossingAnalyzer
from analysis.dwell_time import DwellTimeAnalyzer
from analysis.crowding import CrowdingAnalyzer
from analysis.diffusion_population import DiffusionPopulationAnalyzer
from analysis.gel_structure import GelStructureAnalyzer
from analysis.microcompartment import MicrocompartmentAnalyzer

from visualization.analysis_plots import AnalysisPlotter
from visualization.tracks import plot_tracks
from visualization.diffusion import plot_diffusion_map

from utils.io import load_image_stack, save_tracks, load_tracks
from utils.processing import enhance_contrast, denoise_image
from utils.logging_config import setup_logging
from management import SPTProject, TreatmentGroup, Cell # Import project management classes
from .management_widget import ProjectManagerWidget # Assuming you have a ProjectManagerWidget defined in management_widget.py or similar
from widgets.analysis_widgets import (
    DwellTimeWidget, CrowdingWidget, DiffusionPopulationWidget,
    GelStructureWidget, MicrocompartmentWidget, MultiChannelAnalysisWidget,
    MplCanvas
from widgets.analysis_widgets import (
    MplCanvas, # Assuming you will use this MplCanvas globally or in widgets
    DiffusionAnalysisWidget,
    ActiveTransportWidget,
    BoundaryCrossingWidget,
    DwellTimeWidget,
    CrowdingWidget,
    DiffusionPopulationWidget,
    GelStructureWidget,
    MicrocompartmentWidget,
    MultiChannelAnalysisWidget,

# Set up logging
logger = logging.getLogger(__name__)
setup_logging(log_level=logging.INFO)

def update_analysis_parameters(self, index):
    """Update the analysis parameters based on the selected analysis type."""
    # Clear existing parameters
    while self.params_layout.count():
        item = self.params_layout.takeAt(0)
        if item.widget():
            item.widget().deleteLater()

    analysis_type = self.analysis_type.currentText()

    if analysis_type == "Diffusion Analysis":
        # Create diffusion analysis widget
        self.diffusion_widget = DiffusionAnalysisWidget()
        self.params_layout.addWidget(self.diffusion_widget)

        # Connect signals
        self.diffusion_widget.analyze_btn.clicked.connect(self.run_diffusion_analysis)
        self.diffusion_widget.export_btn.clicked.connect(self.export_diffusion_results)
        self.diffusion_widget.plot_type.currentIndexChanged.connect(self.update_diffusion_plot)
    analysis_type = self.analysis_type.currentText().lower()
    
    # Clear existing parameters
    self.clear_parameter_layout()
    
    if analysis_type == "diffusion analysis":
        self.setup_diffusion_parameters()
    elif analysis_type == "active transport":
        self.setup_active_transport_parameters()
    elif analysis_type == "boundary crossing":
        self.setup_boundary_crossing_parameters()
    elif analysis_type == "dwell time analysis":
        self.setup_dwell_time_parameters()
    elif analysis_type == "diffusion population":
        self.setup_diffusion_population_parameters()
    elif analysis_type == "gel structure":
        self.setup_gel_structure_parameters()
    elif analysis_type == "microcompartment analysis":
        self.setup_microcompartment_parameters()
 def run_analysis(self):
    """Run selected analysis on tracks"""
    if self.tracks_df is None:
        QMessageBox.warning(self, "Warning", "No tracks available")
        return

    try:
        # Get analysis parameters
        analysis_type = self.analysis_type.currentText().lower()
        analysis_params = self.get_analysis_parameters()

        # Show progress bar
        self.progress_bar.setVisible(True)
        self.setEnabled(False)

        # Create worker thread
        self.worker = WorkerThread(
            "analyze",
            {
                "analysis_type": analysis_type,
                "tracks_df": self.tracks_df,
                "analysis_params": analysis_params,
                "analysis_manager": self.analysis_manager
            }
        )

        # Connect signals
        self.worker.progress_updated.connect(self.progress_bar.setValue)
        self.worker.operation_completed.connect(self.analysis_completed)
        self.worker.error_occurred.connect(self.analysis_error)

        # Start worker
        self.worker.start()
def run_analysis(self):
    """Run selected analysis on tracks."""
    if self.tracks_df is None:
        QMessageBox.warning(self, "Warning", "No tracks available")
        return

    try:
        # Get the currently selected analysis type and its parameters widget
        analysis_type = self.analysis_type.currentText().lower()
        current_widget = self.params_stack.currentWidget() # Get the currently visible widget

        if not hasattr(current_widget, 'get_current_parameters'):
             raise AttributeError(f"Current widget for '{analysis_type}' does not have a 'get_current_parameters' method.")

        analysis_params = current_widget.get_current_parameters()

        # --- Add project/global settings to analysis_params ---
        # Many analyzers need pixel size, frame interval, particle radius, etc.
        # Pass these from the main GUI's project settings.
        analysis_params['pixel_size'] = self.project_settings.get('pixel_size', 1.0)
        analysis_params['frame_interval'] = self.project_settings.get('frame_interval', 0.014)
        analysis_params['particle_radius'] = self.project_settings.get('particle_radius', 5.0) # in nm, analyzer might need um
        analysis_params['temperature'] = self.project_settings.get('temperature', 25.0) # in C

        # Add compartment masks if available and needed by the analyzer
        if hasattr(self, 'compartment_masks') and self.compartment_masks is not None:
             # Need to check if the specific analyzer supports/needs compartment_masks
             # For simplicity here, just pass if available. Analyzer should handle.
             analysis_params['compartment_masks'] = self.compartment_masks


        # Show progress bar
        self.progress_bar.setVisible(True)
        self.setEnabled(False)

        # Create worker thread
        self.worker = WorkerThread(
            "analyze",
            {
                "analysis_type": analysis_type,
                "tracks_df": self.tracks_df,
                "analysis_params": analysis_params,
                "analysis_manager": self.analysis_manager
            }
        )

        # Connect signals
        self.worker.progress_updated.connect(self.progress_bar.setValue)
        self.worker.operation_completed.connect(self.analysis_completed)
        self.worker.error_occurred.connect(self.analysis_error)

        # Start worker
        self.worker.start()

    except Exception as e:
        self.handle_analysis_error(e) # Ensure handle_analysis_error exists or use a simple message box

# Ensure you have a method like handle_analysis_error or just use QMessageBox
def handle_analysis_error(self, error_msg):
    logger.error(f"Analysis error: {error_msg}")
    QMessageBox.critical(self, "Analysis Error", f"Analysis failed: {error_msg}")
    self.progress_bar.setVisible(False)
    self.setEnabled(True)
    except Exception as e:
        self.handle_analysis_error(e)    
 def display_analysis_results(self, results, analysis_type):
    """Display analysis results based on type"""
    if analysis_type == "diffusion":
        self.display_diffusion_results(results)
    elif analysis_type == "active_transport":
        self.display_active_transport_results(results)
    elif analysis_type == "boundary_crossing":
        self.display_boundary_crossing_results(results)
    elif analysis_type == "dwell_time":
        self.display_dwell_time_results(results)
    elif analysis_type == "diffusion_population":
        self.display_diffusion_population_results(results)
    elif analysis_type == "gel_structure":
        self.display_gel_structure_results(results)
    elif analysis_type == "microcompartment":
        self.display_microcompartment_results(results)          
class MplCanvas(FigureCanvas):
    """Canvas for matplotlib figures in the GUI."""
    
    def __init__(self, width=5, height=4, dpi=100):
        self.fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = self.fig.add_subplot(111)
        super(MplCanvas, self).__init__(self.fig)
        self.fig.tight_layout()

class DiffusionAnalysisWidget(QWidget):
    """Widget for diffusion analysis controls and visualization."""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setup_ui()
        
    def setup_ui(self):
        layout = QVBoxLayout(self)
        
        # Parameters group
        params_group = QGroupBox("Diffusion Analysis Parameters")
        params_layout = QFormLayout()
        
        # MSD calculation parameters
        self.max_lag_spinbox = QSpinBox()
        self.max_lag_spinbox.setRange(5, 100)
        self.max_lag_spinbox.setValue(20)
        self.max_lag_spinbox.setToolTip("Maximum time lag for MSD calculation")
        params_layout.addRow("Max Lag Time:", self.max_lag_spinbox)
        
        self.min_track_length = QSpinBox()
        self.min_track_length.setRange(5, 100)
        self.min_track_length.setValue(10)
        self.min_track_length.setToolTip("Minimum track length to analyze")
        params_layout.addRow("Min Track Length:", self.min_track_length)
        
        # Model selection
        self.model_selection = QComboBox()
        self.model_selection.addItems([
            "All Models",
            "Simple Diffusion",
            "Anomalous Diffusion",
            "Confined Diffusion",
            "Directed Motion"
        ])
        params_layout.addRow("Diffusion Model:", self.model_selection)
        
        # Fitting parameters
        self.max_fit_points = QSpinBox()
        self.max_fit_points.setRange(3, 50)
        self.max_fit_points.setValue(10)
        self.max_fit_points.setToolTip("Maximum number of points to use for model fitting")
        params_layout.addRow("Max Fit Points:", self.max_fit_points)
        
        params_group.setLayout(params_layout)
        layout.addWidget(params_group)
        
        # Results visualization options
        viz_group = QGroupBox("Visualization Options")
        viz_layout = QVBoxLayout()
        
        # Plot type selection
        self.plot_type = QComboBox()
        self.plot_type.addItems([
            "MSD Curves",
            "Diffusion Coefficient Map",
            "Alpha Distribution",
            "Model Comparison"
        ])
        viz_layout.addWidget(QLabel("Plot Type:"))
        viz_layout.addWidget(self.plot_type)
        
        # Display options
        self.show_individual_tracks = QCheckBox("Show Individual Tracks")
        self.show_individual_tracks.setChecked(True)
        viz_layout.addWidget(self.show_individual_tracks)
        
        self.show_ensemble_average = QCheckBox("Show Ensemble Average")
        self.show_ensemble_average.setChecked(True)
        viz_layout.addWidget(self.show_ensemble_average)
        
        self.show_model_fits = QCheckBox("Show Model Fits")
        self.show_model_fits.setChecked(True)
        viz_layout.addWidget(self.show_model_fits)
        
        viz_group.setLayout(viz_layout)
        layout.addWidget(viz_group)
        
        # Analysis buttons
        button_layout = QHBoxLayout()
        
        self.analyze_btn = QPushButton("Analyze Diffusion")
        button_layout.addWidget(self.analyze_btn)
        
        self.export_btn = QPushButton("Export Results")
        button_layout.addWidget(self.export_btn)
        
        layout.addLayout(button_layout)
        
        # Results display
        results_group = QGroupBox("Analysis Results")
        results_layout = QVBoxLayout()
        
        self.results_text = QTextEdit()
        self.results_text.setReadOnly(True)
        results_layout.addWidget(self.results_text)
        
        results_group.setLayout(results_layout)
        layout.addWidget(results_group)
class WorkerThread(QThread):
    """Worker thread for CPU-intensive operations."""
    
    progress_updated = pyqtSignal(int)
    operation_completed = pyqtSignal(object)
    error_occurred = pyqtSignal(str)
    
    def __init__(self, operation_type, params=None):
        super().__init__()
        self.operation_type = operation_type
        self.params = params if params else {}
        
    def run(self):
        try:
            result = None
            if self.operation_type == "detect_particles":
                # Perform particle detection
                detector = ParticleDetector(**self.params.get("detector_params", {}))
                frames = self.params.get("frames")
                
                total_frames = len(frames)
                results = []
                
                for i, frame in enumerate(frames):
                    particles = detector.detect(frame)
                    results.append(particles)
                    progress = int((i + 1) / total_frames * 100)
                    self.progress_updated.emit(progress)
                
                result = results
                
            elif self.operation_type == "track_particles":
                # Perform track linking
                detections = self.params.get("detections")
                linker_method = self.params.get("linker_method", "hungarian")
                linker_params = self.params.get("linker_params", {})
                
                linker = get_linker(linker_method, **linker_params)
                tracks = []
                
                total_frames = len(detections) - 1
                for i in range(total_frames):
                    # Link frames i and i+1
                    linked = linker.link_detections(detections[i], detections[i+1])
                    tracks.append(linked)
                    progress = int((i + 1) / total_frames * 100)
                    self.progress_updated.emit(progress)
                
                result = tracks
                
            elif self.operation_type == "analyze_diffusion":
                # Perform diffusion analysis
                tracks_df = self.params.get("tracks_df")
                pixel_size = self.params.get("pixel_size", 0.1)  # μm per pixel
                dt = self.params.get("frame_interval", 0.1)  # seconds per frame
                
                # Create analyzer and compute diffusion coefficients
                from spt_analyzer.analysis.diffusion_models import compute_msd, fit_diffusion_models
                
                track_ids = tracks_df["track_id"].unique()
                total_tracks = len(track_ids)
                results = {}
                
                for i, track_id in enumerate(track_ids):
                    track = tracks_df[tracks_df["track_id"] == track_id]
                    if len(track) >= 5:  # Minimum track length
                        msd = compute_msd(track, pixel_size, dt)
                        model_fits = fit_diffusion_models(msd, dt)
                        results[track_id] = model_fits
                    
                    progress = int((i + 1) / total_tracks * 100)
                    self.progress_updated.emit(progress)
                
                result = results
                
            # Add more operation types as needed
            else:
                raise ValueError(f"Unknown operation type: {self.operation_type}")
            
            self.operation_completed.emit(result)
            
        except Exception as e:
            logger.error(f"Error in worker thread: {str(e)}", exc_info=True)
            self.error_occurred.emit(str(e))
def run(self):
        try:
            if self.operation_type == "detect_particles":
                frames = self.params.get("frames")
                analysis_manager = self.params.get("analysis_manager")

                results = []
                total_frames = len(frames)

                for i, frame in enumerate(frames):
                    particles = analysis_manager.detect_particles(frame)
                    results.append(particles)
                    progress = int((i + 1) / total_frames * 100)
                    self.progress_updated.emit(progress)

                self.operation_completed.emit(results)

            elif self.operation_type == "track_particles":
                detections = self.params.get("detections")
                analysis_manager = self.params.get("analysis_manager")

                tracks = analysis_manager.track_particles(detections)
                self.operation_completed.emit(tracks)

            else:
                raise ValueError(f"Unsupported operation type: {self.operation_type}")

        except Exception as e:
            self.error_occurred.emit(str(e))

class ProjectSettingsDialog(QDialog):
    """Dialog for configuring project settings."""
    
    def __init__(self, parent=None, current_settings=None):
        super().__init__(parent)
        self.setWindowTitle("Project Settings")
        self.setMinimumWidth(400)
        
        settings = current_settings or {}
        
        # Create layout
        layout = QVBoxLayout()
        form_layout = QFormLayout()
        
        # Project name
        self.project_name = QLineEdit(settings.get("project_name", ""))
        form_layout.addRow("Project Name:", self.project_name)
        
        # Pixel size
        self.pixel_size = QDoubleSpinBox()
        self.pixel_size.setRange(0.001, 10.0)
        self.pixel_size.setSingleStep(0.01)
        self.pixel_size.setValue(settings.get("pixel_size", 0.1))
        self.pixel_size.setSuffix(" μm")
        form_layout.addRow("Pixel Size:", self.pixel_size)
        
        # Frame interval
        self.frame_interval = QDoubleSpinBox()
        self.frame_interval.setRange(0.001, 60.0)
        self.frame_interval.setSingleStep(0.01)
        self.frame_interval.setValue(settings.get("frame_interval", 0.1))
        self.frame_interval.setSuffix(" s")
        form_layout.addRow("Frame Interval:", self.frame_interval)
        
        # Temperature
        self.temperature = QDoubleSpinBox()
        self.temperature.setRange(0.1, 50.0)
        self.temperature.setSingleStep(0.1)
        self.temperature.setValue(settings.get("temperature", 25.0))
        self.temperature.setSuffix(" °C")
        form_layout.addRow("Temperature:", self.temperature)
        
        # Particles type/radius
        self.particle_radius = QDoubleSpinBox()
        self.particle_radius.setRange(0.1, 1000.0)
        self.particle_radius.setSingleStep(0.1)
        self.particle_radius.setValue(settings.get("particle_radius", 5.0))
        self.particle_radius.setSuffix(" nm")
        form_layout.addRow("Particle Radius:", self.particle_radius)
        
        layout.addLayout(form_layout)
        
        # Add notes field
        notes_group = QGroupBox("Project Notes")
        notes_layout = QVBoxLayout()
        self.notes = QTextEdit(settings.get("notes", ""))
        notes_layout.addWidget(self.notes)
        notes_group.setLayout(notes_layout)
        layout.addWidget(notes_group)
        
        # Add buttons
        buttons = QDialogButtonBox(
            QDialogButtonBox.Ok | QDialogButtonBox.Cancel,
            Qt.Horizontal, self)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)
        
        self.setLayout(layout)
    
    def get_settings(self):
        """Return the current settings as a dictionary."""
        return {
            "project_name": self.project_name.text(),
            "pixel_size": self.pixel_size.value(),
            "frame_interval": self.frame_interval.value(),
            "temperature": self.temperature.value(),
            "particle_radius": self.particle_radius.value(),
            "notes": self.notes.toPlainText()
        }
class WorkerThread(QThread):
    """Worker thread for CPU-intensive operations in SPT analysis.
    
    This thread handles particle detection, track linking, and various analyses
    while keeping the GUI responsive. Operations are performed through an
    AnalysisManager instance.
    
    Signals:
        progress_updated (int): Emits progress percentage (0-100)
        operation_completed (object): Emits results when operation completes
        error_occurred (str): Emits error message if operation fails
    """
    
    progress_updated = pyqtSignal(int)
    operation_completed = pyqtSignal(object)
    error_occurred = pyqtSignal(str)

    def __init__(self, operation_type, params=None):
        """Initialize the worker thread.
        
        Args:
            operation_type (str): Type of operation to perform
                ('detect_particles', 'track_particles', or 'analyze')
            params (dict, optional): Parameters for the operation. Must include
                'analysis_manager' and operation-specific parameters.
        """
        super().__init__()
        self.operation_type = operation_type
        self.params = params if params else {}

    def run(self):
        """Execute the specified operation.
        
        The operation type determines which analysis is performed:
        - 'detect_particles': Detect particles in image frames
        - 'track_particles': Link particle detections into tracks
        - 'analyze': Perform specified analysis on tracks
        
        All operations require an analysis_manager in params.
        """
        try:
            # Validate analysis manager
            analysis_manager = self.params.get("analysis_manager")
            if not analysis_manager:
                raise ValueError("Analysis manager not provided")

            # Execute requested operation
            if self.operation_type == "detect_particles":
                self._run_detection(analysis_manager)
            elif self.operation_type == "track_particles":
                self._run_tracking(analysis_manager)
            elif self.operation_type == "analyze":
                self._run_analysis(analysis_manager)
            else:
                raise ValueError(f"Unknown operation type: {self.operation_type}")

        except Exception as e:
            self.error_occurred.emit(str(e))

    def _run_detection(self, analysis_manager):
        """Run particle detection on image frames."""
        frames = self.params.get("frames")
        if not frames:
            raise ValueError("No frames provided for detection")

        results = []
        total_frames = len(frames)

        for i, frame in enumerate(frames):
            # Detect particles in current frame
            particles = analysis_manager.detect_particles(frame)
            results.append(particles)
            
            # Update progress
            progress = int((i + 1) / total_frames * 100)
            self.progress_updated.emit(progress)

        self.operation_completed.emit(results)

    def _run_tracking(self, analysis_manager):
        """Run track linking on particle detections."""
        detections = self.params.get("detections")
        if not detections:
            raise ValueError("No detections provided for tracking")

        tracks = analysis_manager.track_particles(detections)
        self.operation_completed.emit(tracks)

    def _run_analysis(self, analysis_manager):
        """Run specified analysis on tracks."""
        # Get analysis parameters
        analysis_type = self.params.get("analysis_type")
        tracks_df = self.params.get("tracks_df")
        analysis_params = self.params.get("analysis_params")

        # Validate parameters
        if not all([analysis_type, tracks_df is not None, analysis_params]):
            raise ValueError("Missing required parameters for analysis")

        # Run analysis
        results = analysis_manager.run_analysis(
            analysis_type,
            tracks_df,
            analysis_params
        )
        self.operation_completed.emit(results)
    progress_updated = pyqtSignal(int)
    operation_completed = pyqtSignal(object)
    error_occurred = pyqtSignal(str)

    def __init__(self, operation_type, params=None):
        super().__init__()
        self.operation_type = operation_type
        self.params = params if params else {}
        # Initialize analysis manager
        self.analysis_manager = AnalysisManager()
    def run(self):
        try:
            analysis_manager = self.params.get("analysis_manager")
            if not analysis_manager:
                raise ValueError("Analysis manager not provided")

            if self.operation_type == "detect_particles":
                frames = self.params.get("frames")
                results = []
                total_frames = len(frames)
                
                for i, frame in enumerate(frames):
                    particles = analysis_manager.detect_particles(frame)
                    results.append(particles)
                    progress = int((i + 1) / total_frames * 100)
                    self.progress_updated.emit(progress)
                
                self.operation_completed.emit(results)

            elif self.operation_type == "track_particles":
                detections = self.params.get("detections")
                tracks = analysis_manager.track_particles(detections)
                self.operation_completed.emit(tracks)

            elif self.operation_type == "analyze":
                analysis_type = self.params.get("analysis_type")
                tracks_df = self.params.get("tracks_df")
                analysis_params = self.params.get("analysis_params")
                
                results = analysis_manager.run_analysis(
                    analysis_type, 
                    tracks_df, 
                    analysis_params
                )
                self.operation_completed.emit(results)

        except Exception as e:
            self.error_occurred.emit(str(e))
    """Worker thread for CPU-intensive operations."""

    progress_updated = pyqtSignal(int)
    operation_completed = pyqtSignal(object)
    error_occurred = pyqtSignal(str)

    def __init__(self, operation_type, params=None):
        super().__init__()
        self.operation_type = operation_type
        self.params = params if params else {}

    def run(self):
        try:
            if self.operation_type == "detect_particles":
                frames = self.params.get("frames")
                analysis_manager = self.params.get("analysis_manager")

                results = []
                total_frames = len(frames)

                for i, frame in enumerate(frames):
                    particles = analysis_manager.detect_particles(frame)
                    results.append(particles)
                    progress = int((i + 1) / total_frames * 100)
                    self.progress_updated.emit(progress)

                self.operation_completed.emit(results)

            elif self.operation_type == "track_particles":
                detections = self.params.get("detections")
                analysis_manager = self.params.get("analysis_manager")

                tracks = analysis_manager.track_particles(detections)
                self.operation_completed.emit(tracks)

            elif self.operation_type == "analyze":
                analysis_type = self.params.get("analysis_type")
                tracks_df = self.params.get("tracks_df")
                analysis_params = self.params.get("analysis_params")
                analysis_manager = self.params.get("analysis_manager")

                results = analysis_manager.run_analysis(
                    analysis_type, tracks_df, analysis_params
                )
                self.operation_completed.emit(results)

        except Exception as e:
            self.error_occurred.emit(str(e))
@dataclass
class DiffusionResults:
    D: float  # Diffusion coefficient
    alpha: float  # Anomalous exponent
    r_squared: float  # Fit quality
    msd_curve: np.ndarray  # MSD curve data
    lag_times: np.ndarray  # Lag times
    model_fit: np.ndarray  # Fitted curve
    def run_analysis(self):
    """Run selected analysis on tracks."""
    if self.tracks_df is None:
        QMessageBox.warning(self, "Warning", "No tracks available")
        return

    try:
        # Get analysis parameters from GUI
        analysis_type = self.analysis_type.currentText().lower()
        analysis_params = self.get_analysis_parameters()

        # Show progress bar
        self.progress_bar.setVisible(True)
        self.setEnabled(False)

        # Create worker thread
        self.worker = WorkerThread(
            "analyze",
            {
                "analysis_type": analysis_type,
                "tracks_df": self.tracks_df,
                "analysis_params": analysis_params,
                "analysis_manager": self.analysis_manager
            }
        )

        # Connect signals
        self.worker.progress_updated.connect(self.progress_bar.setValue)
        self.worker.operation_completed.connect(self.analysis_completed)
        self.worker.error_occurred.connect(self.analysis_error)

        # Start worker
        self.worker.start()

    except Exception as e:
        logger.error(f"Analysis error: {str(e)}", exc_info=True)
        QMessageBox.critical(self, "Error", f"Analysis failed: {str(e)}")
def run_analysis(self):
    """Run selected analysis on tracks."""
    if self.tracks_df is None:
        QMessageBox.warning(self, "Warning", "No tracks available")
        return

    try:
        # Get analysis parameters from GUI
        analysis_type = self.analysis_type.currentText().lower()
        analysis_params = self.get_analysis_parameters()

        # Show progress bar
        self.progress_bar.setVisible(True)
        self.setEnabled(False)

        # Create worker thread
        self.worker = WorkerThread(
            "analyze",
            {
                "analysis_type": analysis_type,
                "tracks_df": self.tracks_df,
                "analysis_params": analysis_params,
                "analysis_manager": self.analysis_manager
            }
        )

        # Connect signals
        self.worker.progress_updated.connect(self.progress_bar.setValue)
        self.worker.operation_completed.connect(self.analysis_completed)
        self.worker.error_occurred.connect(self.analysis_error)

        # Start worker
        self.worker.start()

    except Exception as e:
        logger.error(f"Analysis error: {str(e)}", exc_info=True)
        QMessageBox.critical(self, "Error", f"Analysis failed: {str(e)}")
def analysis_completed(self, results):
    """Handle completion of analysis operation."""
    # Get the analysis type that just completed
    analysis_type = self.analysis_type.currentText().lower() # Assumes the dropdown selection hasn't changed

    # Store results
    self.analysis_results[analysis_type] = results

    # Find the corresponding widget and update its display
    analysis_widget = self.analysis_widgets.get(analysis_type)
    if analysis_widget and hasattr(analysis_widget, 'update_results_display'):
         analysis_widget.update_results_display(results)
         logger.debug(f"Updated display for: {analysis_type}")
    else:
         logger.warning(f"Could not update display for analysis type: {analysis_type}")
         # Fallback to a generic display if the specific widget is missing or incomplete
         self.display_analysis_results_generic(results, analysis_type)


    # Update data summary
    self.update_data_summary()

    # Reset UI
    self.progress_bar.setVisible(False)
    self.setEnabled(True)
    self.statusBar().showMessage(f"{analysis_type.title()} analysis completed", 3000)

# You might need a generic fallback display method
def display_analysis_results_generic(self, results, analysis_type):
     self.results_summary.setText(f"Results for {analysis_type.title()}:\n\n" + str(results))
     self.results_table.clear() # Clear generic table
     self.results_table.setRowCount(0)
     self.results_canvas.clear() # Clear generic plot canvas
     self.results_canvas.axes.text(0.5, 0.5, f"Generic Display for {analysis_type.title()}", ha='center', va='center')
     self.results_canvas.draw()

def analysis_error(self, error_msg):
    """Handle analysis error"""
    logger.error(f"Analysis error: {error_msg}")
    QMessageBox.critical(self, "Error", f"Analysis failed: {error_msg}")

    # Reset UI
    self.progress_bar.setVisible(False)
    self.setEnabled(True)
class DiffusionAnalyzer:
    """Analyzes particle diffusion characteristics"""

    def __init__(self, pixel_size: float = 0.1, frame_interval: float = 0.1,
                 max_lag: int = 20, min_track_length: int = 5):
        self.pixel_size = pixel_size  # μm per pixel
        self.frame_interval = frame_interval  # seconds
        self.max_lag = max_lag
        self.min_track_length = min_track_length

    def compute_msd(self, track: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Compute Mean Square Displacement for a single track"""
        positions = track[['x', 'y']].values * self.pixel_size
        n_frames = len(positions)

        max_lag = min(self.max_lag, n_frames - 1)
        msd = np.zeros(max_lag)
        lag_times = np.arange(1, max_lag + 1) * self.frame_interval

        for lag in range(1, max_lag + 1):
            # Calculate displacements for this lag time
            disp = positions[lag:] - positions[:-lag]
            # Calculate squared distances
            sq_disp = np.sum(disp**2, axis=1)
            # Average over all time points
            msd[lag-1] = np.mean(sq_disp)

        return msd, lag_times

    def fit_diffusion_model(self, msd: np.ndarray, lag_times: np.ndarray) -> Dict:
        """Fit diffusion models to MSD curve"""
        # Take log of both arrays for linear fitting
        log_times = np.log(lag_times)
        log_msd = np.log(msd)

        # Linear fit in log-log space
        slope, intercept, r_value, _, _ = stats.linregress(log_times, log_msd)

        # Calculate parameters
        alpha = slope  # Anomalous exponent
        D = np.exp(intercept) / (2 * 2)  # Diffusion coefficient (2D)
        r_squared = r_value**2

        # Generate model fit curve
        model_fit = np.exp(intercept + slope * log_times)

        return {
            'D': D,
            'alpha': alpha,
            'r_squared': r_squared,
            'model_fit': model_fit
        }

    def analyze_track(self, track: pd.DataFrame) -> DiffusionResults:
        """Analyze diffusion characteristics of a single track"""
        # Compute MSD
        msd, lag_times = self.compute_msd(track)

        # Fit diffusion model
        fit_results = self.fit_diffusion_model(msd, lag_times)

        return DiffusionResults(
            D=fit_results['D'],
            alpha=fit_results['alpha'],
            r_squared=fit_results['r_squared'],
            msd_curve=msd,
            lag_times=lag_times,
            model_fit=fit_results['model_fit']
        )

    def analyze(self, tracks_df: pd.DataFrame) -> Dict:
        """Analyze all tracks in the dataset"""
        results = []
        track_ids = tracks_df['track_id'].unique()

        for track_id in track_ids:
            track = tracks_df[tracks_df['track_id'] == track_id]

            if len(track) >= self.min_track_length:
                track_results = self.analyze_track(track)
                results.append({
                    'track_id': track_id,
                    'D': track_results.D,
                    'alpha': track_results.alpha,
                    'r_squared': track_results.r_squared,
                    'track_length': len(track),
                    'msd_curve': track_results.msd_curve,
                    'lag_times': track_results.lag_times,
                    'model_fit': track_results.model_fit
                })

        return {
            'results_df': pd.DataFrame(results),
            'analysis_params': {
                'pixel_size': self.pixel_size,
                'frame_interval': self.frame_interval,
                'max_lag': self.max_lag,
                'min_track_length': self.min_track_length
            }
        }
@dataclass
class TransportResults:
    is_active: bool
    velocity: float
    direction: float
    run_length: float
    duration: float
    start_frame: int
    end_frame: int

class ActiveTransportAnalyzer:
    """Analyzes active transport characteristics"""

    def __init__(self, pixel_size: float = 0.1, frame_interval: float = 0.1,
                 min_alpha: float = 1.3, min_velocity: float = 0.1,
                 min_run_length: float = 0.5, min_duration: float = 0.5):
        self.pixel_size = pixel_size
        self.frame_interval = frame_interval
        self.min_alpha = min_alpha
        self.min_velocity = min_velocity  # μm/s
        self.min_run_length = min_run_length  # μm
        self.min_duration = min_duration  # seconds

    def compute_velocity(self, positions: np.ndarray) -> Tuple[float, float]:
        """Compute velocity and direction of motion"""
        # Convert positions to μm
        positions = positions * self.pixel_size

        # Calculate displacements
        displacements = np.diff(positions, axis=0)

        # Calculate instantaneous velocities
        velocities = np.sqrt(np.sum(displacements**2, axis=1)) / self.frame_interval

        # Calculate average velocity
        mean_velocity = np.mean(velocities)

        # Calculate overall direction (in radians)
        total_displacement = positions[-1] - positions[0]
        direction = np.arctan2(total_displacement[1], total_displacement[0])

        return mean_velocity, direction

    def identify_runs(self, track: pd.DataFrame) -> List[TransportResults]:
        """Identify periods of active transport"""
        positions = track[['x', 'y']].values
        frames = track['frame'].values
        runs = []

        # Use sliding window to identify potential runs
        window_size = max(int(self.min_duration / self.frame_interval), 5)

        for i in range(len(positions) - window_size):
            window = positions[i:i+window_size]

            # Compute velocity for this window
            velocity, direction = self.compute_velocity(window)

            # Compute run length
            run_length = np.linalg.norm(
                (window[-1] - window[0]) * self.pixel_size
            )

            # Check if this window shows active transport
            duration = window_size * self.frame_interval

            if (velocity >= self.min_velocity and
                run_length >= self.min_run_length and
                duration >= self.min_duration):

                runs.append(TransportResults(
                    is_active=True,
                    velocity=velocity,
                    direction=direction,
                    run_length=run_length,
                    duration=duration,
                    start_frame=frames[i],
                    end_frame=frames[i+window_size-1]
                ))

        return runs

    def analyze_track(self, track: pd.DataFrame) -> Dict:
        """Analyze active transport characteristics of a single track"""
        # Identify active transport runs
        runs = self.identify_runs(track)

        # Compute overall statistics
        if runs:
            mean_velocity = np.mean([run.velocity for run in runs])
            total_run_length = sum(run.run_length for run in runs)
            total_duration = sum(run.duration for run in runs)
            num_runs = len(runs)
        else:
            mean_velocity = 0
            total_run_length = 0
            total_duration = 0
            num_runs = 0

        return {
            'runs': runs,
            'mean_velocity': mean_velocity,
            'total_run_length': total_run_length,
            'total_duration': total_duration,
            'num_runs': num_runs
        }

    def analyze(self, tracks_df: pd.DataFrame) -> Dict:
        """Analyze all tracks in the dataset"""
        results = []
        track_ids = tracks_df['track_id'].unique()

        for track_id in track_ids:
            track = tracks_df[tracks_df['track_id'] == track_id]
            track_results = self.analyze_track(track)

            # Store results
            results.append({
                'track_id': track_id,
                'mean_velocity': track_results['mean_velocity'],
                'total_run_length': track_results['total_run_length'],
                'total_duration': track_results['total_duration'],
                'num_runs': track_results['num_runs'],
                'runs': track_results['runs']
            })

        return {
            'results_df': pd.DataFrame(results),
            'analysis_params': {
                'pixel_size': self.pixel_size,
                'frame_interval': self.frame_interval,
                'min_alpha': self.min_alpha,
                'min_velocity': self.min_velocity,
                'min_run_length': self.min_run_length,
                'min_duration': self.min_duration
            }
        }    
# spt_gui.py

class SPTAnalyzerGUI(QMainWindow):
    

    def setup_analysis_widgets(self):
        # Create all analysis widgets
        self.analysis_widgets = {
            'diffusion analysis': DiffusionAnalysisWidget(parent=self),
            'active transport': ActiveTransportWidget(parent=self),
            'boundary crossing': BoundaryCrossingWidget(parent=self),
            'dwell time analysis': DwellTimeWidget(parent=self),
            'crowding effects': CrowdingWidget(parent=self),
            'diffusion population': DiffusionPopulationWidget(parent=self),
            'gel structure': GelStructureWidget(parent=self),
            'microcompartment analysis': MicrocompartmentWidget(parent=self),
            'multi-channel analysis': MultiChannelAnalysisWidget(parent=self),
            # Add other analysis widgets here as they are created
        }

        # Add them to the analysis stack
        for name, widget in self.analysis_widgets.items():
            self.analysis_stack.addWidget(widget)
            self.analysis_type_combo.addItem(name.replace('_', ' ').title())

    def run_analysis(self):
        analysis_type = self.analysis_type_combo.currentText().lower().replace(' ', '_')
        widget = self.analysis_widgets[analysis_type]

        # Get parameters from the current widget
        params = widget.get_parameters()

        # Run analysis in background thread
        self.worker = AnalysisWorker(
            self.analysis_manager,
            analysis_type,
            self.current_tracks,
            params
        )

        # Connect signals
        self.worker.finished.connect(self.on_analysis_complete)
        self.worker.error.connect(self.on_analysis_error)
        self.worker.result.connect(self.update_results)

        # Start analysis
        self.worker_thread.start()

    def update_results(self, results):
        analysis_type = self.analysis_type_combo.currentText().lower().replace(' ', '_')
        widget = self.analysis_widgets[analysis_type]

        # Clear previous plots
        widget.plot_widget.clear()

        # Update plots based on analysis type
        if analysis_type == 'diffusion_models':
            self.plot_diffusion_models(results, widget.plot_widget)
        elif analysis_type == 'diffusion_population':
            self.plot_diffusion_population(results, widget.plot_widget)
        # ... similar for other analysis types ...

        # Update text summary
        widget.results_text.setText(self.format_results_summary(results))

        # Add other analysis tabs similarly
        # ...

        self.setCentralWidget(self.analysis_tabs)

    def _connect_signals(self):
        """Connect signals and slots."""
        # Connect analysis widgets signals
        for widget in self.analysis_widgets.values():
            widget.analysis_requested.connect(self._handle_analysis_request)
            widget.export_requested.connect(self._handle_export_request)
            widget.parameters_changed.connect(self._handle_parameter_change)

    def _handle_analysis_request(self, analysis_type: str):
        """Handle analysis request from widgets."""
        try:
            if not hasattr(self, 'tracks_df') or self.tracks_df is None:
                raise ValueError("No tracking data loaded")

            widget = self.analysis_widgets[analysis_type]
            analyzer = self.analyzers[analysis_type]

            # Show progress dialog
            progress = QProgressDialog(
                f"Running {analysis_type} analysis...",
                "Cancel",
                0,
                100,
                self
            )
            progress.setWindowModality(Qt.WindowModal)

            # Run analysis in background
            worker = AnalysisWorker(
                analyzer=analyzer,
                data=self.tracks_df,
                parameters=widget.get_parameters()
            )

            # Connect worker signals
            worker.progress.connect(progress.setValue)
            worker.finished.connect(progress.close)
            worker.result.connect(
                lambda results: self._handle_analysis_complete(
                    analysis_type,
                    results
                )
            )
            worker.error.connect(self._handle_analysis_error)

            # Start analysis
            self.thread_pool.start(worker)

        except Exception as e:
            self._show_error("Analysis Error", str(e))

    def _handle_analysis_complete(self, analysis_type: str, results: AnalysisResults):
        """Handle analysis completion."""
        try:
            # Update widget with results
            widget = self.analysis_widgets[analysis_type]
            widget.update_results(results)

            # Update status
            self.statusBar().showMessage(
                f"{analysis_type} analysis completed successfully",
                5000
            )

        except Exception as e:
            self._show_error("Error", f"Failed to update results: {str(e)}")

    def _handle_export_request(self, analysis_type: str, file_path: str):
        """Handle export request."""
        try:
            analyzer = self.analyzers[analysis_type]
            if analyzer.results is None:
                raise ValueError("No results available for export")

            success, message = self.analysis_exporter.export_results(
                analysis_type,
                analyzer.results,
                file_path
            )

            if success:
                self.statusBar().showMessage(message, 5000)
            else:
                raise ValueError(message)

        except Exception as e:
            self._show_error("Export Error", str(e))

        # Initialize components
        self.init_ui()
        self.init_analysis_components()
        self.setup_connections()

        # Analysis state
        self.current_project = None
        self.tracks_df = None
        self.analysis_results = {}
        self.auto_update_analysis = True

    def init_analysis_components(self):
        """Initialize analysis-related components."""
        # Initialize Analysis Manager
        self.analysis_manager = AnalysisManager()

        # Initialize Analysis Exporter
        self.analysis_exporter = AnalysisExporter(parent=self)

        # Create analysis widgets
        self.create_analysis_widgets()

        # Setup analysis toolbar
        self.setup_analysis_toolbar()

    def create_analysis_tab(self):
    """Create the analysis tab."""
    analysis_tab = QWidget()
    layout = QVBoxLayout()

    splitter = QSplitter(Qt.Horizontal)

    # Analysis menu (left side)
    analysis_menu = QWidget()
    menu_layout = QVBoxLayout(analysis_menu)

    analysis_type_label = QLabel("Select Analysis Type:")
    menu_layout.addWidget(analysis_type_label)

    # Analysis type selection
    self.analysis_type = QComboBox()
    # Add items here - these should match the keys in self.analysis_widgets
    self.analysis_type.addItems([
        "Diffusion Analysis",
        "Active Transport",
        "Boundary Crossing",
        "Dwell Time Analysis",
        "Crowding Effects",
        "Diffusion Population",
        "Gel Structure",
        "Microcompartment Analysis",
        "Multi-Channel Analysis"
    ])
    self.analysis_type.currentIndexChanged.connect(self.update_analysis_parameters)
    menu_layout.addWidget(self.analysis_type)

    # Parameter area (managed by QStackedWidget)
    self.params_group = QGroupBox("Analysis Parameters")
    params_layout = QVBoxLayout() # Use QVBoxLayout for the stacked widget
    self.params_stack = QStackedWidget() # <--- Add QStackedWidget

    # Add each analysis widget to the stacked widget
    for widget in self.analysis_widgets.values():
        self.params_stack.addWidget(widget)

    params_layout.addWidget(self.params_stack)
    self.params_group.setLayout(params_layout)
    menu_layout.addWidget(self.params_group)

    # Run analysis button
    run_analysis_btn = QPushButton("Run Analysis")
    run_analysis_btn.clicked.connect(self.run_analysis)
    menu_layout.addWidget(run_analysis_btn)

    menu_layout.addStretch()

    # Results display (right side) - Keep as is, or update tab names if needed
    results_widget = QWidget()
    results_layout = QVBoxLayout(results_widget)

    results_label = QLabel("Analysis Results")
    results_label.setAlignment(Qt.AlignCenter)
    results_layout.addWidget(results_label)

    # Tabs for different result views
    self.results_tabs = QTabWidget()

    # Summary tab
    summary_tab = QWidget()
    summary_layout = QVBoxLayout(summary_tab)
    self.results_summary = QTextEdit()
    self.results_summary.setReadOnly(True)
    summary_layout.addWidget(self.results_summary)
    self.results_tabs.addTab(summary_tab, "Summary")

    # Table tab
    table_tab = QWidget()
    table_layout = QVBoxLayout(table_tab)
    self.results_table = QTableWidget()
    table_layout.addWidget(self.results_table)
    self.results_tabs.addTab(table_tab, "Table")

    # Plot tab
    plot_tab = QWidget()
    plot_layout = QVBoxLayout(plot_tab)
    # Ensure this canvas is used for generic plot display if needed
    self.results_canvas = MplCanvas(width=5, height=4, dpi=100)
    plot_layout.addWidget(self.results_canvas)
    plot_toolbar = NavigationToolbar(self.results_canvas, self)
    plot_layout.addWidget(plot_toolbar)
    self.results_tabs.addTab(plot_tab, "Plot")

    results_layout.addWidget(self.results_tabs)

    # Export results button
    export_results_btn = QPushButton("Export Results")
    export_results_btn.clicked.connect(self.export_results)
    results_layout.addWidget(export_results_btn)


    # Add widgets to splitter
    splitter.addWidget(analysis_menu)
    splitter.addWidget(results_widget)
    splitter.setSizes([300, 700]) # Set initial sizes

    layout.addWidget(splitter)

    analysis_tab.setLayout(layout)
    self.tabs.addTab(analysis_tab, "Analysis")

    # Initialize the first analysis type (will call update_analysis_parameters)
    self.update_analysis_parameters(0)

    def create_active_transport_tab(self):
        """Create and setup active transport analysis tab."""
        # Create container widget and layout
        self.active_transport_container = QWidget()
        layout = QVBoxLayout(self.active_transport_container)

        # Create active transport widget
        self.active_transport_widget = ActiveTransportWidget(parent=self)
        layout.addWidget(self.active_transport_widget)

        # Add export button
        export_btn = QPushButton("Export Results")
        export_btn.clicked.connect(lambda: self.export_analysis_results("active_transport"))
        layout.addWidget(export_btn)

        # Add to tab widget
        self.analysis_tab_widget.addTab(
            self.active_transport_container,
            "Active Transport"
        )

    def setup_analysis_toolbar(self):
        """Setup the analysis toolbar."""
        self.analysis_toolbar = self.addToolBar("Analysis")

        # Add analysis actions
        self.run_analysis_action = QAction("Run Analysis", self)
        self.run_analysis_action.setIcon(QIcon.fromTheme("system-run"))
        self.run_analysis_action.triggered.connect(self.run_current_analysis)
        self.analysis_toolbar.addAction(self.run_analysis_action)

        # Export action
        self.export_action = QAction("Export Results", self)
        self.export_action.setIcon(QIcon.fromTheme("document-save"))
        self.export_action.triggered.connect(self.export_current_results)
        self.analysis_toolbar.addAction(self.export_action)

        # Auto-update toggle
        self.auto_update_action = QAction("Auto Update", self)
        self.auto_update_action.setCheckable(True)
        self.auto_update_action.setChecked(True)
        self.auto_update_action.triggered.connect(self.toggle_auto_update)
        self.analysis_toolbar.addAction(self.auto_update_action)

    def setup_connections(self):
        """Setup signal connections."""
        # Connect active transport signals
        self.active_transport_widget.parameters_changed.connect(
            self.on_transport_params_changed
        )

        # Connect analysis manager signals
        self.analysis_manager.analysis_completed.connect(
            self.on_analysis_completed
        )
        self.analysis_manager.error_occurred.connect(
            self.on_analysis_error
        )

    def export_analysis_results(self, analysis_type):
        """Export results for specific analysis type."""
        if analysis_type not in self.analysis_results:
            QMessageBox.warning(
                self,
                "Export Error",
                f"No results available for {analysis_type} analysis."
            )
            return

        try:
            success, message = self.analysis_exporter.export_results(
                analysis_type,
                self.analysis_results[analysis_type]
            )

            if success:
                self.statusBar().showMessage(message, 3000)
            else:
                QMessageBox.warning(self, "Export Error", message)

        except Exception as e:
            QMessageBox.critical(
                self,
                "Export Error",
                f"Failed to export results: {str(e)}"
            )

    def export_current_results(self):
        """Export results for current analysis tab."""
        current_tab = self.analysis_tab_widget.currentWidget()

        if isinstance(current_tab, QWidget):
            tab_name = self.analysis_tab_widget.tabText(
                self.analysis_tab_widget.currentIndex()
            )

            analysis_type = tab_name.lower().replace(" ", "_")
            self.export_analysis_results(analysis_type)

    def toggle_auto_update(self, checked):
        """Toggle auto-update analysis."""
        self.auto_update_analysis = checked
        self.active_transport_widget.auto_update = checked

    def on_transport_params_changed(self, params):
        """Handle changes in transport analysis parameters."""
        if self.auto_update_analysis:
            self.run_transport_analysis(params)

    def run_transport_analysis(self, params=None):
        """Run active transport analysis."""
        if self.tracks_df is None:
            QMessageBox.warning(
                self,
                "Analysis Error",
                "No tracks loaded. Please load tracking data first."
            )
            return

        try:
            # Show progress dialog
            self.progress_dialog = QProgressDialog(
                "Running active transport analysis...",
                "Cancel",
                0,
                100,
                self
            )
            self.progress_dialog.setWindowModality(Qt.WindowModal)

            # Get parameters if not provided
            if params is None:
                params = self.active_transport_widget.get_current_parameters()

            # Run analysis in background
            self.analysis_thread = QThread()
            self.analysis_worker = AnalysisWorker(
                self.analysis_manager,
                "active_transport",
                self.tracks_df,
                params
            )

            # Setup worker
            self.analysis_worker.moveToThread(self.analysis_thread)
            self.analysis_thread.started.connect(self.analysis_worker.run)
            self.analysis_worker.finished.connect(self.analysis_thread.quit)
            self.analysis_worker.finished.connect(self.analysis_worker.deleteLater)
            self.analysis_thread.finished.connect(self.analysis_thread.deleteLater)

            # Connect progress and results
            self.analysis_worker.progress.connect(self.progress_dialog.setValue)
            self.analysis_worker.result.connect(self.on_transport_analysis_completed)
            self.analysis_worker.error.connect(self.on_analysis_error)

            # Start analysis
            self.analysis_thread.start()

        except Exception as e:
            QMessageBox.critical(
                self,
                "Analysis Error",
                f"Failed to start analysis: {str(e)}"
            )

    def on_transport_analysis_completed(self, results):
        """Handle completion of transport analysis."""
        try:
            # Store results
            self.analysis_results['active_transport'] = results

            # Update widget
            self.active_transport_widget.update_results(results)

            # Close progress dialog
            if hasattr(self, 'progress_dialog'):
                self.progress_dialog.close()

            # Show success message
            self.statusBar().showMessage(
                "Active transport analysis completed successfully",
                3000
            )

        except Exception as e:
            QMessageBox.critical(
                self,
                "Error",
                f"Failed to process analysis results: {str(e)}"
            )

    def on_analysis_error(self, error_msg):
        """Handle analysis errors."""
        QMessageBox.critical(self, "Analysis Error", error_msg)
        if hasattr(self, 'progress_dialog'):
            self.progress_dialog.close()

    def __init__(self):
        super().__init__()
        # ... existing initialization code ...

        # Initialize analysis manager
        self.analysis_manager = AnalysisManager()

    def detect_particles(self):
        """Detect particles in the image stack."""
        if self.image_stack is None:
            QMessageBox.warning(self, "Warning", "No image stack loaded")
            return

        try:
            # Get detection parameters from GUI
            detector_params = {
                "method": self.detection_method.currentText().lower(),
                "min_sigma": self.min_sigma.value(),
                "max_sigma": self.max_sigma.value(),
                "num_sigma": self.num_sigma.value(),
                "threshold": self.threshold.value(),
                "exclude_border": 2,
                "subpixel": True
            }

            # Setup detector
            self.analysis_manager.setup_detector(detector_params)

            # Create worker thread for detection
            self.worker = WorkerThread(
                "detect_particles",
                {"frames": self.image_stack,
                 "analysis_manager": self.analysis_manager}
            )

            # ... rest of the detection code ...

    def link_tracks(self):
        """Link detected particles into tracks."""
        if not self.detections:
            QMessageBox.warning(self, "Warning", "No particles detected")
            return

        try:
            # Get tracking parameters from GUI
            tracking_params = {
                "linking_method": self.linking_method.currentText().lower(),
                "max_distance": self.max_distance.value(),
                "max_gap_closing": self.max_gap_closing.value(),
                "min_track_length": self.min_track_length.value()
            }

            # Setup tracker
            self.analysis_manager.setup_tracker(tracking_params)

            # Create worker thread for tracking
            self.worker = WorkerThread(
                "track_particles",
                {"detections": self.detections,
                 "analysis_manager": self.analysis_manager}
            )

            # ... rest of the tracking code ...

    def run_analysis(self):
        """Run selected analysis on tracks."""
        if self.tracks_df is None:
            QMessageBox.warning(self, "Warning", "No tracks available")
            return

        try:
            # Get analysis parameters from GUI
            analysis_type = self.analysis_type.currentText().lower()
            analysis_params = self.get_analysis_parameters()

            # Run analysis through manager
            results = self.analysis_manager.run_analysis(
                analysis_type,
                self.tracks_df,
                analysis_params
            )

            # Update results display
            self.display_analysis_results(results)

        except Exception as e:
            logger.error(f"Analysis error: {str(e)}", exc_info=True)
            QMessageBox.critical(self, "Error", f"Analysis failed: {str(e)}")
    """Main window for the SPT Analyzer application."""
    
    def __init__(self):
        super().__init__()
        
        # Set window properties
        self.setWindowTitle("SPT Analyzer")
        self.setMinimumSize(1200, 800)
        
        # Initialize data structures
        self.image_stack = None
        self.current_frame = 0
        self.detections = []
        self.tracks_df = None
        self.analysis_results = {}
        
        # Project settings
        self.project_settings = {
            "project_name": "Untitled Project",
            "pixel_size": 0.1,  # μm per pixel
            "frame_interval": 0.1,  # seconds per frame
            "temperature": 25.0,  # °C
            "particle_radius": 5.0,  # nm
            "notes": ""
        }
        
        # Set up UI
        self.setup_ui()
        
        # Set up settings
        self.settings = QSettings("SPTAnalyzer", "SPTAnalyzerApp")
        self.load_app_settings()
        
        # Show welcome message
        self.statusBar().showMessage("Welcome to SPT Analyzer. Start by creating or loading a project.")
        
 def setup_ui(self):
    """Set up the main UI components."""
    # Create central widget and main layout
    central_widget = QWidget()
    self.setCentralWidget(central_widget)
    main_layout = QVBoxLayout(central_widget)

    # Create tab widget
    self.tabs = QTabWidget()
    self.tabs.setTabPosition(QTabWidget.North)
    self.tabs.currentChanged.connect(self.handle_tab_change)

    # Create tabs
    self.create_project_tab()
    self.create_image_processing_tab()
    self.create_tracking_tab()
    self.create_analysis_tab()
    self.create_visualization_tab()
    self.create_batch_tab()

    # Add tabs to tab widget
    main_layout.addWidget(self.tabs)

    # Create status bar
    self.statusBar = QStatusBar()
    self.setStatusBar(self.statusBar)

    # Create progress bar in status bar
    self.progress_bar = QProgressBar()
    self.progress_bar.setVisible(False)
    self.progress_bar.setMaximumWidth(250)
    self.statusBar.addPermanentWidget(self.progress_bar)

    # Set up menu bar
    self.create_menu_bar()

    # Connect menu bar actions
    self.connect_menu_actions()

    # Connect tab-specific widgets
    self.connect_project_tab_widgets()
    self.connect_image_processing_tab_widgets()
    self.connect_tracking_tab_widgets()
    self.connect_analysis_tab_widgets()
    self.connect_visualization_tab_widgets()
    self.connect_batch_tab_widgets()
    # Setup boundary crossing analysis section
    self.setup_boundary_crossing_ui()
    self.setup_boundary_crossing_connections()
    # Set up matplotlib widgets
    self.setup_matplotlib_widgets()
def connect_menu_actions(self):
    """Connect menu bar actions to their respective slots."""
    # File menu connections
    menubar = self.menuBar()
    file_menu = menubar.findChild(QMenu, "")  # First menu is File

    # Connect file menu actions
    for action in file_menu.actions():
        if action.text() == "&New Project":
            action.triggered.connect(self.new_project)
        elif action.text() == "&Open Project":
            action.triggered.connect(self.open_project)
        elif action.text() == "&Save Project":
            action.triggered.connect(self.save_project)
        elif action.text() == "E&xit":
            action.triggered.connect(self.close)

    # Connect import/export submenu actions
    for submenu in file_menu.findChildren(QMenu):
        if submenu.title() == "Import":
            for action in submenu.actions():
                if action.text() == "Image Stack":
                    action.triggered.connect(self.import_image_stack)
                elif action.text() == "Tracks":
                    action.triggered.connect(self.import_tracks)
        elif submenu.title() == "Export":
            for action in submenu.actions():
                if action.text() == "Tracks":
                    action.triggered.connect(self.export_tracks)
                elif action.text() == "Analysis Results":
                    action.triggered.connect(self.export_results)
                elif action.text() == "Current Figure":
                    action.triggered.connect(self.export_figure)

    # Edit menu connections
    edit_menu = menubar.findChildren(QMenu)[1]  # Second menu is Edit
    for action in edit_menu.actions():
        if action.text() == "&Settings":
            action.triggered.connect(self.edit_project_settings)

    # Help menu connections
    help_menu = menubar.findChildren(QMenu)[-1]  # Last menu is Help
    for action in help_menu.actions():
        if action.text() == "&About":
            action.triggered.connect(self.show_about_dialog)
        elif action.text() == "&Documentation":
            action.triggered.connect(self.show_documentation)
def setup_boundary_crossing_ui(self):
    """Setup UI elements for boundary crossing analysis"""
    # Create boundary crossing group box
    boundary_group = QGroupBox("Boundary Crossing Analysis")
    layout = QVBoxLayout()

    # Parameters section
    param_layout = QFormLayout()
    self.pixel_size_spinbox = QDoubleSpinBox()
    self.pixel_size_spinbox.setRange(0.001, 1000.0)
    self.pixel_size_spinbox.setValue(1.0)
    self.pixel_size_spinbox.setSuffix(" µm/pixel")

    self.dt_spinbox = QDoubleSpinBox()
    self.dt_spinbox.setRange(0.001, 1000.0)
    self.dt_spinbox.setValue(0.014)
    self.dt_spinbox.setSuffix(" s")

    param_layout.addRow("Pixel Size:", self.pixel_size_spinbox)
    param_layout.addRow("Time Interval:", self.dt_spinbox)

    # Buttons
    button_layout = QHBoxLayout()
    self.boundary_analyze_btn = QPushButton("Analyze Crossings")
    self.angular_dist_btn = QPushButton("Angular Distribution")
    self.angular_dist_btn.setEnabled(False)
    button_layout.addWidget(self.boundary_analyze_btn)
    button_layout.addWidget(self.angular_dist_btn)

    # Results tables
    self.crossing_results_table = QTableWidget()
    self.angular_summary_table = QTableWidget()

    # Plot widget
    self.angular_plot_widget = MatplotlibWidget()

    # Add all to layout
    layout.addLayout(param_layout)
    layout.addLayout(button_layout)
    layout.addWidget(self.crossing_results_table)
    layout.addWidget(self.angular_summary_table)
    layout.addWidget(self.angular_plot_widget)

    boundary_group.setLayout(layout)

    # Add to analysis tab
    self.analysis_tab_layout.addWidget(boundary_group)
def connect_project_tab_widgets(self):
    """Connect widgets in the project tab to their slots."""
    project_tab = self.tabs.widget(0)  # Project tab is the first tab

    # Find buttons in the project tab
    buttons = project_tab.findChildren(QPushButton)
    for button in buttons:
        if button.text() == "New Project":
            button.clicked.connect(self.new_project)
        elif button.text() == "Load Project":
            button.clicked.connect(self.open_project)
        elif button.text() == "Save Project":
            button.clicked.connect(self.save_project)
        elif button.text() == "Edit Settings":
            button.clicked.connect(self.edit_project_settings)
def setup_active_transport_connections(self):
    """Setup connections for active transport analysis in the Analysis tab"""
    # Connect analysis buttons
    self.superdiffusion_btn.clicked.connect(self.run_superdiffusion_analysis)
    self.directed_motion_btn.clicked.connect(self.run_directed_motion_analysis)
    self.transport_params_btn.clicked.connect(self.calculate_transport_parameters)
    
    # Connect parameter update widgets
    self.dt_spinbox.valueChanged.connect(self.update_transport_params)
    self.min_track_length_spinbox.valueChanged.connect(self.update_transport_params)
    self.min_alpha_spinbox.valueChanged.connect(self.update_transport_params)
    self.min_duration_spinbox.valueChanged.connect(self.update_transport_params)
    self.min_displacement_spinbox.valueChanged.connect(self.update_transport_params)

def update_transport_params(self):
    """Update active transport analysis parameters"""
    self.transport_params = {
        'dt': self.dt_spinbox.value(),
        'min_track_length': self.min_track_length_spinbox.value(),
        'min_superdiffusive_alpha': self.min_alpha_spinbox.value(),
        'min_duration': self.min_duration_spinbox.value(),
        'min_displacement': self.min_displacement_spinbox.value()
    }

def run_superdiffusion_analysis(self):
    """Execute superdiffusion analysis"""
    try:
        if not hasattr(self, 'tracks_df') or self.tracks_df is None:
            self.show_error_message("No tracks loaded", "Please load tracking data first.")
            return

        # Initialize analyzer with current parameters
        self.active_transport_analyzer = ActiveTransportAnalyzer(
            dt=self.transport_params['dt'],
            min_track_length=self.transport_params['min_track_length'],
            min_superdiffusive_alpha=self.transport_params['min_superdiffusive_alpha']
        )

        # Run analysis
        results = self.active_transport_analyzer.analyze_superdiffusion(
            self.tracks_df,
            self.compartment_masks if hasattr(self, 'compartment_masks') else None
        )

        # Update results table
        self.update_superdiffusion_results_table(results)
        
        # Plot results
        self.plot_superdiffusion_results(results)

        # Update status
        n_superdiffusive = sum(1 for r in results.values() if r['superdiffusive'])
        self.status_bar.showMessage
def setup_active_transport_ui(self):
    """Setup UI elements for active transport analysis"""
    # Create active transport group box
    transport_group = QGroupBox("Active Transport Analysis")
    layout = QVBoxLayout()

    # Parameters section
    param_layout = QFormLayout()

    self.dt_spinbox = QDoubleSpinBox()
    self.dt_spinbox.setRange(0.001, 1.0)
    self.dt_spinbox.setValue(0.014)
    self.dt_spinbox.setSuffix(" s")

    self.min_track_length_spinbox = QSpinBox()
    self.min_track_length_spinbox.setRange(5, 100)
    self.min_track_length_spinbox.setValue(10)

    self.min_alpha_spinbox = QDoubleSpinBox()
    self.min_alpha_spinbox.setRange(1.0, 2.0)
    self.min_alpha_spinbox.setValue(1.3)
    self.min_alpha_spinbox.setSingleStep(0.1)

    self.min_duration_spinbox = QDoubleSpinBox()
    self.min_duration_spinbox.setRange(0.1, 10.0)
    self.min_duration_spinbox.setValue(1.0)
    self.min_duration_spinbox.setSuffix(" s")

    self.min_displacement_spinbox = QDoubleSpinBox()
    self.min_displacement_spinbox.setRange(0.1, 20.0)
    self.min_displacement_spinbox.setValue(2.0)
    self.min_displacement_spinbox.setSuffix(" μm")

    param_layout.addRow("Time Interval:", self.dt_spinbox)
    param_layout.addRow("Min Track Length:", self.min_track_length_spinbox)
    param_layout.addRow("Min Alpha:", self.min_alpha_spinbox)
    param_layout.addRow("Min Duration:", self.min_duration_spinbox)
    param_layout.addRow("Min Displacement:", self.min_displacement_spinbox)

    # Analysis buttons
    button_layout = QHBoxLayout()
    self.superdiffusion_btn = QPushButton("Analyze Superdiffusion")
    self.directed_motion_btn = QPushButton("Analyze Directed Motion")
    self.transport_params_btn = QPushButton("Calculate Parameters")
    button_layout.addWidget(self.superdiffusion_btn)
    button_layout.addWidget(self.directed_motion_btn)
    button_layout.addWidget(self.transport_params_btn)

    # Compartment selection
    self.compartment_combo = QComboBox()
    self.compartment_combo.addItem("All")

    # Results tables
    self.superdiffusion_table = QTableWidget()
    self.directed_motion_table = QTableWidget()

    # Plot widgets
    self.superdiffusion_plot_widget = MatplotlibWidget()
    self.directed_motion_plot_widget = MatplotlibWidget()

    # Transport parameters text display
    self.transport_params_text = QTextEdit()
    self.transport_params_text.setReadOnly(True)

    # Add all to layout
    layout.addLayout(param_layout)
    layout.addLayout(button_layout)
    layout.addWidget(QLabel("Select Compartment:"))
    layout.addWidget(self.compartment_combo)
    layout.addWidget(QLabel("Superdiffusion Results:"))
    layout.addWidget(self.superdiffusion_table)
    layout.addWidget(self.superdiffusion_plot_widget)
    layout.addWidget(QLabel("Directed Motion Results:"))
    layout.addWidget(self.directed_motion_table)
    layout.addWidget(self.directed_motion_plot_widget)
    layout.addWidget(QLabel("Transport Parameters:"))
    layout.addWidget(self.transport_params_text)

    transport_group.setLayout(layout)

    # Add to analysis tab
    self.analysis_tab_layout.addWidget(transport_group)  
def setup_active_transport_ui(self):
    """Setup UI elements for active transport analysis"""
    # Create active transport group box
    transport_group = QGroupBox("Active Transport Analysis")
    layout = QVBoxLayout()

    # Parameters section
    param_layout = QFormLayout()

    self.dt_spinbox = QDoubleSpinBox()
    self.dt_spinbox.setRange(0.001, 1.0)
    self.dt_spinbox.setValue(0.014)
    self.dt_spinbox.setSuffix(" s")

    self.min_track_length_spinbox = QSpinBox()
    self.min_track_length_spinbox.setRange(5, 100)
    self.min_track_length_spinbox.setValue(10)

    self.min_alpha_spinbox = QDoubleSpinBox()
    self.min_alpha_spinbox.setRange(1.0, 2.0)
    self.min_alpha_spinbox.setValue(1.3)
    self.min_alpha_spinbox.setSingleStep(0.1)

    self.min_duration_spinbox = QDoubleSpinBox()
    self.min_duration_spinbox.setRange(0.1, 10.0)
    self.min_duration_spinbox.setValue(1.0)
    self.min_duration_spinbox.setSuffix(" s")

    self.min_displacement_spinbox = QDoubleSpinBox()
    self.min_displacement_spinbox.setRange(0.1, 20.0)
    self.min_displacement_spinbox.setValue(2.0)
    self.min_displacement_spinbox.setSuffix(" μm")

    param_layout.addRow("Time Interval:", self.dt_spinbox)
    param_layout.addRow("Min Track Length:", self.min_track_length_spinbox)
    param_layout.addRow("Min Alpha:", self.min_alpha_spinbox)
    param_layout.addRow("Min Duration:", self.min_duration_spinbox)
    param_layout.addRow("Min Displacement:", self.min_displacement_spinbox)

    # Analysis buttons
    button_layout = QHBoxLayout()
    self.superdiffusion_btn = QPushButton("Analyze Superdiffusion")
    self.directed_motion_btn = QPushButton("Analyze Directed Motion")
    self.transport_params_btn = QPushButton("Calculate Parameters")
    button_layout.addWidget(self.superdiffusion_btn)
    button_layout.addWidget(self.directed_motion_btn)
    button_layout.addWidget(self.transport_params_btn)

    # Compartment selection
    self.compartment_combo = QComboBox()
    self.compartment_combo.addItem("All")

    # Results tables
    self.superdiffusion_table = QTableWidget()
    self.directed_motion_table = QTableWidget()

    # Plot widgets
    self.superdiffusion_plot_widget = MatplotlibWidget()
    self.directed_motion_plot_widget = MatplotlibWidget()

    # Transport parameters text display
    self.transport_params_text = QTextEdit()
    self.transport_params_text.setReadOnly(True)

    # Add all to layout
    layout.addLayout(param_layout)
    layout.addLayout(button_layout)
    layout.addWidget(QLabel("Select Compartment:"))
    layout.addWidget(self.compartment_combo)
    layout.addWidget(QLabel("Superdiffusion Results:"))
    layout.addWidget(self.superdiffusion_table)
    layout.addWidget(self.superdiffusion_plot_widget)
    layout.addWidget(QLabel("Directed Motion Results:"))
    layout.addWidget(self.directed_motion_table)
    layout.addWidget(self.directed_motion_plot_widget)
    layout.addWidget(QLabel("Transport Parameters:"))
    layout.addWidget(self.transport_params_text)

    transport_group.setLayout(layout)

    # Add to analysis tab
    self.analysis_tab_layout.addWidget(transport_group)
def setup_detector_ui(self):
    """Setup UI elements for particle detection"""
    # Create detector group box
    detector_group = QGroupBox("Particle Detection")
    layout = QVBoxLayout()

    # Method selection
    method_layout = QFormLayout()
    self.detection_method_combo = QComboBox()
    self.detection_method_combo.addItems(['Gaussian', 'Laplacian', 'DoH', 'LoG', 'Wavelet'])
    method_layout.addRow("Detection Method:", self.detection_method_combo)

    # Basic parameters
    param_layout = QFormLayout()

    self.threshold_spinbox = QDoubleSpinBox()
    self.threshold_spinbox.setRange(0, 1000)
    self.threshold_spinbox.setValue(0.5)
    self.threshold_spinbox.setSingleStep(0.1)

    self.threshold_relative_check = QCheckBox("Relative")
    self.threshold_relative_check.setChecked(True)

    threshold_layout = QHBoxLayout()
    threshold_layout.addWidget(self.threshold_spinbox)
    threshold_layout.addWidget(self.threshold_relative_check)
    param_layout.addRow("Threshold:", threshold_layout)

    self.min_distance_spinbox = QSpinBox()
    self.min_distance_spinbox.setRange(1, 100)
    self.min_distance_spinbox.setValue(5)
    param_layout.addRow("Min Distance:", self.min_distance_spinbox)

    self.diameter_spinbox = QSpinBox()
    self.diameter_spinbox.setRange(3, 100)
    self.diameter_spinbox.setValue(7)
    param_layout.addRow("Particle Diameter:", self.diameter_spinbox)

    # Subpixel refinement
    subpixel_layout = QHBoxLayout()
    self.subpixel_check = QCheckBox("Enable")
    self.subpixel_check.setChecked(True)
    self.subpixel_method_combo = QComboBox()
    self.subpixel_method_combo.addItems(['CoM', 'Gaussian'])
    subpixel_layout.addWidget(self.subpixel_check)
    subpixel_layout.addWidget(self.subpixel_method_combo)
    param_layout.addRow("Subpixel:", subpixel_layout)

    # Wavelet-specific parameters
    wavelet_layout = QFormLayout()

    self.wavelet_type_label = QLabel("Wavelet Type:")
    self.wavelet_type_combo = QComboBox()
    self.wavelet_type_combo.addItems(['db4', 'sym5', 'haar'])
    wavelet_layout.addRow(self.wavelet_type_label, self.wavelet_type_combo)

    self.wavelet_levels_label = QLabel("Wavelet Levels:")
    self.wavelet_levels_spinbox = QSpinBox()
    self.wavelet_levels_spinbox.setRange(1, 10)
    self.wavelet_levels_spinbox.setValue(3)
    wavelet_layout.addRow(self.wavelet_levels_label, self.wavelet_levels_spinbox)

    self.wavelet_method_label = QLabel("Enhancement Method:")
    self.wavelet_method_combo = QComboBox()
    self.wavelet_method_combo.addItems(['Residual', 'Direct'])
    wavelet_layout.addRow(self.wavelet_method_label, self.wavelet_method_combo)

    # Control buttons
    button_layout = QHBoxLayout()
    self.detect_btn = QPushButton("Detect Particles")
    self.preview_check = QCheckBox("Live Preview")
    button_layout.addWidget(self.detect_btn)
    button_layout.addWidget(self.preview_check)

    # Results display
    self.detection_results_table = QTableWidget()
    self.detection_plot_widget = MatplotlibWidget()

    # Add all to layout
    layout.addLayout(method_layout)
    layout.addLayout(param_layout)
    layout.addLayout(wavelet_layout)
    layout.addLayout(button_layout)
    layout.addWidget(self.detection_results_table)
    layout.addWidget(self.detection_plot_widget)

    detector_group.setLayout(layout)

    # Add to detection tab
    self.detection_tab_layout.addWidget(detector_group)

    # Initialize parameter visibility
    self.update_parameter_visibility('gaussian')   
# Add to SPTAnalyzerGUI class
# In SPTAnalyzerGUI class, modify create_analysis_tab method:
def save_project(self):
    """Save current project state."""
    try:
        file_path, _ = QFileDialog.getSaveFileName(
            self, "Save Project",
            "",
            "SPT Project (*.h5);;All Files (*.*)"
        )

        if file_path:
            # Prepare project data
            project_data = {
                'settings': self.project_settings,
                'tracks': self.tracks_df.to_dict() if self.tracks_df is not None else None,
                'analysis_results': self.analysis_results,
                'image_info': {
                    'shape': self.image_stack.shape if self.image_stack is not None else None,
                    'pixel_size': self.project_settings['pixel_size'],
                    'frame_interval': self.project_settings['frame_interval']
                } if self.image_stack is not None else None
            }

            # Save project using io module
            from utils.io import save_analysis_results
            save_analysis_results(project_data, file_path)

            # Save image stack if exists
            if self.image_stack is not None:
                image_path = os.path.splitext(file_path)[0] + "_images.tif"
                from utils.io import save_image_stack
                save_image_stack(self.image_stack, image_path)

            self.statusBar().showMessage(f"Project saved to {file_path}", 3000)

    except Exception as e:
        QMessageBox.critical(self, "Error", f"Failed to save project: {str(e)}")
        logger.error(f"Project save error: {e}", exc_info=True)

def open_project(self):
    """Load a saved project."""
    try:
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Open Project",
            "",
            "SPT Project (*.h5);;All Files (*.*)"
        )

        if file_path:
            # Load project data using io module
            from utils.io import load_analysis_results
            project_data = load_analysis_results(file_path)

            # Restore project settings
            self.project_settings = project_data.get('settings', {})
            self.update_project_display()

            # Restore tracks
            if project_data.get('tracks'):
                self.tracks_df = pd.DataFrame.from_dict(project_data['tracks'])

            # Restore analysis results
            self.analysis_results = project_data.get('analysis_results', {})

            # Try to load associated image stack
            image_path = os.path.splitext(file_path)[0] + "_images.tif"
            if os.path.exists(image_path):
                from utils.io import load_image_stack
                self.image_stack = load_image_stack(image_path)
                self.update_image_display()

            self.update_data_summary()
            self.statusBar().showMessage(f"Project loaded from {file_path}", 3000)

    except Exception as e:
        QMessageBox.critical(self, "Error", f"Failed to load project: {str(e)}")
        logger.error(f"Project load error: {e}", exc_info=True)

def import_tracks(self):
    """Import tracks from file."""
    try:
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Import Tracks",
            "",
            "All Supported Files (*.csv *.xlsx *.h5 *.json *.xml);;CSV (*.csv);;Excel (*.xlsx);;HDF5 (*.h5);;JSON (*.json);;TrackMate XML (*.xml);;All Files (*.*)"
        )

        if file_path:
            from utils.io import load_tracks
            self.tracks_df = load_tracks(file_path)

            # Update display
            self.update_tracking_display()
            self.update_data_summary()
            self.statusBar().showMessage(f"Imported {len(self.tracks_df)} track positions", 3000)

    except Exception as e:
        QMessageBox.critical(self, "Error", f"Failed to import tracks: {str(e)}")
        logger.error(f"Track import error: {e}", exc_info=True)

def export_tracks(self):
    """Export tracks to file."""
    try:
        if self.tracks_df is None:
            QMessageBox.warning(self, "Warning", "No tracks available to export")
            return

        file_path, selected_filter = QFileDialog.getSaveFileName(
            self, "Export Tracks",
            "",
            "CSV (*.csv);;Excel (*.xlsx);;HDF5 (*.h5);;JSON (*.json)"
        )

        if file_path:
            from utils.io import save_tracks
            save_tracks(self.tracks_df, file_path)
            self.statusBar().showMessage(f"Tracks exported to {file_path}", 3000)
def import_image_stack(self):
    """Import image stack."""
    try:
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Import Image Stack",
            "",
            "TIFF Files (*.tif *.tiff);;All Files (*.*)"
        )

        if file_path:
            from utils.io import load_image_stack
            self.image_stack = load_image_stack(
                file_path,
                scale=1.0,
                subset=None
            )

            # Update UI
            self.frame_slider.setMaximum(len(self.image_stack) - 1)
            self.tracking_frame_slider.setMaximum(len(self.image_stack) - 1)
            self.update_display_frame(0)
            self.update_data_summary()

            self.statusBar().showMessage(
                f"Loaded image stack with {len(self.image_stack)} frames",
                3000
            )

    except Exception as e:
        QMessageBox.critical(self, "Error", f"Failed to load image stack: {str(e)}")
        logger.error(f"Image stack load error: {e}", exc_info=True)
def import_image_stack(self):
    """Import image stack from file."""
    try:
        # Create file dialog with extended format support
        file_dialog = QFileDialog(self)
        file_dialog.setWindowTitle("Import Image Stack")
        
        # Define supported formats
        formats = [
            "Image files (*.tif *.tiff *.ims *.nd2 *.lsm *.czi)",
            "TIFF files (*.tif *.tiff)",
            "Imaris files (*.ims)",
            "Nikon ND2 files (*.nd2)",
            "Zeiss LSM files (*.lsm)",
            "Zeiss CZI files (*.czi)",
            "All files (*.*)"
        ]
        
        file_dialog.setNameFilters(formats)
        file_dialog.setFileMode(QFileDialog.ExistingFile)
        
        if file_dialog.exec_():
            file_path = file_dialog.selectedFiles()[0]
            
            # Show loading dialog with options
            loading_dialog = ImageLoadingDialog(self)
            if loading_dialog.exec_():
                # Get loading parameters
                params = loading_dialog.get_parameters()
                
                # Show progress dialog
                progress = QProgressDialog("Loading image stack...", "Cancel", 0, 100, self)
                progress.setWindowModality(Qt.WindowModal)
                
                # Create worker thread for loading
                self.worker = WorkerThread(
                    "load_image_stack",
                    {
                        "file_path": file_path,
                        "scale": params["scale"],
                        "subset": params["subset"]
                    }
                )
                
                # Connect signals
                self.worker.progress_updated.connect(progress.setValue)
                self.worker.operation_completed.connect(self.image_stack_loaded)
                self.worker.error_occurred.connect(self.loading_error)
                
                # Start worker
                self.worker.start()
                
    except Exception as e:
        self.show_error_message("Import Error", str(e))
        logger.error(f"Error importing image stack: {e}", exc_info=True)

class ImageLoadingDialog(QDialog):
    """Dialog for configuring image stack loading parameters."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Image Loading Options")
        self.setup_ui()

    def setup_ui(self):
        layout = QVBoxLayout()

        # Scaling options
        scale_group = QGroupBox("Intensity Scaling")
        scale_layout = QFormLayout()

        self.scale_spinbox = QDoubleSpinBox()
        self.scale_spinbox.setRange(0.001, 1000.0)
        self.scale_spinbox.setValue(1.0)
        self.scale_spinbox.setSingleStep(0.1)
        scale_layout.addRow("Scale Factor:", self.scale_spinbox)

        scale_group.setLayout(scale_layout)
        layout.addWidget(scale_group)

        # Frame subset options
        subset_group = QGroupBox("Frame Subset")
        subset_layout = QFormLayout()

        self.load_subset_check = QCheckBox("Load subset of frames")
        subset_layout.addRow(self.load_subset_check)

        self.start_frame = QSpinBox()
        self.start_frame.setRange(0, 999999)
        self.start_frame.setEnabled(False)
        subset_layout.addRow("Start Frame:", self.start_frame)

        self.end_frame = QSpinBox()
        self.end_frame.setRange(1, 999999)
        self.end_frame.setEnabled(False)
        subset_layout.addRow("End Frame:", self.end_frame)

        self.load_subset_check.stateChanged.connect(self.toggle_subset_options)

        subset_group.setLayout(subset_layout)
        layout.addWidget(subset_group)

        # Channel options
        channels_group = QGroupBox("Channels")
        channels_layout = QVBoxLayout()

        self.all_channels_radio = QRadioButton("Load all channels")
        self.all_channels_radio.setChecked(True)
        channels_layout.addWidget(self.all_channels_radio)

        self.select_channels_radio = QRadioButton("Select channels")
        channels_layout.addWidget(self.select_channels_radio)

        self.channel_list = QListWidget()
        self.channel_list.setEnabled(False)
        channels_layout.addWidget(self.channel_list)

        self.select_channels_radio.toggled.connect(self.channel_list.setEnabled)

        channels_group.setLayout(channels_layout)
        layout.addWidget(channels_group)

        # Buttons
        buttons = QDialogButtonBox(
            QDialogButtonBox.Ok | QDialogButtonBox.Cancel,
            Qt.Horizontal, self)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

        self.setLayout(layout)

    def toggle_subset_options(self, state):
        """Enable/disable subset spinboxes."""
        self.start_frame.setEnabled(state)
        self.end_frame.setEnabled(state)

    def get_parameters(self) -> Dict:
        """Return the selected loading parameters."""
        params = {
            "scale": self.scale_spinbox.value(),
            "subset": None if not self.load_subset_check.isChecked() else
                     (self.start_frame.value(), self.end_frame.value()),
            "channels": "all" if self.all_channels_radio.isChecked() else
                       [item.text() for item in self.channel_list.selectedItems()]
        }
        return params        
def edit_project_settings(self):
class ProjectManagerWidget(QWidget):
    """Widget for managing SPT projects."""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.project = None
        self.setup_ui()
        
    def setup_ui(self):
        layout = QVBoxLayout(self)
        
        # Project Info Group
        info_group = QGroupBox("Project Information")
        info_layout = QFormLayout()
        
        self.project_name = QLineEdit()
        info_layout.addRow("Project Name:", self.project_name)
        
        self.description = QTextEdit()
        self.description.setMaximumHeight(60)
        info_layout.addRow("Description:", self.description)
        
        info_group.setLayout(info_layout)
        layout.addWidget(info_group)
        
        # Treatment Groups Group
        treatment_group = QGroupBox("Treatment Groups")
        treatment_layout = QVBoxLayout()
        
        # Treatment list
        self.treatment_list = QListWidget()
        self.treatment_list.itemSelectionChanged.connect(self.update_cell_list)
        treatment_layout.addWidget(self.treatment_list)
        
        # Treatment buttons
        treatment_btn_layout = QHBoxLayout()
        self.add_treatment_btn = QPushButton("Add Treatment")
        self.add_treatment_btn.clicked.connect(self.add_treatment_group)
        self.remove_treatment_btn = QPushButton("Remove Treatment")
        self.remove_treatment_btn.clicked.connect(self.remove_treatment_group)
        treatment_btn_layout.addWidget(self.add_treatment_btn)
        treatment_btn_layout.addWidget(self.remove_treatment_btn)
        treatment_layout.addLayout(treatment_btn_layout)
        
        treatment_group.setLayout(treatment_layout)
        layout.addWidget(treatment_group)
        
        # Cells Group
        cells_group = QGroupBox("Cells")
        cells_layout = QVBoxLayout()
        
        self.cell_list = QListWidget()
        cells_layout.addWidget(self.cell_list)
        
        # Cell buttons
        cell_btn_layout = QHBoxLayout()
        self.add_cell_btn = QPushButton("Add Cell")
        self.add_cell_btn.clicked.connect(self.add_cell)
        self.remove_cell_btn = QPushButton("Remove Cell")
        self.remove_cell_btn.clicked.connect(self.remove_cell)
        cell_btn_layout.addWidget(self.add_cell_btn)
        cell_btn_layout.addWidget(self.remove_cell_btn)
        cells_layout.addLayout(cell_btn_layout)
        
        cells_group.setLayout(cells_layout)
        layout.addWidget(cells_group)
        
        # Project actions
        actions_layout = QHBoxLayout()
        self.save_project_btn = QPushButton("Save Project")
        self.save_project_btn.clicked.connect(self.save_project)
        self.load_project_btn = QPushButton("Load Project")
        self.load_project_btn.clicked.connect(self.load_project)
        actions_layout.addWidget(self.save_project_btn)
        actions_layout.addWidget(self.load_project_btn)
        layout.addLayout(actions_layout)
    """Edit project settings."""
    try:
        dialog = ProjectSettingsDialog(self, self.project_settings)
        if dialog.exec_() == QDialog.Accepted:
            # Get new settings
            new_settings = dialog.get_settings()

            # Save settings to file
            from utils.io import save_config
            config_path = os.path.join(
                os.path.dirname(os.path.abspath(__file__)),
                "config",
                "project_settings.yaml"
            )
            save_config(new_settings, config_path)

            # Update current settings
            self.project_settings = new_settings
            self.update_project_display()
            self.update_data_summary()

    except Exception as e:
        QMessageBox.critical(self, "Error", f"Failed to save settings: {str(e)}")
        logger.error(f"Settings save error: {e}", exc_info=True)

def create_new_project(self):
    """Create a new SPT project."""
    name = self.project_name.text().strip()
    if not name:
        QMessageBox.warning(self, "Warning", "Please enter a project name")
        return

    description = self.description.toPlainText()
    self.project = SPTProject(name, description)
    self.update_ui()

def add_treatment_group(self):
    """Add a new treatment group."""
    if not self.project:
        QMessageBox.warning(self, "Warning", "Please create or load a project first")
        return

    dialog = QDialog(self)
    dialog.setWindowTitle("Add Treatment Group")
    layout = QFormLayout(dialog)

    name_input = QLineEdit()
    layout.addRow("Name:", name_input)

    desc_input = QTextEdit()
    desc_input.setMaximumHeight(60)
    layout.addRow("Description:", desc_input)

    metadata_input = QTextEdit()
    metadata_input.setMaximumHeight(60)
    layout.addRow("Metadata (JSON):", metadata_input)

    buttons = QDialogButtonBox(
        QDialogButtonBox.Ok | QDialogButtonBox.Cancel,
        Qt.Horizontal, dialog)
    buttons.accepted.connect(dialog.accept)
    buttons.rejected.connect(dialog.reject)
    layout.addRow(buttons)

    if dialog.exec_() == QDialog.Accepted:
        try:
            name = name_input.text().strip()
            description = desc_input.toPlainText()
            metadata_text = metadata_input.toPlainText()
            metadata = json.loads(metadata_text) if metadata_text else None

            self.project.add_treatment_group(name, description, metadata)
            self.update_treatment_list()

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to add treatment group: {str(e)}")

def add_cell(self):
    """Add a cell to the selected treatment group."""
    if not self.project:
        QMessageBox.warning(self, "Warning", "Please create or load a project first")
        return

    selected_items = self.treatment_list.selectedItems()
    if not selected_items:
        QMessageBox.warning(self, "Warning", "Please select a treatment group")
        return

    treatment_name = selected_items[0].text()

    dialog = QDialog(self)
    dialog.setWindowTitle("Add Cell")
    layout = QFormLayout(dialog)

    cell_id_input = QLineEdit()
    layout.addRow("Cell ID:", cell_id_input)

    # File selection
    file_layout = QHBoxLayout()
    file_input = QLineEdit()
    file_btn = QPushButton("Browse...")
    file_btn.clicked.connect(lambda: self.browse_data_file(file_input))
    file_layout.addWidget(file_input)
    file_layout.addWidget(file_btn)
    layout.addRow("Data File:", file_layout)

    metadata_input = QTextEdit()
    metadata_input.setMaximumHeight(60)
    layout.addRow("Metadata (JSON):", metadata_input)

    buttons = QDialogButtonBox(
        QDialogButtonBox.Ok | QDialogButtonBox.Cancel,
        Qt.Horizontal, dialog)
    buttons.accepted.connect(dialog.accept)
    buttons.rejected.connect(dialog.reject)
    layout.addRow(buttons)

    if dialog.exec_() == QDialog.Accepted:
        try:
            cell_id = cell_id_input.text().strip()
            data_file = file_input.text()
            metadata_text = metadata_input.toPlainText()
            metadata = json.loads(metadata_text) if metadata_text else None

            self.project.add_cell_to_treatment(treatment_name, cell_id, data_file, metadata)
            self.update_cell_list()

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to add cell: {str(e)}")
def create_project_tab(self):
    """Create the project management tab."""
    project_tab = QWidget()
    layout = QVBoxLayout()

    # Create and add project manager widget
    self.project_manager = ProjectManagerWidget()
    layout.addWidget(self.project_manager)

    project_tab.setLayout(layout)
    self.tabs.addTab(project_tab, "Project")
# Add to visualization tab
def setup_animation_controls(self):
    animation_group = QGroupBox("Track Animation")
    layout = QVBoxLayout()
    
    # Animation parameters
    params_layout = QFormLayout()
    
    self.animation_fps = QSpinBox()
    self.animation_fps.setRange(1, 60)
    self.animation_fps.setValue(10)
    params_layout.addRow("FPS:", self.animation_fps)
    
    self.trail_length = QSpinBox()
    self.trail_length.setRange(1, 50)
    self.trail_length.setValue(10)
    params_layout.addRow("Trail Length:", self.trail_length)
    
    # Animation controls
    btn_layout = QHBoxLayout()
    self.create_animation_btn = QPushButton("Create Animation")
    self.create_animation_btn.clicked.connect(self.create_track_animation)
    self.save_animation_btn = QPushButton("Save Animation")
    self.save_animation_btn.setEnabled(False)
    self.save_animation_btn.clicked.connect(self.save_track_animation)
    
    btn_layout.addWidget(self.create_animation_btn)
    btn_layout.addWidget(self.save_animation_btn)
    
    layout.addLayout(params_layout)
    layout.addLayout(btn_layout)
    animation_group.setLayout(layout)
    return animation_group
def create_interactive_visualization(self):
    """Create interactive track visualization with hover info."""
    if not self.tracks_df is None:
        try:
            # Create interactive plot
            html_output = create_interactive_plot(
                self.tracks_df,
                self.analysis_results.get('diffusion'),
                self.analysis_results.get('clustering'),
                pixel_size=self.project_settings['pixel_size']
            )
            
            # Display in QWebEngineView
            self.web_view.setHtml(html_output)
            self.web_view.show()
            
        except Exception as e:
            self.show_error_message("Visualization Error", str(e))
def add_correlation_analysis(self):
    """Add correlation matrix analysis to results."""
    if self.analysis_results:
        try:
            # Get numerical columns from results
            results_df = pd.DataFrame()
            for analysis_type, results in self.analysis_results.items():
                if isinstance(results, dict) and 'results_df' in results:
                    numeric_cols = results['results_df'].select_dtypes(include=[np.number]).columns
                    results_df = pd.concat([results_df, results['results_df'][numeric_cols]], axis=1)

            # Create correlation matrix
            fig = plot_correlation_matrix(
                results_df,
                title="Analysis Parameters Correlation",
                figsize=(10, 8)
            )

            # Add to results display
            self.add_figure_to_results(fig, "Correlation Analysis")

        except Exception as e:
            self.show_error_message("Correlation Analysis Error", str(e))
class UtilityWidget(QWidget):
    """Widget for utility functions and visualization tools."""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setup_ui()
        self.setup_connections()
        
    def setup_ui(self):
        """Setup the utility widget UI."""
        layout = QVBoxLayout(self)
        
        # Visualization Tools Group
        viz_group = QGroupBox("Visualization Tools")
        viz_layout = QVBoxLayout()
        
        # Track Animation
        anim_group = QGroupBox("Track Animation")
        anim_layout = QFormLayout()
        
        self.fps_spinbox = QSpinBox()
        self.fps_spinbox.setRange(1, 60)
        self.fps_spinbox.setValue(10)
        anim_layout.addRow("FPS:", self.fps_spinbox)
        
        self.trail_length_spinbox = QSpinBox()
        self.trail_length_spinbox.setRange(1, 50)
        self.trail_length_spinbox.setValue(10)
        anim_layout.addRow("Trail Length:", self.trail_length_spinbox)
        
        self.marker_size_spinbox = QSpinBox()
        self.marker_size_spinbox.setRange(1, 20)
        self.marker_size_spinbox.setValue(5)
        anim_layout.addRow("Marker Size:", self.marker_size_spinbox)
        
        self.show_background_check = QCheckBox("Show Background")
        self.show_background_check.setChecked(True)
        anim_layout.addRow("", self.show_background_check)
        
        self.create_animation_btn = QPushButton("Create Animation")
        anim_layout.addRow("", self.create_animation_btn)
        
        anim_group.setLayout(anim_layout)
        viz_layout.addWidget(anim_group)
        
        # Interactive Plot Controls
        interactive_group = QGroupBox("Interactive Plot")
        interactive_layout = QFormLayout()
        
        self.plot_type_combo = QComboBox()
        self.plot_type_combo.addItems([
            "Track Overview",
            "MSD Analysis",
            "Step Size Distribution",
            "Angular Distribution"
        ])
        interactive_layout.addRow("Plot Type:", self.plot_type_combo)
        
        self.create_interactive_btn = QPushButton("Create Interactive Plot")
        interactive_layout.addRow("", self.create_interactive_btn)
        
        interactive_group.setLayout(interactive_layout)
        viz_layout.addWidget(interactive_group)
        
        # Correlation Analysis
        correlation_group = QGroupBox("Correlation Analysis")
        correlation_layout = QFormLayout()
        
        self.correlation_features = QListWidget()
        self.correlation_features.setSelectionMode(QListWidget.MultiSelection)
        correlation_layout.addRow("Select Features:", self.correlation_features)
        
        self.create_correlation_btn = QPushButton("Create Correlation Matrix")
        correlation_layout.addRow("", self.create_correlation_btn)
        
        correlation_group.setLayout(correlation_layout)
        viz_layout.addWidget(correlation_group)
        
        viz_group.setLayout(viz_layout)
        layout.addWidget(viz_group)
        
        # Plot Display
        self.plot_canvas = MplCanvas(width=8, height=6, dpi=100)
        layout.addWidget(self.plot_canvas)
        
        # Export Controls
        export_layout = QHBoxLayout()
        self.export_plot_btn = QPushButton("Export Plot")
        self.export_animation_btn = QPushButton("Export Animation")
        export_layout.addWidget(self.export_plot_btn)
        export_layout.addWidget(self.export_animation_btn)
        layout.addLayout(export_layout)
        
    def setup_connections(self):
        """Setup widget connections."""
        self.create_animation_btn.clicked.connect(self.create_animation)
        self.create_interactive_btn.clicked.connect(self.create_interactive_plot)
        self.create_correlation_btn.clicked.connect(self.create_correlation_matrix)
        self.export_plot_btn.clicked.connect(self.export_current_plot)
        self.export_animation_btn.clicked.connect(self.export_animation)
        
    def create_animation(self):
        """Create track animation."""
        try:
            if not hasattr(self.parent(), 'tracks_df') or self.parent().tracks_df is None:
                raise ValueError("No tracks available for animation")
            
            params = {
                'fps': self.fps_spinbox.value(),
                'trail_length': self.trail_length_spinbox.value(),
                'marker_size': self.marker_size_spinbox.value(),
                'background': self.parent().image_stack[0] if self.show_background_check.isChecked() else None,
                'pixel_size': self.parent().project_settings['pixel_size']
            }
            
            # Create animation using utils function
            output_file = "track_animation.mp4"
            create_track_animation(
                self.parent().tracks_df,
                output_file,
                **params
            )
            
            QMessageBox.information(self, "Success", f"Animation saved to {output_file}")
            
        except Exception as e:
            QMessageBox.critical(self, "Error", str(e))
            
    def create_interactive_plot(self):
        """Create interactive plot."""
        try:
            if not hasattr(self.parent(), 'tracks_df') or self.parent().tracks_df is None:
                raise ValueError("No tracks available for plotting")
            
            plot_type = self.plot_type_combo.currentText()
            
            # Get additional data based on plot type
            diffusion_df = self.parent().analysis_results.get('diffusion', {}).get('results_df', None)
            cluster_df = self.parent().analysis_results.get('clustering', {}).get('results_df', None)
            
            # Create interactive plot
            html_output = create_interactive_plot(
                self.parent().tracks_df,
                diffusion_df,
                cluster_df,
                title=plot_type
            )
            
            # Save and open HTML file
            with open('interactive_plot.html', 'w') as f:
                f.write(html_output)
            
            # Open in default browser
            import webbrowser
            webbrowser.open('interactive_plot.html')
            
        except Exception as e:
            QMessageBox.critical(self, "Error", str(e))
            
    def create_correlation_matrix(self):
        """Create correlation matrix plot."""
        try:
            if not hasattr(self.parent(), 'tracks_df') or self.parent().tracks_df is None:
                raise ValueError("No tracks available for correlation analysis")
            
            # Get selected features
            selected_features = [item.text() for item in self.correlation_features.selectedItems()]
            if not selected_features:
                raise ValueError("Please select features for correlation analysis")
            
            # Create correlation matrix
            fig = plot_correlation_matrix(
                self.parent().tracks_df,
                columns=selected_features
            )
            
            # Update canvas
            self.plot_canvas.figure = fig
            self.plot_canvas.draw()
            
        except Exception as e:
            QMessageBox.critical(self, "Error", str(e))
            
    def export_current_plot(self):
        """Export current plot to file."""
        try:
            filename, _ = QFileDialog.getSaveFileName(
                self,
                "Save Plot",
                "",
                "PNG Files (*.png);;PDF Files (*.pdf);;SVG Files (*.svg)"
            )
            
            if filename:
                save_figure(
                    self.plot_canvas.figure,
                    filename,
                    dpi=300,
                    transparent=False
                )
                QMessageBox.information(self, "Success", f"Plot saved to {filename}")
                
        except Exception as e:
            QMessageBox.critical(self, "Error", str(e))
            
    def export_animation(self):
        """Export animation to file."""
        try:
            filename, _ = QFileDialog.getSaveFileName(
                self,
                "Save Animation",
                "",
                "MP4 Files (*.mp4);;GIF Files (*.gif)"
            )
            
            if filename:
                self.create_animation()  # This will save to the selected file
                QMessageBox.information(self, "Success", f"Animation saved to {filename}")
                
        except Exception as e:
            QMessageBox.critical(self, "Error", str(e))
def create_visualization_tab(self):
    """Create the visualization tab with utility tools."""
    visualization_tab = QWidget()
    layout = QVBoxLayout()
    
    # Add utility widget
    self.utility_widget = UtilityWidget(self)
    layout.addWidget(self.utility_widget)
    
    visualization_tab.setLayout(layout)
    self.tabs.addTab(visualization_tab, "Visualization")
class MultiChannelAnalysisWidget(QWidget):
    def __init__(self):
        super().__init__()
        self.setup_ui()
        self.setup_connections()

    def setup_ui(self):
        # Add UI elements for multi-channel analysis
        pass

    def setup_connections(self):
        # Connect signals and slots
        pass
def run_batch_process(self):
    try:
        # Get batch parameters
        batch_params = self.get_batch_parameters()

        # Create worker thread
        self.worker = WorkerThread(
            "batch_process",
            {
                "datasets": self.batch_datasets,
                "analysis_params": batch_params,
                "export_dir": self.batch_export_dir
            }
        )

        # Connect signals
        self.worker.progress_updated.connect(self.progress_bar.setValue)
        self.worker.operation_completed.connect(self.batch_completed)
        self.worker.error_occurred.connect(self.batch_error)

        # Start worker
        self.worker.start()

    except Exception as e:
        self.handle_batch_error(e)
def save_project_state(self):
    """Save complete project state including analysis results"""
    project_state = {
        'settings': self.project_settings,
        'analysis_results': self.analysis_results,
        'visualization_settings': self.get_visualization_settings(),
        'batch_settings': self.get_batch_settings()
    }

    # Save to file
    with open(self.project_file, 'w') as f:
        json.dump(project_state, f)
def export_analysis_results(self, analysis_type):
    """Export analysis results in standardized format"""
    try:
        results = self.analysis_results.get(analysis_type)
        if results is None:
            raise ValueError(f"No results available for {analysis_type}")

        # Get export format from user
        export_format = self.get_export_format()

        # Export based on format
        if export_format == 'csv':
            self.export_to_csv(results)
        elif export_format == 'excel':
            self.export_to_excel(results)
        elif export_format == 'hdf5':
            self.export_to_hdf5(results)

    except Exception as e:
        self.handle_export_error(e)  
class AnalysisExporter:
    """Handles standardized export of analysis results."""
    
    def __init__(self, parent=None):
        self.parent = parent
        self.supported_formats = {
            'csv': self.export_to_csv,
            'excel': self.export_to_excel,
            'hdf5': self.export_to_hdf5,
            'json': self.export_to_json
        }

    def export_results(self, analysis_type, results, file_path=None):
        """Export analysis results in specified format."""
        try:
            if file_path is None:
                file_path, export_format = self.get_export_path_and_format()
                if not file_path:  # User cancelled
                    return
            else:
                export_format = os.path.splitext(file_path)[1][1:].lower()

            if export_format not in self.supported_formats:
                raise ValueError(f"Unsupported export format: {export_format}")

            # Standardize results format
            formatted_results = self.format_results(analysis_type, results)
            
            # Perform export
            self.supported_formats[export_format](formatted_results, file_path)
            
            return True, f"Results exported successfully to {file_path}"

        except Exception as e:
            return False, f"Export failed: {str(e)}"

    def format_results(self, analysis_type, results):
        """Standardize results format based on analysis type."""
        if analysis_type == "diffusion":
            return self.format_diffusion_results(results)
        elif analysis_type == "active_transport":
            return self.format_active_transport_results(results)
        elif analysis_type == "boundary_crossing":
            return self.format_boundary_results(results)
        else:
            return results

    def export_to_csv(self, results, file_path):
        """Export results to CSV format."""
        if isinstance(results, dict):
            # Handle dictionary of DataFrames
            for key, df in results.items():
                base, ext = os.path.splitext(file_path)
                df_path = f"{base}_{key}{ext}"
                df.to_csv(df_path, index=False)
        else:
            # Single DataFrame
            results.to_csv(file_path, index=False)

    def export_to_excel(self, results, file_path):
        """Export results to Excel format."""
        if isinstance(results, dict):
            with pd.ExcelWriter(file_path) as writer:
                for key, df in results.items():
                    df.to_excel(writer, sheet_name=key, index=False)
        else:
            results.to_excel(file_path, index=False)

    def export_to_hdf5(self, results, file_path):
        """Export results to HDF5 format."""
        if isinstance(results, dict):
            with pd.HDFStore(file_path) as store:
                for key, df in results.items():
                    store[f'/{key}'] = df
        else:
            results.to_hdf(file_path, key='results', mode='w')

    def export_to_json(self, results, file_path):
        """Export results to JSON format."""
        if isinstance(results, dict):
            json_dict = {key: df.to_dict('records') for key, df in results.items()}
        else:
            json_dict = results.to_dict('records')
        
        with open(file_path, 'w') as f:
            json.dump(json_dict, f, indent=2)

    def get_export_path_and_format(self):
        """Get export file path and format from user."""
        file_dialog = QFileDialog()
        file_dialog.setAcceptMode(QFileDialog.AcceptSave)
        
        filters = "CSV (*.csv);;Excel (*.xlsx);;HDF5 (*.h5);;JSON (*.json)"
        file_path, selected_filter = file_dialog.getSaveFileName(
            self.parent,
            "Export Analysis Results",
            "",
            filters
        )
        
        if file_path:
            export_format = selected_filter.split('*')[1][1:-1].lstrip('.')
            return file_path, export_format
        
        return None, None
class ActiveTransportWidget(QWidget):
    """Widget for Active Transport Analysis controls and visualization."""
    
    # Signal for parameter updates
    parameters_changed = pyqtSignal(dict)
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setup_ui()
        self.setup_connections()
        self.current_results = None

    def setup_ui(self):
        """Setup the user interface."""
        layout = QVBoxLayout(self)
        
        # Parameters Group
        params_group = QGroupBox("Analysis Parameters")
        params_layout = QFormLayout()
        
        # Parameter controls
        self.min_alpha_spin = QDoubleSpinBox()
        self.min_alpha_spin.setRange(1.0, 2.0)
        self.min_alpha_spin.setValue(1.3)
        self.min_alpha_spin.setSingleStep(0.1)
        params_layout.addRow("Min. Alpha:", self.min_alpha_spin)
        
        self.min_velocity_spin = QDoubleSpinBox()
        self.min_velocity_spin.setRange(0.01, 10.0)
        self.min_velocity_spin.setValue(0.1)
        self.min_velocity_spin.setSuffix(" μm/s")
        params_layout.addRow("Min. Velocity:", self.min_velocity_spin)
        
        self.min_run_length_spin = QDoubleSpinBox()
        self.min_run_length_spin.setRange(0.1, 50.0)
        self.min_run_length_spin.setValue(0.5)
        self.min_run_length_spin.setSuffix(" μm")
        params_layout.addRow("Min. Run Length:", self.min_run_length_spin)
        
        self.min_duration_spin = QDoubleSpinBox()
        self.min_duration_spin.setRange(0.1, 10.0)
        self.min_duration_spin.setValue(0.5)
        self.min_duration_spin.setSuffix(" s")
        params_layout.addRow("Min. Duration:", self.min_duration_spin)
        
        params_group.setLayout(params_layout)
        layout.addWidget(params_group)
        
        # Visualization Options
        viz_group = QGroupBox("Visualization")
        viz_layout = QVBoxLayout()
        
        self.plot_type_combo = QComboBox()
        self.plot_type_combo.addItems([
            "Velocity Distribution",
            "Run Length Distribution",
            "Track Classification",
            "Spatial Distribution"
        ])
        viz_layout.addWidget(QLabel("Plot Type:"))
        viz_layout.addWidget(self.plot_type_combo)
        
        self.show_tracks_check = QCheckBox("Show Individual Tracks")
        self.show_tracks_check.setChecked(True)
        viz_layout.addWidget(self.show_tracks_check)
        
        viz_group.setLayout(viz_layout)
        layout.addWidget(viz_group)
        
        # Results Display
        self.results_tabs = QTabWidget()
        
        # Summary tab
        self.summary_text = QTextEdit()
        self.summary_text.setReadOnly(True)
        self.results_tabs.addTab(self.summary_text, "Summary")
        
        # Detailed results tab
        self.results_table = QTableWidget()
        self.results_tabs.addTab(self.results_table, "Detailed Results")
        
        # Plots tab
        self.plot_widget = MplCanvas(width=6, height=4, dpi=100)
        self.results_tabs.addTab(self.plot_widget, "Plots")
        
        layout.addWidget(self.results_tabs)

    def setup_connections(self):
        """Setup signal connections."""
        # Parameter change connections
        self.min_alpha_spin.valueChanged.connect(self.on_parameter_changed)
        self.min_velocity_spin.valueChanged.connect(self.on_parameter_changed)
        self.min_run_length_spin.valueChanged.connect(self.on_parameter_changed)
        self.min_duration_spin.valueChanged.connect(self.on_parameter_changed)
        
        # Visualization connections
        self.plot_type_combo.currentIndexChanged.connect(self.update_plot)
        self.show_tracks_check.stateChanged.connect(self.update_plot)

    def on_parameter_changed(self):
        """Handle parameter changes."""
        params = self.get_current_parameters()
        self.parameters_changed.emit(params)
        
        # Update analysis if auto-update is enabled
        if hasattr(self, 'auto_update') and self.auto_update:
            self.update_analysis()

    def get_current_parameters(self):
        """Get current analysis parameters."""
        return {
            'min_alpha': self.min_alpha_spin.value(),
            'min_velocity': self.min_velocity_spin.value(),
            'min_run_length': self.min_run_length_spin.value(),
            'min_duration': self.min_duration_spin.value()
        }

    def update_analysis(self, tracks_df=None):
        """Update analysis with current parameters."""
        if tracks_df is not None:
            self.tracks_df = tracks_df
            
        if not hasattr(self, 'tracks_df') or self.tracks_df is None:
            return
            
        try:
            # Get current parameters
            params = self.get_current_parameters()
            
            # Create analyzer
            analyzer = ActiveTransportAnalyzer(**params)
            
            # Run analysis
            self.current_results = analyzer.analyze(self.tracks_df)
            
            # Update display
            self.update_results_display()
            self.update_plot()
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Analysis failed: {str(e)}")

    def update_results_display(self):
        """Update results display."""
        if self.current_results is None:
            return
            
        # Update summary
        summary = self.generate_summary()
        self.summary_text.setText(summary)
        
        # Update results table
        self.update_results_table()

    def update_plot(self):
        """Update plot based on selected visualization type."""
        if self.current_results is None:
            return
            
        plot_type = self.plot_type_combo.currentText()
        
        try:
            self.plot_widget.axes.clear()
            
            if plot_type == "Velocity Distribution":
                self.plot_velocity_distribution()
            elif plot_type == "Run Length Distribution":
                self.plot_run_length_distribution()
            elif plot_type == "Track Classification":
                self.plot_track_classification()
            elif plot_type == "Spatial Distribution":
                self.plot_spatial_distribution()
                
            self.plot_widget.draw()
            
        except Exception as e:
            QMessageBox.warning(self, "Plot Error", f"Failed to update plot: {str(e)}")

    def plot_velocity_distribution(self):
        """Plot velocity distribution."""
        velocities = self.current_results['results_df']['velocity']
        self.plot_widget.axes.hist(velocities, bins=30, density=True)
        self.plot_widget.axes.set_xlabel('Velocity (μm/s)')
        self.plot_widget.axes.set_ylabel('Density')
        self.plot_widget.axes.set_title('Velocity Distribution')

    # Add other plotting methods as needed...
# In SPTAnalyzerGUI class

def setup_active_transport_analysis(self):
    """Setup active transport analysis integration."""
    # Create active transport widget
    self.active_transport_widget = ActiveTransportWidget()

    # Add to analysis tab
    self.analysis_tab_layout.addWidget(self.active_transport_widget)

    # Connect signals
    self.active_transport_widget.parameters_changed.connect(self.on_transport_params_changed)

def on_transport_params_changed(self, params):
    """Handle changes in transport analysis parameters."""
    if self.auto_update_analysis:
        self.run_transport_analysis(params)

def run_transport_analysis(self, params=None):
    """Run active transport analysis."""
    try:
        if self.tracks_df is None:
            raise ValueError("No tracks available for analysis")

        if params is None:
            params = self.active_transport_widget.get_current_parameters()

        # Show progress
        self.progress_bar.setVisible(True)
        self.setEnabled(False)

        # Create worker thread
        self.worker = WorkerThread(
            "analyze",
            {
                "analysis_type": "active_transport",
                "tracks_df": self.tracks_df,
                "analysis_params": params,
                "analysis_manager": self.analysis_manager
            }
        )

        # Connect signals
        self.worker.progress_updated.connect(self.progress_bar.setValue)
        self.worker.operation_completed.connect(self.transport_analysis_completed)
        self.worker.error_occurred.connect(self.analysis_error)

        # Start worker
        self.worker.start()

    except Exception as e:
        self.show_error_message("Analysis Error", str(e))
        logger.error(f"Transport analysis error: {e}", exc_info=True)

def transport_analysis_completed(self, results):
    """Handle completion of transport analysis."""
    try:
        # Store results
        self.analysis_results['active_transport'] = results

        # Update widget display
        self.active_transport_widget.update_analysis(results)

        # Reset UI
        self.progress_bar.setVisible(False)
        self.setEnabled(True)
        self.statusBar().showMessage("Active transport analysis completed", 3000)

    except Exception as e:
        self.show_error_message("Error", f"Failed to display results: {str(e)}")
        logger.error(f"Error displaying transport results: {e}", exc_info=True)

def export_results(self):
    """Export analysis results."""
    try:
        if not self.analysis_results:
            QMessageBox.warning(self, "Warning", "No analysis results to export")
            return

        file_path, selected_filter = QFileDialog.getSaveFileName(
            self, "Export Analysis Results",
            "",
            "HDF5 (*.h5);;JSON (*.json)"
        )

        if file_path:
            from utils.io import save_analysis_results
            save_analysis_results(self.analysis_results, file_path)
            self.statusBar().showMessage(f"Results exported to {file_path}", 3000)

    except Exception as e:
        QMessageBox.critical(self, "Error", f"Failed to export results: {str(e)}")
        logger.error(f"Results export error: {e}", exc_info=True)
    except Exception as e:
        QMessageBox.critical(self, "Error", f"Failed to export tracks: {str(e)}")
        logger.error(f"Track export error: {e}", exc_info=True)
def create_analysis_tab(self):
    # ... existing code ...

    # Add Multi-Channel Analysis to analysis types
    self.analysis_type.addItem("Multi-Channel Analysis")

    # Create multi-channel widget
    self.multi_channel_widget = MultiChannelAnalysisWidget()

    # Add to results tabs
    self.results_tabs.addTab(self.multi_channel_widget, "Multi-Channel")

    # ... rest of existing code ...

def update_analysis_parameters(self, index):
    """Update the analysis parameters display based on the selected analysis type."""
    analysis_type = self.analysis_type.currentText().lower()

    # Find the index of the corresponding widget in the stacked widget
    widget_to_show = self.analysis_widgets.get(analysis_type)
    if widget_to_show:
        self.params_stack.setCurrentWidget(widget_to_show)
        logger.debug(f"Switched to parameters for: {analysis_type}")
    else:
        logger.warning(f"No widget found for analysis type: {analysis_type}")
        # Optionally hide the parameter group box or show a default message
        # self.params_stack.setCurrentWidget(self.default_params_widget) # Need a default widget if types are missing
def run_diffusion_analysis(self):
    """Run diffusion analysis with current parameters."""
    try:
        if self.tracks_df is None:
            raise ValueError("No tracks available for analysis")

        # Get parameters from GUI
        params = {
            'pixel_size': self.project_settings['pixel_size'],
            'frame_interval': self.project_settings['frame_interval'],
            'max_lag': self.diffusion_widget.max_lag_spinbox.value(),
            'min_track_length': self.diffusion_widget.min_track_length.value(),
            'max_fit_points': self.diffusion_widget.max_fit_points.value(),
            'model_type': self.diffusion_widget.model_selection.currentText().lower().replace(" ", "_")
        }

        # Show progress bar
        self.progress_bar.setVisible(True)
        self.setEnabled(False)

        # Create worker thread
        self.worker = WorkerThread(
            "analyze",
            {
                "analysis_type": "diffusion",
                "tracks_df": self.tracks_df,
                "analysis_params": params,
                "analysis_manager": self.analysis_manager
            }
        )

        # Connect signals
        self.worker.progress_updated.connect(self.progress_bar.setValue)
        self.worker.operation_completed.connect(self.diffusion_analysis_completed)
        self.worker.error_occurred.connect(self.analysis_error)

        # Start worker
        self.worker.start()

    except Exception as e:
        self.show_error_message("Analysis Error", str(e))
        logger.error(f"Diffusion analysis error: {e}", exc_info=True)

def diffusion_analysis_completed(self, results):
    """Handle completion of diffusion analysis."""
    try:
        # Store results
        self.analysis_results['diffusion'] = results

        # Update results display
        self.display_diffusion_results(results)

        # Update plot
        self.update_diffusion_plot()

        # Reset UI
        self.progress_bar.setVisible(False)
        self.setEnabled(True)
        self.statusBar().showMessage("Diffusion analysis completed", 3000)

    except Exception as e:
        self.show_error_message("Error", f"Failed to display results: {str(e)}")
        logger.error(f"Error displaying diffusion results: {e}", exc_info=True)

def update_diffusion_plot(self):
    """Update the diffusion analysis plot based on selected plot type."""
    if 'diffusion' not in self.analysis_results:
        return

    plot_type = self.diffusion_widget.plot_type.currentText()
    results = self.analysis_results['diffusion']

    try:
        fig = self.results_canvas.figure
        fig.clear()
        ax = fig.add_subplot(111)

        if plot_type == "MSD Curves":
            self.plot_msd_curves(ax, results)
        elif plot_type == "Diffusion Coefficient Map":
            self.plot_diffusion_map(ax, results)
        elif plot_type == "Alpha Distribution":
            self.plot_alpha_distribution(ax, results)
        elif plot_type == "Model Comparison":
            self.plot_model_comparison(ax, results)

        fig.tight_layout()
        self.results_canvas.draw()

    except Exception as e:
        self.show_error_message("Plot Error", str(e))
        logger.error(f"Error updating diffusion plot: {e}", exc_info=True)

def display_diffusion_results(self, results: Dict):
    """Display diffusion analysis results"""
    # Update summary text
    results_df = results['results_df']
    summary = (
        f"Diffusion Analysis Results:\n\n"
        f"Number of tracks analyzed: {len(results_df)}\n"
        f"Mean D: {results_df['D'].mean():.3f} μm²/s\n"
        f"Mean α: {results_df['alpha'].mean():.3f}\n"
        f"Tracks with α > 1: {sum(results_df['alpha'] > 1)}\n"
        f"Tracks with α < 1: {sum(results_df['alpha'] < 1)}\n"
    )
    self.results_summary.setText(summary)

    # Update results table
    self.update_results_table(results_df)

    # Create plots
    plotter = AnalysisPlotter()

    # MSD curves plot
    msd_fig = plotter.plot_msd_curves(results)
    self.results_canvas.figure = msd_fig
    self.results_canvas.draw()

    # Diffusion map
    if self.image_stack is not None:
        diff_map_fig = plotter.plot_diffusion_map(
            self.tracks_df,
            results,
            np.max(self.image_stack, axis=0)
        )
        self.viz_canvas.figure = diff_map_fig
        self.viz_canvas.draw()

def run_active_transport_analysis(self):
    """Run active transport analysis on tracks"""
    try:
        # Get parameters from GUI
        params = {
            'pixel_size': self.project_settings['pixel_size'],
            'frame_interval': self.project_settings['frame_interval'],
            'min_alpha': self.min_superdiffusive_alpha.value(),
            'min_velocity': 0.1,  # μm/s
            'min_run_length': 0.5,  # μm
            'min_duration': 0.5  # seconds
        }

        # Run analysis
        results = self.analysis_manager.run_analysis(
            'active_transport',
            self.tracks_df,
            params
        )

        # Update results display
        self.display_active_transport_results(results)

    except Exception as e:
        logger.error(f"Active transport analysis error: {str(e)}", exc_info=True)
        QMessageBox.critical(self, "Error", f"Analysis failed: {str(e)}")     
def connect_image_processing_tab_widgets(self):
    """Connect widgets in the image processing tab to their slots."""
    image_tab = self.tabs.widget(1)  # Image processing tab is the second tab

    # Frame navigation
    self.frame_slider.valueChanged.connect(self.update_display_frame)

    # Find buttons for frame navigation
    buttons = image_tab.findChildren(QPushButton)
    for button in buttons:
        if button.text() == "Previous":
            button.clicked.connect(lambda: self.frame_slider.setValue(self.frame_slider.value() - 1))
        elif button.text() == "Next":
            button.clicked.connect(lambda: self.frame_slider.setValue(self.frame_slider.value() + 1))
        elif button.text() == "Import Image Stack":
            button.clicked.connect(self.import_image_stack)
        elif button.text() == "Apply Processing":
            button.clicked.connect(self.apply_image_processing)
        elif button.text() == "Reset Processing":
            button.clicked.connect(self.reset_image_processing)
        elif button.text() == "Perform Segmentation":
            button.clicked.connect(self.perform_segmentation)

    # Connect segmentation checkbox
    self.show_segmentation.stateChanged.connect(self.update_display_frame)

def connect_tracking_tab_widgets(self):
    """Connect widgets in the tracking tab to their slots."""
    tracking_tab = self.tabs.widget(2)  # Tracking tab is the third tab

    # Frame navigation
    self.tracking_frame_slider.valueChanged.connect(self.update_tracking_frame)

    # Find buttons for frame navigation
    buttons = tracking_tab.findChildren(QPushButton)
    for button in buttons:
        if button.text() == "Previous":
            button.clicked.connect(lambda: self.tracking_frame_slider.setValue(self.tracking_frame_slider.value() - 1))
        elif button.text() == "Next":
            button.clicked.connect(lambda: self.tracking_frame_slider.setValue(self.tracking_frame_slider.value() + 1))
        elif button.text() == "Detect Particles":
            button.clicked.connect(self.detect_particles)
        elif button.text() == "Link Tracks":
            button.clicked.connect(self.link_tracks)
        elif button.text() == "Filter Tracks":
            button.clicked.connect(self.filter_tracks)
        elif button.text() == "Export Tracks":
            button.clicked.connect(self.export_tracks)

    # Display options
    self.show_detections.stateChanged.connect(self.update_tracking_display)
    self.show_tracks.stateChanged.connect(self.update_tracking_display)
    self.track_history.valueChanged.connect(self.update_tracking_display)
def setup_detector_connections(self):
    """Setup connections for particle detection in the Detection tab"""
    # Connect method selection
    self.detection_method_combo.currentTextChanged.connect(self.update_detector_params)

    # Connect parameter update widgets
    self.threshold_spinbox.valueChanged.connect(self.update_detector_params)
    self.threshold_relative_check.stateChanged.connect(self.update_detector_params)
    self.min_distance_spinbox.valueChanged.connect(self.update_detector_params)
    self.diameter_spinbox.valueChanged.connect(self.update_detector_params)
    self.subpixel_check.stateChanged.connect(self.update_detector_params)
    self.subpixel_method_combo.currentTextChanged.connect(self.update_detector_params)

    # Connect wavelet-specific parameters
    self.wavelet_type_combo.currentTextChanged.connect(self.update_detector_params)
    self.wavelet_levels_spinbox.valueChanged.connect(self.update_detector_params)
    self.wavelet_method_combo.currentTextChanged.connect(self.update_detector_params)

    # Connect detection button
    self.detect_btn.clicked.connect(self.run_detection)

    # Connect preview update
    self.preview_check.stateChanged.connect(self.update_detection_preview)

def update_detector_params(self):
    """Update particle detector parameters based on GUI inputs"""
    method = self.detection_method_combo.currentText().lower()

    # Update parameter visibility based on method
    self.update_parameter_visibility(method)

    # Store parameters
    self.detector_params = {
        'method': method,
        'threshold': self.threshold_spinbox.value(),
        'threshold_is_relative': self.threshold_relative_check.isChecked(),
        'min_distance': self.min_distance_spinbox.value(),
        'diameter': self.diameter_spinbox.value(),
        'subpixel': self.subpixel_check.isChecked(),
        'subpixel_method': self.subpixel_method_combo.currentText().lower()
    }

    # Add wavelet-specific parameters if applicable
    if method == 'wavelet':
        self.detector_params.update({
            'wavelet_type': self.wavelet_type_combo.currentText(),
            'wavelet_levels': self.wavelet_levels_spinbox.value(),
            'wavelet_enhancement_method': self.wavelet_method_combo.currentText().lower()
        })

    # Update preview if enabled
    if self.preview_check.isChecked():
        self.update_detection_preview()

def update_parameter_visibility(self, method):
    """Update visibility of parameter widgets based on detection method"""
    # Show/hide wavelet-specific parameters
    wavelet_widgets = [self.wavelet_type_label, self.wavelet_type_combo,
                      self.wavelet_levels_label, self.wavelet_levels_spinbox,
                      self.wavelet_method_label, self.wavelet_method_combo]

    for widget in wavelet_widgets:
        widget.setVisible(method == 'wavelet')

def run_detection(self):
    """Execute particle detection on current image"""
    try:
        if not hasattr(self, 'current_image') or self.current_image is None:
            self.show_error_message("No Image", "Please load an image first.")
            return

        # Create detector with current parameters
        detector = ParticleDetector(**self.detector_params)

        # Run detection
        positions = detector.detect(self.current_image)

        # Store results
        self.detection_results = positions

        # Update results display
        self.update_detection_results(positions)

        # Update status
        self.status_bar.showMessage(f"Detected {len(positions)} particles.")

    except Exception as e:
        self.show_error_message("Detection Error", str(e))
        logging.error(f"Particle detection error: {e}", exc_info=True)

def update_detection_results(self, positions):
    """Update the display of detection results"""
    # Update results table
    self.detection_results_table.setRowCount(0)
    headers = ["Particle ID", "Y Position", "X Position"]
    self.detection_results_table.setColumnCount(len(headers))
    self.detection_results_table.setHorizontalHeaderLabels(headers)

    for i, (y, x) in enumerate(positions):
        row = self.detection_results_table.rowCount()
        self.detection_results_table.insertRow(row)

        self.detection_results_table.setItem(row, 0, QTableWidgetItem(str(i+1)))
        self.detection_results_table.setItem(row, 1, QTableWidgetItem(f"{y:.2f}"))
        self.detection_results_table.setItem(row, 2, QTableWidgetItem(f"{x:.2f}"))

    # Update visualization
    self.plot_detection_results(positions)

def plot_detection_results(self, positions):
    """Plot detection results overlay on the image"""
    try:
        fig, ax = plt.subplots()

        # Display image
        ax.imshow(self.current_image, cmap='gray')

        # Plot detected positions
        if len(positions) > 0:
            ax.plot(positions[:, 1], positions[:, 0], 'r+', markersize=10,
                   markeredgewidth=2, label='Detected particles')

        ax.set_title(f'Detected Particles: {len(positions)}')
        ax.legend()

        # Update the plot widget
        self.detection_plot_widget.canvas.figure = fig
        self.detection_plot_widget.canvas.draw()

    except Exception as e:
        logging.error(f"Error plotting detection results: {e}", exc_info=True)

def update_detection_preview(self):
    """Update real-time preview of detection results"""
    if not self.preview_check.isChecked():
        return

    try:
        if hasattr(self, 'current_image') and self.current_image is not None:
            # Create detector with current parameters
            detector = ParticleDetector(**self.detector_params)

            # Run detection on current ROI or full image
            if hasattr(self, 'current_roi'):
                preview_image = self.current_image[self.current_roi]
            else:
                preview_image = self.current_image

            # Run detection
            positions = detector.detect(preview_image)

            # Update preview display
            self.plot_detection_results(positions)

    except Exception as e:
        logging.error(f"Error updating detection preview: {e}", exc_info=True)
def setup_detector_ui(self):
    """Setup UI elements for particle detection"""
    # Create detector group box
    detector_group = QGroupBox("Particle Detection")
    layout = QVBoxLayout()

    # Method selection
    method_layout = QFormLayout()
    self.detection_method_combo = QComboBox()
    self.detection_method_combo.addItems(['Gaussian', 'Laplacian', 'DoH', 'LoG', 'Wavelet'])
    method_layout.addRow("Detection Method:", self.detection_method_combo)

    # Basic parameters
    param_layout = QFormLayout()

    self.threshold_spinbox = QDoubleSpinBox()
    self.threshold_spinbox.setRange(0, 1000)
    self.threshold_spinbox.setValue(0.5)
    self.threshold_spinbox.setSingleStep(0.1)

    self.threshold_relative_check = QCheckBox("Relative")
    self.threshold_relative_check.setChecked(True)

    threshold_layout = QHBoxLayout()
    threshold_layout.addWidget(self.threshold_spinbox)
    threshold_layout.addWidget(self.threshold_relative_check)
    param_layout.addRow("Threshold:", threshold_layout)

    self.min_distance_spinbox = QSpinBox()
    self.min_distance_spinbox.setRange(1, 100)
    self.min_distance_spinbox.setValue(5)
    param_layout.addRow("Min Distance:", self.min_distance_spinbox)

    self.diameter_spinbox = QSpinBox()
    self.diameter_spinbox.setRange(3, 100)
    self.diameter_spinbox.setValue(7)
    param_layout.addRow("Particle Diameter:", self.diameter_spinbox)

    # Subpixel refinement
    subpixel_layout = QHBoxLayout()
    self.subpixel_check = QCheckBox("Enable")
    self.subpixel_check.setChecked(True)
    self.subpixel_method_combo = QComboBox()
    self.subpixel_method_combo.addItems(['CoM', 'Gaussian'])
    subpixel_layout.addWidget(self.subpixel_check)
    subpixel_layout.addWidget(self.subpixel_method_combo)
    param_layout.addRow("Subpixel:", subpixel_layout)

    # Wavelet-specific parameters
    wavelet_layout = QFormLayout()

    self.wavelet_type_label = QLabel("Wavelet Type:")
    self.wavelet_type_combo = QComboBox()
    self.wavelet_type_combo.addItems(['db4', 'sym5', 'haar'])
    wavelet_layout.addRow(self.wavelet_type_label, self.wavelet_type_combo)

    self.wavelet_levels_label = QLabel("Wavelet Levels:")
    self.wavelet_levels_spinbox = QSpinBox()
    self.wavelet_levels_spinbox.setRange(1, 10)
    self.wavelet_levels_spinbox.setValue(3)
    wavelet_layout.addRow(self.wavelet_levels_label, self.wavelet_levels_spinbox)

    self.wavelet_method_label = QLabel("Enhancement Method:")
    self.wavelet_method_combo = QComboBox()
    self.wavelet_method_combo.addItems(['Residual', 'Direct'])
    wavelet_layout.addRow(self.wavelet_method_label, self.wavelet_method_combo)

    # Control buttons
    button_layout = QHBoxLayout()
    self.detect_btn = QPushButton("Detect Particles")
    self.preview_check = QCheckBox("Live Preview")
    button_layout.addWidget(self.detect_btn)
    button_layout.addWidget(self.preview_check)

    # Results display
    self.detection_results_table = QTableWidget()
    self.detection_plot_widget = MatplotlibWidget()

    # Add all to layout
    layout.addLayout(method_layout)
    layout.addLayout(param_layout)
    layout.addLayout(wavelet_layout)
    layout.addLayout(button_layout)
    layout.addWidget(self.detection_results_table)
    layout.addWidget(self.detection_plot_widget)

    detector_group.setLayout(layout)

    # Add to detection tab
    self.detection_tab_layout.addWidget(detector_group)

    # Initialize parameter visibility
    self.update_parameter_visibility('gaussian')            
def setup_matplotlib_widgets(self):
    """Set up all Matplotlib widgets and their configurations."""
    # Create matplotlib canvases with custom configurations
    self.image_canvas = MplCanvas(width=6, height=5, dpi=100)
    self.tracking_canvas = MplCanvas(width=6, height=5, dpi=100)
    self.results_canvas = MplCanvas(width=6, height=5, dpi=100)
    self.viz_canvas = MplCanvas(width=6, height=5, dpi=100)

    # Enable interactive mode for real-time updates
    plt.ion()

    # Configure each canvas for animations
    for canvas in [self.image_canvas, self.tracking_canvas,
                  self.results_canvas, self.viz_canvas]:
        canvas.fig.patch.set_facecolor('#f0f0f0')  # Match Qt background
        canvas.axes.set_facecolor('#ffffff')
        canvas.fig.tight_layout()

    # Set up animation objects
    self.tracking_animation = None
    self.image_animation = None

class MplCanvas(FigureCanvas):
    """Enhanced canvas for matplotlib figures in the GUI."""

    def __init__(self, width=5, height=4, dpi=100):
        # Create figure with subplots
        self.fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = self.fig.add_subplot(111)

        # Initialize FigureCanvas
        super(MplCanvas, self).__init__(self.fig)

        # Enable mouse wheel zooming
        self.scroll_factor = 1.2
        self.connect_zoom_events()

        # Enable pan/zoom with mouse drag
        self.pan_enabled = False
        self.connect_pan_events()

        # Store view history for navigation
        self.view_history = []
        self.current_view = 0

    def connect_zoom_events(self):
        """Connect mouse wheel events for zooming."""
        self.mpl_connect('scroll_event', self.zoom)

    def connect_pan_events(self):
        """Connect mouse events for panning."""
        self.mpl_connect('button_press_event', self.on_press)
        self.mpl_connect('button_release_event', self.on_release)
        self.mpl_connect('motion_notify_event', self.on_motion)

    def zoom(self, event):
        """Handle mouse wheel zoom events."""
        if event.inaxes:
            cur_xlim = self.axes.get_xlim()
            cur_ylim = self.axes.get_ylim()

            xdata = event.xdata
            ydata = event.ydata

            if event.button == 'up':
                scale_factor = 1/self.scroll_factor
            else:
                scale_factor = self.scroll_factor

            new_width = (cur_xlim[1] - cur_xlim[0]) * scale_factor
            new_height = (cur_ylim[1] - cur_ylim[0]) * scale_factor

            relx = (cur_xlim[1] - xdata)/(cur_xlim[1] - cur_xlim[0])
            rely = (cur_ylim[1] - ydata)/(cur_ylim[1] - cur_ylim[0])

            self.axes.set_xlim([xdata - new_width * (1-relx), xdata + new_width * relx])
            self.axes.set_ylim([ydata - new_height * (1-rely), ydata + new_height * rely])

            self.draw()

    def on_press(self, event):
        """Handle mouse button press for panning."""
        if event.button == 3:  # Right mouse button
            self.pan_enabled = True
            self._pan_start = (event.xdata, event.ydata)

    def on_release(self, event):
        """Handle mouse button release for panning."""
        if event.button == 3:
            self.pan_enabled = False

    def on_motion(self, event):
        """Handle mouse motion for panning."""
        if self.pan_enabled and event.inaxes:
            dx = event.xdata - self._pan_start[0]
            dy = event.ydata - self._pan_start[1]

            cur_xlim = self.axes.get_xlim()
            cur_ylim = self.axes.get_ylim()

            self.axes.set_xlim(cur_xlim - dx)
            self.axes.set_ylim(cur_ylim - dy)

            self.draw()

def setup_tracking_animation(self):
    """Set up animation for particle tracking visualization."""
    if self.image_stack is None or self.tracks_df is None:
        return None

    fig = self.tracking_canvas.fig
    ax = self.tracking_canvas.axes

    # Initialize plot elements
    background = ax.imshow(self.image_stack[0], cmap='gray')
    particles_scatter = ax.scatter([], [], c='r', s=50)
    tracks_lines = []

    def init():
        """Initialize animation."""
        ax.clear()
        ax.imshow(self.image_stack[0], cmap='gray')
        return [background, particles_scatter] + tracks_lines

    def update(frame):
        """Update animation frame."""
        # Update background image
        background.set_array(self.image_stack[frame])

        # Update particle positions
        if self.show_detections.isChecked() and frame < len(self.detections):
            frame_detections = self.detections[frame]
            if len(frame_detections) > 0:
                particles_scatter.set_offsets(frame_detections[:, :2])
            else:
                particles_scatter.set_offsets(np.empty((0, 2)))

        # Update tracks
        if self.show_tracks.isChecked():
            history = self.track_history.value()
            current_tracks = self.tracks_df[
                (self.tracks_df['frame'] >= max(0, frame - history)) &
                (self.tracks_df['frame'] <= frame)
            ]

            # Clear previous tracks
            for line in tracks_lines:
                line.remove()
            tracks_lines.clear()

            # Draw new tracks
            for track_id in current_tracks['track_id'].unique():
                track = current_tracks[current_tracks['track_id'] == track_id]
                line, = ax.plot(track['x'], track['y'], 'b-', linewidth=1)
                tracks_lines.append(line)

        return [background, particles_scatter] + tracks_lines

    # Create animation
    anim = FuncAnimation(
        fig, update, init_func=init,
        frames=len(self.image_stack), interval=50,
        blit=True, repeat=True
    )

    return anim

def setup_diffusion_map_animation(self):
    """Set up animation for diffusion coefficient map visualization."""
    if 'diffusion' not in self.analysis_results:
        return None

    fig = self.viz_canvas.fig
    ax = self.viz_canvas.axes

    # Get diffusion results
    diff_results = self.analysis_results['diffusion']['results_df']

    # Create colormap
    colormap = self.diff_colormap.currentText()
    norm = mcolors.LogNorm(
        vmin=self.diff_min.value(),
        vmax=self.diff_max.value()
    )

    # Initialize plot elements
    if self.show_diff_background.isChecked() and self.image_stack is not None:
        background = ax.imshow(np.max(self.image_stack, axis=0), cmap='gray')
    else:
        background = None

    scatter = ax.scatter([], [], c=[], cmap=colormap, norm=norm)

    def init():
        """Initialize animation."""
        ax.clear()
        if background is not None:
            ax.imshow(np.max(self.image_stack, axis=0), cmap='gray')
        return [scatter]

    def update(frame):
        """Update animation frame."""
        # Get positions and diffusion coefficients for current frame
        frame_data = self.tracks_df[self.tracks_df['frame'] == frame]
        frame_diff = pd.merge(
            frame_data,
            diff_results[['track_id', 'D']],
            on='track_id', how='left'
        )

        # Update scatter plot
        scatter.set_offsets(frame_diff[['x', 'y']])
        scatter.set_array(frame_diff['D'])

        return [scatter]

    # Create animation
    anim = FuncAnimation(
        fig, update, init_func=init,
        frames=len(self.image_stack), interval=50,
        blit=True, repeat=True
    )

    return anim

def create_msd_plot(self):
    """Create MSD plot with interactive features."""
    if 'diffusion' not in self.analysis_results:
        return

    ax = self.results_canvas.axes
    ax.clear()

    # Get MSD data
    diff_results = self.analysis_results['diffusion']['results_df']

    # Plot individual MSDs with interactive selection
    for track_id in diff_results['track_id'].unique():
        track_data = diff_results[diff_results['track_id'] == track_id]
        line, = ax.plot(track_data['lag_time'], track_data['msd'],
                       alpha=0.3, picker=5)
        line.track_id = track_id

    ax.set_xlabel('Time Lag (s)')
    ax.set_ylabel('MSD (μm²)')
    ax.set_xscale('log')
    ax.set_yscale('log')

    def on_pick(event):
        """Handle picking events for MSD curves."""
        line = event.artist
        track_id = line.track_id

        # Highlight selected curve
        for l in ax.get_lines():
            l.set_alpha(0.3)
        line.set_alpha(1.0)

        # Show track info
        track_data = diff_results[diff_results['track_id'] == track_id]
        info_text = f"Track {track_id}:\n"
        info_text += f"D = {track_data['D'].iloc[0]:.3f} μm²/s\n"
        info_text += f"α = {track_data['alpha'].iloc[0]:.3f}"

        # Update info display
        self.results_summary.setText(info_text)

        self.results_canvas.draw()

    self.results_canvas.mpl_connect('pick_event', on_pick)

def create_interactive_track_plot(self):
    """Create interactive track plot with hovering and selection."""
    if self.tracks_df is None:
        return

    ax = self.viz_canvas.axes
    ax.clear()

    # Plot tracks
    unique_tracks = self.tracks_df['track_id'].unique()
    colors = plt.cm.viridis(np.linspace(0, 1, len(unique_tracks)))

    # Create track collection
    track_lines = []
    for i, track_id in enumerate(unique_tracks):
        track = self.tracks_df[self.tracks_df['track_id'] == track_id]
        line, = ax.plot(track['x'], track['y'], color=colors[i],
                       alpha=0.5, picker=5)
        line.track_id = track_id
        track_lines.append(line)

    # Add background image if available
    if self.image_stack is not None and self.show_background.isChecked():
        ax.imshow(np.max(self.image_stack, axis=0), cmap='gray')

    def on_hover(event):
        """Handle hover events."""
        if event.inaxes != ax:
            return

        for line in track_lines:
            contains, _ = line.contains(event)
            if contains:
                line.set_linewidth(2)
                line.set_alpha(1.0)

                # Show track info
                track_id = line.track_id
                track = self.tracks_df[self.tracks_df['track_id'] == track_id]
                info = f"Track {track_id}\n"
                info += f"Length: {len(track)} frames\n"
                info += f"Start: ({track['x'].iloc[0]:.1f}, {track['y'].iloc[0]:.1f})\n"
                info += f"End: ({track['x'].iloc[-1]:.1f}, {track['y'].iloc[-1]:.1f})"

                # Update status bar
                self.statusBar().showMessage(info)
            else:
                line.set_linewidth(1)
                line.set_alpha(0.5)

        self.viz_canvas.draw()

    def on_pick(event):
        """Handle picking events."""
        line = event.artist
        track_id = line.track_id

        # Highlight selected track
        for l in track_lines:
            if l.track_id == track_id:
                l.set_linewidth(3)
                l.set_alpha(1.0)
            else:
                l.set_linewidth(1)
                l.set_alpha(0.3)

        self.viz_canvas.draw()

        # Show detailed track analysis
        self.show_track_analysis(track_id)

    self.viz_canvas.mpl_connect('motion_notify_event', on_hover)
    self.viz_canvas.mpl_connect('pick_event', on_pick)

def show_track_analysis(self, track_id):
    """Show detailed analysis for a selected track."""
    if 'diffusion' not in self.analysis_results:
        return

    # Get track data
    track = self.tracks_df[self.tracks_df['track_id'] == track_id]
    diff_results = self.analysis_results['diffusion']['results_df']
    track_diff = diff_results[diff_results['track_id'] == track_id]

    # Create analysis dialog
    dialog = QDialog(self)
    dialog.setWindowTitle(f"Track {track_id} Analysis")
    layout = QVBoxLayout()

    # Create matplotlib canvas for track visualization
    canvas = MplCanvas(width=6, height=4, dpi=100)
    fig = canvas.fig

    # Create subplots
    gs = fig.add_gridspec(2, 2)
    ax1 = fig.add_subplot(gs[0, 0])  # Track trajectory
    ax2 = fig.add_subplot(gs[0, 1])  # MSD plot
    ax3 = fig.add_subplot(gs[1, 0])  # Step size distribution
    ax4 = fig.add_subplot(gs[1, 1])  # Angle distribution

    # Plot track trajectory
    ax1.plot(track['x'], track['y'], 'b-')
    ax1.scatter(track['x'].iloc[0], track['y'].iloc[0], c='g', label='Start')
    ax1.scatter(track['x'].iloc[-1], track['y'].iloc[-1], c='r', label='End')
    ax1.set_title('Trajectory')
    ax1.legend()

    # Plot MSD
    ax2.plot(track_diff['lag_time'], track_diff['msd'], 'b.-')
    ax2.set_xscale('log')
    ax2.set_yscale('log')
    ax2.set_title('MSD')

    # Plot step size distribution
    steps = np.sqrt(np.diff(track['x'])**2 + np.diff(track['y'])**2)
    ax3.hist(steps, bins=20, density=True)
    ax3.set_title('Step Size Distribution')

    # Plot angle distribution
    dx = np.diff(track['x'])
    dy = np.diff(track['y'])
    angles = np.arctan2(dy, dx)
    ax4.hist(angles, bins=36, density=True)
    ax4.set_title('Angle Distribution')

    fig.tight_layout()

    # Add canvas to dialog
    layout.addWidget(canvas)

    # Add text information
    info_text = QTextEdit()
    info_text.setReadOnly(True)
    info_text.setText(
        f"Track Analysis Results:\n\n"
        f"Number of positions: {len(track)}\n"
        f"Track duration: {len(track) * self.project_settings['frame_interval']:.2f} s\n"
        f"Total distance: {np.sum(steps):.2f} μm\n"
        f"Net displacement: {np.sqrt((track['x'].iloc[-1] - track['x'].iloc[0])**2 + (track['y'].iloc[-1] - track['y'].iloc[0])**2):.2f} μm\n"
        f"Diffusion coefficient: {track_diff['D'].iloc[0]:.3f} μm²/s\n"
        f"Anomalous exponent: {track_diff['alpha'].iloc[0]:.3f}\n"
    )
    layout.addWidget(info_text)

    # Add close button
    close_btn = QPushButton("Close")
    close_btn.clicked.connect(dialog.close)
    layout.addWidget(close_btn)

    dialog.setLayout(layout)
    dialog.exec_()
def connect_analysis_tab_widgets(self):
    """Connect widgets in the analysis tab to their slots."""
    analysis_tab = self.tabs.widget(3)  # Analysis tab is the fourth tab

    # Analysis type selection
    self.analysis_type.currentIndexChanged.connect(self.update_analysis_parameters)

    # Find run analysis button
    buttons = analysis_tab.findChildren(QPushButton)
    for button in buttons:
        if button.text() == "Run Analysis":
            button.clicked.connect(self.run_analysis)
        elif button.text() == "Export Results":
            button.clicked.connect(self.export_results)

def connect_visualization_tab_widgets(self):
    """Connect widgets in the visualization tab to their slots."""
    visualization_tab = self.tabs.widget(4)  # Visualization tab is the fifth tab

    # Visualization type selection
    self.viz_type.currentIndexChanged.connect(self.update_visualization_parameters)

    # Find buttons
    buttons = visualization_tab.findChildren(QPushButton)
    for button in buttons:
        if button.text() == "Generate Visualization":
            button.clicked.connect(self.generate_visualization)
        elif button.text() == "Export Visualization":
            button.clicked.connect(self.export_figure)

def connect_batch_tab_widgets(self):
    """Connect widgets in the batch processing tab to their slots."""
    batch_tab = self.tabs.widget(5)  # Batch tab is the sixth tab

    # Find buttons
    buttons = batch_tab.findChildren(QPushButton)
    for button in buttons:
        if button.text() == "Add Dataset":
            button.clicked.connect(self.add_batch_dataset)
        elif button.text() == "Remove Dataset":
            button.clicked.connect(self.remove_batch_dataset)
        elif button.text() == "Browse...":
            button.clicked.connect(self.select_batch_export_dir)
        elif button.text() == "Run Batch Process":
            button.clicked.connect(self.run_batch_process)

def handle_tab_change(self, index):
    """Handle tab change events."""
    tab_names = ["Project", "Image Processing", "Tracking", "Analysis", "Visualization", "Batch Processing"]
    if 0 <= index < len(tab_names):
        self.statusBar().showMessage(f"Switched to {tab_names[index]} tab")

def perform_segmentation(self):
    """Perform image segmentation."""
    if self.image_stack is None:
        QMessageBox.warning(self, "Warning", "No image stack loaded")
        return

    try:
        # Get segmentation parameters
        seg_method = self.seg_method.currentText().lower()

        # Implement segmentation logic here
        QMessageBox.information(
            self, "Segmentation",
            f"Segmentation using {seg_method} method would be performed here."
        )

        # Update display if show segmentation is checked
        if self.show_segmentation.isChecked():
            self.update_display_frame(self.current_frame)

    except Exception as e:
        logger.error(f"Error in segmentation: {str(e)}", exc_info=True)
        QMessageBox.critical(self, "Error", f"Segmentation failed: {str(e)}")
def setup_boundary_crossing_connections(self):
    """Setup connections for boundary crossing analysis in the Analysis tab"""
    # Assuming these widgets exist in your GUI
    self.boundary_analyze_btn.clicked.connect(self.run_boundary_crossing_analysis)
    self.angular_dist_btn.clicked.connect(self.analyze_angular_distribution)

    # Connect parameter update widgets
    self.pixel_size_spinbox.valueChanged.connect(self.update_boundary_params)
    self.dt_spinbox.valueChanged.connect(self.update_boundary_params)

def update_boundary_params(self):
    """Update boundary crossing analysis parameters"""
    self.boundary_params = {
        'pixel_size': self.pixel_size_spinbox.value(),
        'dt': self.dt_spinbox.value()
    }

def run_boundary_crossing_analysis(self):
    """Execute boundary crossing analysis"""
    try:
        if not hasattr(self, 'tracks_df') or self.tracks_df is None:
            self.show_error_message("No tracks loaded", "Please load tracking data first.")
            return

        if not hasattr(self, 'compartment_masks') or not self.compartment_masks:
            self.show_error_message("No compartments defined",
                                  "Please define compartment masks first.")
            return

        # Initialize analyzer with current parameters
        self.boundary_analyzer = BoundaryCrossingAnalyzer(
            dt=self.boundary_params.get('dt', 0.014)
        )

        # Run analysis
        crossing_events = self.boundary_analyzer.analyze_boundary_crossings(
            self.tracks_df,
            self.compartment_masks
        )

        # Update results table
        self.update_crossing_results_table(crossing_events)

        # Update status
        self.status_bar.showMessage(
            f"Found {len(crossing_events)} boundary crossing events."
        )

        # Enable angular analysis button if events were found
        self.angular_dist_btn.setEnabled(len(crossing_events) > 0)

    except Exception as e:
        self.show_error_message("Analysis Error", str(e))
        logging.error(f"Boundary crossing analysis error: {e}", exc_info=True)

def analyze_angular_distribution(self):
    """Execute angular distribution analysis"""
    try:
        if not hasattr(self, 'boundary_analyzer') or not self.boundary_analyzer.crossing_events:
            self.show_error_message("No crossing events",
                                  "Please run boundary crossing analysis first.")
            return

        # Run angular analysis
        angular_results = self.boundary_analyzer.analyze_angular_distribution(
            self.compartment_masks,
            pixel_size=self.boundary_params.get('pixel_size', 1.0)
        )

        if angular_results['status'] == 'Computed':
            # Plot angular distribution
            self.plot_angular_distribution(angular_results)

            # Update summary table
            self.update_angular_summary_table(angular_results['boundary_summary'])

            self.status_bar.showMessage(
                f"Analyzed angular distribution for {len(angular_results['crossing_angles'])} crossings."
            )

    except Exception as e:
        self.show_error_message("Analysis Error", str(e))
        logging.error(f"Angular distribution analysis error: {e}", exc_info=True)

def update_crossing_results_table(self, crossing_events):
    """Update the results table with boundary crossing events"""
    # Clear existing table
    self.crossing_results_table.setRowCount(0)

    # Set up table columns
    headers = ["Track ID", "Frame From", "Frame To",
              "From Compartment", "To Compartment",
              "Position From", "Position To"]
    self.crossing_results_table.setColumnCount(len(headers))
    self.crossing_results_table.setHorizontalHeaderLabels(headers)

    # Add events to table
    for event in crossing_events:
        row = self.crossing_results_table.rowCount()
        self.crossing_results_table.insertRow(row)

        self.crossing_results_table.setItem(row, 0,
            QTableWidgetItem(str(event['track_id'])))
        self.crossing_results_table.setItem(row, 1,
            QTableWidgetItem(str(event['frame_from'])))
        self.crossing_results_table.setItem(row, 2,
            QTableWidgetItem(str(event['frame_to'])))
        self.crossing_results_table.setItem(row, 3,
            QTableWidgetItem(event['from_compartment']))
        self.crossing_results_table.setItem(row, 4,
            QTableWidgetItem(event['to_compartment']))
        self.crossing_results_table.setItem(row, 5,
            QTableWidgetItem(str(event['position_from'])))
        self.crossing_results_table.setItem(row, 6,
            QTableWidgetItem(str(event['position_to'])))

def plot_angular_distribution(self, angular_results):
    """Plot angular distribution results using matplotlib"""
    try:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

        # Histogram of crossing angles
        angles = angular_results['crossing_angles']
        ax1.hist(angles, bins=36, range=(-180, 180))
        ax1.set_xlabel('Crossing Angle (degrees)')
        ax1.set_ylabel('Frequency')
        ax1.set_title('Distribution of Crossing Angles')

        # Polar plot
        ax2 = plt.subplot(122, projection='polar')
        angles_rad = np.deg2rad(angles)
        ax2.hist(angles_rad, bins=36)
        ax2.set_title('Polar Distribution')

        # Update the plot widget in the GUI
        self.angular_plot_widget.canvas.figure = fig
        self.angular_plot_widget.canvas.draw()

    except Exception as e:
        logging.error(f"Error plotting angular distribution: {e}", exc_info=True)

def update_angular_summary_table(self, boundary_summary):
    """Update the summary table with angular distribution statistics"""
    # Clear existing table
    self.angular_summary_table.setRowCount(0)

    # Set up table columns
    headers = ["Boundary Pair", "Count", "Mean Angle", "Circular Mean",
              "Std Dev", "Circular Std Dev"]
    self.angular_summary_table.setColumnCount(len(headers))
    self.angular_summary_table.setHorizontalHeaderLabels(headers)

    # Add summary data to table
    for boundary_pair, data in boundary_summary.items():
        row = self.angular_summary_table.rowCount()
        self.angular_summary_table.insertRow(row)

        self.angular_summary_table.setItem(row, 0,
            QTableWidgetItem(f"{boundary_pair[0]} - {boundary_pair[1]}"))
        self.angular_summary_table.setItem(row, 1,
            QTableWidgetItem(str(data['count'])))
        self.angular_summary_table.setItem(row, 2,
            QTableWidgetItem(f"{data['mean_angle']:.2f}°"))
        self.angular_summary_table.setItem(row, 3,
            QTableWidgetItem(f"{data['circular_mean_angle']:.2f}°"))
        self.angular_summary_table.setItem(row, 4,
            QTableWidgetItem(f"{data['std_angle']:.2f}°"))
        self.angular_summary_table.setItem(row, 5,
            QTableWidgetItem(f"{data['circular_std_dev_deg']:.2f}°"))
# Worker thread connections
def setup_worker_connections(self, worker):
    """Set up connections for a worker thread."""
    worker.progress_updated.connect(self.update_progress)
    worker.operation_completed.connect(self.handle_worker_completion)
    worker.error_occurred.connect(self.operation_error)

def handle_worker_completion(self, result):
    """Handle worker thread completion based on operation type."""
    if self.worker.operation_type == "detect_particles":
        self.detection_completed(result)
    elif self.worker.operation_type == "track_particles":
        self.linking_completed(result)
    elif self.worker.operation_type == "analyze_diffusion":
        # Handle diffusion analysis results
        self.analysis_results["diffusion"] = result
        self.display_analysis_results(result, "diffusion")
        self.update_data_summary()
        self.statusBar().showMessage("Diffusion analysis completed", 3000)

    # Reset UI
    self.progress_bar.setVisible(False)
    self.setEnabled(True)
        """Set up the main UI components."""
        # Create central widget and main layout
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)
        
        # Create tab widget
        self.tabs = QTabWidget()
        self.tabs.setTabPosition(QTabWidget.North)
        
        # Create tabs
        self.create_project_tab()
        self.create_image_processing_tab()
        self.create_tracking_tab()
        self.create_analysis_tab()
        self.create_visualization_tab()
        self.create_batch_tab()
        
        # Add tabs to tab widget
        main_layout.addWidget(self.tabs)
        
        # Create status bar
        self.statusBar = QStatusBar()
        self.setStatusBar(self.statusBar)
        
        # Create progress bar in status bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        self.progress_bar.setMaximumWidth(250)
        self.statusBar.addPermanentWidget(self.progress_bar)
        
        # Set up menu bar
        self.create_menu_bar()
    
    def create_menu_bar(self):
        """Create the application menu bar."""
        menubar = self.menuBar()
        
        # File menu
        file_menu = menubar.addMenu("&File")
        
        new_project_action = QAction("&New Project", self)
        new_project_action.setShortcut("Ctrl+N")
        new_project_action.triggered.connect(self.new_project)
        file_menu.addAction(new_project_action)
        
        open_project_action = QAction("&Open Project", self)
        open_project_action.setShortcut("Ctrl+O")
        open_project_action.triggered.connect(self.open_project)
        file_menu.addAction(open_project_action)
        
        save_project_action = QAction("&Save Project", self)
        save_project_action.setShortcut("Ctrl+S")
        save_project_action.triggered.connect(self.save_project)
        file_menu.addAction(save_project_action)
        
        file_menu.addSeparator()
        
        import_submenu = file_menu.addMenu("Import")
        
        import_images_action = QAction("Image Stack", self)
        import_images_action.triggered.connect(self.import_image_stack)
        import_submenu.addAction(import_images_action)
        
        import_tracks_action = QAction("Tracks", self)
        import_tracks_action.triggered.connect(self.import_tracks)
        import_submenu.addAction(import_tracks_action)
        
        export_submenu = file_menu.addMenu("Export")
        
        export_tracks_action = QAction("Tracks", self)
        export_tracks_action.triggered.connect(self.export_tracks)
        export_submenu.addAction(export_tracks_action)
        
        export_results_action = QAction("Analysis Results", self)
        export_results_action.triggered.connect(self.export_results)
        export_submenu.addAction(export_results_action)
        
        export_figure_action = QAction("Current Figure", self)
        export_figure_action.triggered.connect(self.export_figure)
        export_submenu.addAction(export_figure_action)
        
        file_menu.addSeparator()
        
        exit_action = QAction("E&xit", self)
        exit_action.setShortcut("Ctrl+Q")
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)
        
        # Edit menu
        edit_menu = menubar.addMenu("&Edit")
        
        settings_action = QAction("&Settings", self)
        settings_action.triggered.connect(self.edit_project_settings)
        edit_menu.addAction(settings_action)
        
        # Tools menu
        tools_menu = menubar.addMenu("&Tools")
        
        batch_process_action = QAction("&Batch Processing", self)
        batch_process_action.triggered.connect(lambda: self.tabs.setCurrentIndex(5))  # Switch to batch tab
        tools_menu.addAction(batch_process_action)
        
        # Help menu
        help_menu = menubar.addMenu("&Help")
        
        about_action = QAction("&About", self)
        about_action.triggered.connect(self.show_about_dialog)
        help_menu.addAction(about_action)
        
        documentation_action = QAction("&Documentation", self)
        documentation_action.triggered.connect(self.show_documentation)
        help_menu.addAction(documentation_action)
    
    def create_project_tab(self):
        """Create the project management tab."""
        project_tab = QWidget()
        layout = QVBoxLayout()
        
        # Project info group
        project_info_group = QGroupBox("Project Information")
        info_layout = QFormLayout()
        
        self.project_name_label = QLabel(self.project_settings["project_name"])
        info_layout.addRow("Project Name:", self.project_name_label)
        
        self.pixel_size_label = QLabel(f"{self.project_settings['pixel_size']} μm")
        info_layout.addRow("Pixel Size:", self.pixel_size_label)
        
        self.frame_interval_label = QLabel(f"{self.project_settings['frame_interval']} s")
        info_layout.addRow("Frame Interval:", self.frame_interval_label)
        
        self.temperature_label = QLabel(f"{self.project_settings['temperature']} °C")
        info_layout.addRow("Temperature:", self.temperature_label)
        
        self.particle_radius_label = QLabel(f"{self.project_settings['particle_radius']} nm")
        info_layout.addRow("Particle Radius:", self.particle_radius_label)
        
        project_info_group.setLayout(info_layout)
        layout.addWidget(project_info_group)
        
        # Project notes group
        notes_group = QGroupBox("Project Notes")
        notes_layout = QVBoxLayout()
        self.project_notes = QTextEdit()
        self.project_notes.setReadOnly(True)
        self.project_notes.setText(self.project_settings["notes"])
        notes_layout.addWidget(self.project_notes)
        notes_group.setLayout(notes_layout)
        layout.addWidget(notes_group)
        
        # Data summary group
        data_summary_group = QGroupBox("Data Summary")
        summary_layout = QFormLayout()
        
        self.image_stack_info = QLabel("No image stack loaded")
        summary_layout.addRow("Image Stack:", self.image_stack_info)
        
        self.detections_info = QLabel("No detections")
        summary_layout.addRow("Detections:", self.detections_info)
        
        self.tracks_info = QLabel("No tracks")
        summary_layout.addRow("Tracks:", self.tracks_info)
        
        self.analysis_info = QLabel("No analysis results")
        summary_layout.addRow("Analysis Results:", self.analysis_info)
        
        data_summary_group.setLayout(summary_layout)
        layout.addWidget(data_summary_group)
        
        # Buttons
        buttons_layout = QHBoxLayout()
        
        new_project_btn = QPushButton("New Project")
        new_project_btn.clicked.connect(self.new_project)
        buttons_layout.addWidget(new_project_btn)
        
        load_project_btn = QPushButton("Load Project")
        load_project_btn.clicked.connect(self.open_project)
        buttons_layout.addWidget(load_project_btn)
        
        save_project_btn = QPushButton("Save Project")
        save_project_btn.clicked.connect(self.save_project)
        buttons_layout.addWidget(save_project_btn)
        
        edit_settings_btn = QPushButton("Edit Settings")
        edit_settings_btn.clicked.connect(self.edit_project_settings)
        buttons_layout.addWidget(edit_settings_btn)
        
        layout.addLayout(buttons_layout)
        
        project_tab.setLayout(layout)
        self.tabs.addTab(project_tab, "Project")
    
    def create_image_processing_tab(self):
        """Create the image processing tab."""
        image_tab = QWidget()
        layout = QVBoxLayout()
        
        # Top controls
        top_controls = QHBoxLayout()
        
        # Frame slider and navigation
        frame_controls = QHBoxLayout()
        
        frame_label = QLabel("Frame:")
        frame_controls.addWidget(frame_label)
        
        self.frame_slider = QSpinBox()
        self.frame_slider.setMinimum(0)
        self.frame_slider.setMaximum(0)
        self.frame_slider.valueChanged.connect(self.update_display_frame)
        frame_controls.addWidget(self.frame_slider)
        
        prev_frame_btn = QPushButton("Previous")
        prev_frame_btn.clicked.connect(lambda: self.frame_slider.setValue(self.frame_slider.value() - 1))
        frame_controls.addWidget(prev_frame_btn)
        
        next_frame_btn = QPushButton("Next")
        next_frame_btn.clicked.connect(lambda: self.frame_slider.setValue(self.frame_slider.value() + 1))
        frame_controls.addWidget(next_frame_btn)
        
        top_controls.addLayout(frame_controls)
        
        top_controls.addStretch()
        
        # Import button
        import_btn = QPushButton("Import Image Stack")
        import_btn.clicked.connect(self.import_image_stack)
        top_controls.addWidget(import_btn)
        
        layout.addLayout(top_controls)
        
        # Main content: image viewer and controls
        main_content = QHBoxLayout()
        
        # Image viewer
        image_view_group = QGroupBox("Image View")
        image_view_layout = QVBoxLayout()
        
        self.image_canvas = MplCanvas(width=5, height=4, dpi=100)
        image_view_layout.addWidget(self.image_canvas)
        
        # Add toolbar
        image_toolbar = NavigationToolbar(self.image_canvas, self)
        image_view_layout.addWidget(image_toolbar)
        
        image_view_group.setLayout(image_view_layout)
        main_content.addWidget(image_view_group, 3)
        
        # Processing controls
        processing_group = QGroupBox("Image Processing")
        processing_layout = QVBoxLayout()
        
        # Preprocessing options
        preproc_form = QFormLayout()
        
        # Contrast enhancement
        self.contrast_method = QComboBox()
        self.contrast_method.addItems(["None", "Stretch", "Equalize", "Adaptive"])
        preproc_form.addRow("Contrast Enhancement:", self.contrast_method)
        
        # Denoising
        self.denoise_method = QComboBox()
        self.denoise_method.addItems(["None", "Gaussian", "Median", "Bilateral", "NLMeans"])
        preproc_form.addRow("Denoising:", self.denoise_method)
        
        # Denoising strength
        self.denoise_strength = QDoubleSpinBox()
        self.denoise_strength.setRange(0.1, 10.0)
        self.denoise_strength.setSingleStep(0.1)
        self.denoise_strength.setValue(1.0)
        preproc_form.addRow("Denoising Strength:", self.denoise_strength)
        
        processing_layout.addLayout(preproc_form)
        
        # Buttons for processing
        process_btn = QPushButton("Apply Processing")
        process_btn.clicked.connect(self.apply_image_processing)
        processing_layout.addWidget(process_btn)
        
        reset_btn = QPushButton("Reset Processing")
        reset_btn.clicked.connect(self.reset_image_processing)
        processing_layout.addWidget(reset_btn)
        
        # Segmentation controls
        seg_group = QGroupBox("Segmentation")
        seg_layout = QFormLayout()
        
        self.seg_method = QComboBox()
        self.seg_method.addItems(["None", "Threshold", "Otsu", "Watershed", "Active Contour"])
        seg_layout.addRow("Method:", self.seg_method)
        
        self.show_segmentation = QCheckBox("Show Segmentation")
        self.show_segmentation.setChecked(False)
        seg_layout.addRow("", self.show_segmentation)
        
        seg_btn = QPushButton("Perform Segmentation")
        seg_layout.addRow("", seg_btn)
        
        seg_group.setLayout(seg_layout)
        processing_layout.addWidget(seg_group)
        
        processing_layout.addStretch()
        
        processing_group.setLayout(processing_layout)
        main_content.addWidget(processing_group, 1)
        
        layout.addLayout(main_content)
        
        image_tab.setLayout(layout)
        self.tabs.addTab(image_tab, "Image Processing")
    
    def create_tracking_tab(self):
        """Create the particle tracking tab."""
        tracking_tab = QWidget()
        layout = QVBoxLayout()
        
        # Top controls
        top_controls = QHBoxLayout()
        
        # Frame controls
        frame_controls = QHBoxLayout()
        
        frame_label = QLabel("Frame:")
        frame_controls.addWidget(frame_label)
        
        self.tracking_frame_slider = QSpinBox()
        self.tracking_frame_slider.setMinimum(0)
        self.tracking_frame_slider.setMaximum(0)
        self.tracking_frame_slider.valueChanged.connect(self.update_tracking_frame)
        frame_controls.addWidget(self.tracking_frame_slider)
        
        prev_frame_btn = QPushButton("Previous")
        prev_frame_btn.clicked.connect(lambda: self.tracking_frame_slider.setValue(self.tracking_frame_slider.value() - 1))
        frame_controls.addWidget(prev_frame_btn)
        
        next_frame_btn = QPushButton("Next")
        next_frame_btn.clicked.connect(lambda: self.tracking_frame_slider.setValue(self.tracking_frame_slider.value() + 1))
        frame_controls.addWidget(next_frame_btn)
        
        top_controls.addLayout(frame_controls)
        
        top_controls.addStretch()
        
        layout.addLayout(top_controls)
        
        # Main content area
        main_content = QHBoxLayout()
        
        # Tracking view
        tracking_view_group = QGroupBox("Tracking View")
        tracking_view_layout = QVBoxLayout()
        
        self.tracking_canvas = MplCanvas(width=5, height=4, dpi=100)
        tracking_view_layout.addWidget(self.tracking_canvas)
        
        # Add toolbar
        tracking_toolbar = NavigationToolbar(self.tracking_canvas, self)
        tracking_view_layout.addWidget(tracking_toolbar)
        
        display_options = QHBoxLayout()
        
        self.show_detections = QCheckBox("Show Detections")
        self.show_detections.setChecked(True)
        self.show_detections.stateChanged.connect(self.update_tracking_display)
        display_options.addWidget(self.show_detections)
        
        self.show_tracks = QCheckBox("Show Tracks")
        self.show_tracks.setChecked(True)
        self.show_tracks.stateChanged.connect(self.update_tracking_display)
        display_options.addWidget(self.show_tracks)
        
        self.track_history = QSpinBox()
        self.track_history.setRange(1, 100)
        self.track_history.setValue(10)
        self.track_history.setPrefix("History: ")
        self.track_history.valueChanged.connect(self.update_tracking_display)
        display_options.addWidget(self.track_history)
        
        tracking_view_layout.addLayout(display_options)
        
        tracking_view_group.setLayout(tracking_view_layout)
        main_content.addWidget(tracking_view_group, 3)
        
        # Tracking controls
        tracking_controls_group = QGroupBox("Tracking Controls")
        tracking_controls_layout = QVBoxLayout()
        
        # Detection parameters
        detection_group = QGroupBox("Particle Detection")
        detection_layout = QFormLayout()
        
        self.detection_method = QComboBox()
        self.detection_method.addItems(["LoG", "DoG", "DoH", "Wavelet", "LocalMax"])
        detection_layout.addRow("Method:", self.detection_method)
        
        self.min_sigma = QDoubleSpinBox()
        self.min_sigma.setRange(0.1, 20.0)
        self.min_sigma.setSingleStep(0.1)
        self.min_sigma.setValue(1.0)
        detection_layout.addRow("Min Sigma:", self.min_sigma)
        
        self.max_sigma = QDoubleSpinBox()
        self.max_sigma.setRange(0.1, 20.0)
        self.max_sigma.setSingleStep(0.1)
        self.max_sigma.setValue(5.0)
        detection_layout.addRow("Max Sigma:", self.max_sigma)
        
        self.num_sigma = QSpinBox()
        self.num_sigma.setRange(1, 20)
        self.num_sigma.setValue(10)
        detection_layout.addRow("Num Sigma:", self.num_sigma)
        
        self.threshold = QDoubleSpinBox()
        self.threshold.setRange(0.0, 1000.0)
        self.threshold.setSingleStep(0.1)
        self.threshold.setValue(0.1)
        detection_layout.addRow("Threshold:", self.threshold)
        
        detect_btn = QPushButton("Detect Particles")
        detect_btn.clicked.connect(self.detect_particles)
        detection_layout.addRow("", detect_btn)
        
        detection_group.setLayout(detection_layout)
        tracking_controls_layout.addWidget(detection_group)
        
        # Linking parameters
        linking_group = QGroupBox("Track Linking")
        linking_layout = QFormLayout()
        
        self.linking_method = QComboBox()
        self.linking_method.addItems(["Hungarian", "NearestNeighbor", "GraphBased", "MHT"])
        linking_layout.addRow("Method:", self.linking_method)
        
        self.max_distance = QDoubleSpinBox()
        self.max_distance.setRange(0.1, 100.0)
        self.max_distance.setSingleStep(0.1)
        self.max_distance.setValue(10.0)
        self.max_distance.setSuffix(" px")
        linking_layout.addRow("Max Distance:", self.max_distance)
        
        self.max_gap_closing = QSpinBox()
        self.max_gap_closing.setRange(0, 10)
        self.max_gap_closing.setValue(2)
        self.max_gap_closing.setSuffix(" frames")
        linking_layout.addRow("Gap Closing:", self.max_gap_closing)
        
        link_btn = QPushButton("Link Tracks")
        link_btn.clicked.connect(self.link_tracks)
        linking_layout.addRow("", link_btn)
        
        linking_group.setLayout(linking_layout)
        tracking_controls_layout.addWidget(linking_group)
        
        # Track filtering
        filtering_group = QGroupBox("Track Filtering")
        filtering_layout = QFormLayout()
        
        self.min_track_length = QSpinBox()
        self.min_track_length.setRange(1, 100)
        self.min_track_length.setValue(5)
        self.min_track_length.setSuffix(" frames")
        filtering_layout.addRow("Min Length:", self.min_track_length)
        
        filter_btn = QPushButton("Filter Tracks")
        filter_btn.clicked.connect(self.filter_tracks)
        filtering_layout.addRow("", filter_btn)
        
        filtering_group.setLayout(filtering_layout)
        tracking_controls_layout.addWidget(filtering_group)
        
        # Add export button
        export_tracks_btn = QPushButton("Export Tracks")
        export_tracks_btn.clicked.connect(self.export_tracks)
        tracking_controls_layout.addWidget(export_tracks_btn)
        
        tracking_controls_layout.addStretch()
        
        tracking_controls_group.setLayout(tracking_controls_layout)
        main_content.addWidget(tracking_controls_group, 1)
        
        layout.addLayout(main_content)
        
        tracking_tab.setLayout(layout)
        self.tabs.addTab(tracking_tab, "Tracking")
    
    def create_analysis_tab(self):
        """Create the analysis tab."""
        analysis_tab = QWidget()
        layout = QVBoxLayout()
        
        # Split view: analysis menu on left, results on right
        splitter = QSplitter(Qt.Horizontal)
        
        # Analysis menu (left side)
        analysis_menu = QWidget()
        menu_layout = QVBoxLayout(analysis_menu)
        
        analysis_type_label = QLabel("Select Analysis Type:")
        menu_layout.addWidget(analysis_type_label)
        
        # Analysis type selection
        self.analysis_type = QComboBox()
        self.analysis_type.addItems([
            "Diffusion Analysis", 
            "Active Transport", 
            "Boundary Crossing", 
            "Dwell Time Analysis",
            "Crowding Effects", 
            "Diffusion Population", 
            "Gel Structure", 
            "Microcompartment Analysis"
        ])
        self.analysis_type.currentIndexChanged.connect(self.update_analysis_parameters)
        menu_layout.addWidget(self.analysis_type)
        
        # Parameters area (changes based on selected analysis)
        self.params_group = QGroupBox("Analysis Parameters")
        self.params_layout = QFormLayout()
        self.params_group.setLayout(self.params_layout)
        menu_layout.addWidget(self.params_group)
        
        # Run analysis button
        run_analysis_btn = QPushButton("Run Analysis")
        run_analysis_btn.clicked.connect(self.run_analysis)
        menu_layout.addWidget(run_analysis_btn)
        
        menu_layout.addStretch()
        
        # Results display (right side)
        results_widget = QWidget()
        results_layout = QVBoxLayout(results_widget)
        
        results_label = QLabel("Analysis Results")
        results_label.setAlignment(Qt.AlignCenter)
        results_layout.addWidget(results_label)
        
        # Tabs for different result views
        self.results_tabs = QTabWidget()
        
        # Summary tab
        summary_tab = QWidget()
        summary_layout = QVBoxLayout(summary_tab)
        self.results_summary = QTextEdit()
        self.results_summary.setReadOnly(True)
        summary_layout.addWidget(self.results_summary)
        self.results_tabs.addTab(summary_tab, "Summary")
        
        # Table tab
        table_tab = QWidget()
        table_layout = QVBoxLayout(table_tab)
        self.results_table = QTableWidget()
        table_layout.addWidget(self.results_table)
        self.results_tabs.addTab(table_tab, "Table")
        
        # Plot tab
        plot_tab = QWidget()
        plot_layout = QVBoxLayout(plot_tab)
        self.results_canvas = MplCanvas(width=5, height=4, dpi=100)
        plot_layout.addWidget(self.results_canvas)
        plot_toolbar = NavigationToolbar(self.results_canvas, self)
        plot_layout.addWidget(plot_toolbar)
        self.results_tabs.addTab(plot_tab, "Plot")
        
        results_layout.addWidget(self.results_tabs)
        
        # Export results button
        export_results_btn = QPushButton("Export Results")
        export_results_btn.clicked.connect(self.export_results)
        results_layout.addWidget(export_results_btn)
        
        # Add widgets to splitter
        splitter.addWidget(analysis_menu)
        splitter.addWidget(results_widget)
        splitter.setSizes([300, 700])  # Set initial sizes
        
        layout.addWidget(splitter)
        
        analysis_tab.setLayout(layout)
        self.tabs.addTab(analysis_tab, "Analysis")
        
        # Initialize the first analysis type
        self.update_analysis_parameters(0)
    
    def create_visualization_tab(self):
        """Create the visualization tab."""
        visualization_tab = QWidget()
        layout = QVBoxLayout()
        
        # Split view: visualization controls on left, display on right
        splitter = QSplitter(Qt.Horizontal)
        
        # Visualization controls (left side)
        controls_widget = QWidget()
        controls_layout = QVBoxLayout(controls_widget)
        
        viz_type_label = QLabel("Visualization Type:")
        controls_layout.addWidget(viz_type_label)
        
        # Visualization type selection
        self.viz_type = QComboBox()
        self.viz_type.addItems([
            "Track Trajectories", 
            "Diffusion Map", 
            "MSD Curves", 
            "Jump Distance Histogram", 
            "Velocity Distribution", 
            "Angle Distribution", 
            "Spatial Clusters",
            "Dwell Time Distribution",
            "Confinement Ratio Map",
            "Anomalous Exponent Map"
        ])
        self.viz_type.currentIndexChanged.connect(self.update_visualization_parameters)
        controls_layout.addWidget(self.viz_type)
        
        # Parameters area (changes based on selected visualization)
        self.viz_params_group = QGroupBox("Visualization Parameters")
        self.viz_params_layout = QFormLayout()
        self.viz_params_group.setLayout(self.viz_params_layout)
        controls_layout.addWidget(self.viz_params_group)
        
        # Generate visualization button
        generate_viz_btn = QPushButton("Generate Visualization")
        generate_viz_btn.clicked.connect(self.generate_visualization)
        controls_layout.addWidget(generate_viz_btn)
        
        controls_layout.addStretch()
        
        # Visualization display (right side)
        viz_display = QWidget()
        viz_display_layout = QVBoxLayout(viz_display)
        
        # Canvas for visualization
        self.viz_canvas = MplCanvas(width=6, height=5, dpi=100)
        viz_display_layout.addWidget(self.viz_canvas)
        
        # Toolbar for visualization
        viz_toolbar = NavigationToolbar(self.viz_canvas, self)
        viz_display_layout.addWidget(viz_toolbar)
        
        # Export visualization button
        export_viz_btn = QPushButton("Export Visualization")
        export_viz_btn.clicked.connect(self.export_figure)
        viz_display_layout.addWidget(export_viz_btn)
        
        # Add widgets to splitter
        splitter.addWidget(controls_widget)
        splitter.addWidget(viz_display)
        splitter.setSizes([300, 700])  # Set initial sizes
        
        layout.addWidget(splitter)
        
        visualization_tab.setLayout(layout)
        self.tabs.addTab(visualization_tab, "Visualization")
        
        # Initialize the first visualization type
        self.update_visualization_parameters(0)
    
    def create_batch_tab(self):
        """Create the batch processing tab."""
        batch_tab = QWidget()
        layout = QVBoxLayout()
        
        # Batch processing instructions
        instructions = QLabel(
            "Batch processing allows you to run the same analysis on multiple datasets. "
            "Add datasets, configure the analysis parameters, and run the batch process."
        )
        instructions.setWordWrap(True)
        layout.addWidget(instructions)
        
        # Dataset management
        datasets_group = QGroupBox("Datasets")
        datasets_layout = QVBoxLayout()
        
        # Dataset list
        self.dataset_list = QListWidget()
        datasets_layout.addWidget(self.dataset_list)
        
        # Dataset controls
        dataset_controls = QHBoxLayout()
        
        add_dataset_btn = QPushButton("Add Dataset")
        add_dataset_btn.clicked.connect(self.add_batch_dataset)
        dataset_controls.addWidget(add_dataset_btn)
        
        remove_dataset_btn = QPushButton("Remove Dataset")
        remove_dataset_btn.clicked.connect(self.remove_batch_dataset)
        dataset_controls.addWidget(remove_dataset_btn)
        
        datasets_layout.addLayout(dataset_controls)
        datasets_group.setLayout(datasets_layout)
        layout.addWidget(datasets_group)
        
        # Batch configuration
        config_group = QGroupBox("Batch Configuration")
        config_layout = QFormLayout()
        
        self.batch_analysis_type = QComboBox()
        self.batch_analysis_type.addItems([
            "Diffusion Analysis", 
            "Active Transport", 
            "Boundary Crossing", 
            "Dwell Time Analysis",
            "Crowding Effects", 
            "Diffusion Population", 
            "Gel Structure", 
            "Microcompartment Analysis"
        ])
        config_layout.addRow("Analysis Type:", self.batch_analysis_type)
        
        self.batch_export_dir = QLineEdit()
        self.batch_export_dir.setReadOnly(True)
        browse_btn = QPushButton("Browse...")
        browse_btn.clicked.connect(self.select_batch_export_dir)
        
        export_dir_layout = QHBoxLayout()
        export_dir_layout.addWidget(self.batch_export_dir)
        export_dir_layout.addWidget(browse_btn)
        
        config_layout.addRow("Export Directory:", export_dir_layout)
        
        config_group.setLayout(config_layout)
        layout.addWidget(config_group)
        
        # Run batch button
        run_batch_btn = QPushButton("Run Batch Process")
        run_batch_btn.clicked.connect(self.run_batch_process)
        layout.addWidget(run_batch_btn)
        
        # Batch progress
        progress_group = QGroupBox("Batch Progress")
        progress_layout = QVBoxLayout()
        
        self.batch_progress = QProgressBar()
        progress_layout.addWidget(self.batch_progress)
        
        self.batch_status = QTextEdit()
        self.batch_status.setReadOnly(True)
        progress_layout.addWidget(self.batch_status)
        
        progress_group.setLayout(progress_layout)
        layout.addWidget(progress_group)
        
        batch_tab.setLayout(layout)
        self.tabs.addTab(batch_tab, "Batch Processing")
    
    def update_analysis_parameters(self, index):
        """Update the analysis parameters based on the selected analysis type."""
        # Clear existing parameters
        while self.params_layout.count():
            item = self.params_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        
        analysis_type = self.analysis_type.currentText()
        
        if analysis_type == "Diffusion Analysis":
            # Add diffusion analysis parameters
            self.msd_max_lag = QSpinBox()
            self.msd_max_lag.setRange(2, 100)
            self.msd_max_lag.setValue(20)
            self.params_layout.addRow("Max Lag:", self.msd_max_lag)
            
            self.msd_fit_points = QSpinBox()
            self.msd_fit_points.setRange(2, 20)
            self.msd_fit_points.setValue(5)
            self.params_layout.addRow("Fit Points:", self.msd_fit_points)
            
            self.model_selection = QComboBox()
            self.model_selection.addItems(["Brownian", "Anomalous", "Confined", "All"])
            self.params_layout.addRow("Model:", self.model_selection)
            
        elif analysis_type == "Active Transport":
            # Add active transport parameters
            self.min_superdiffusive_alpha = QDoubleSpinBox()
            self.min_superdiffusive_alpha.setRange(1.1, 2.0)
            self.min_superdiffusive_alpha.setSingleStep(0.1)
            self.min_superdiffusive_alpha.setValue(1.3)
            self.params_layout.addRow("Min α:", self.min_superdiffusive_alpha)
            
            self.min_track_length_at = QSpinBox()
            self.min_track_length_at.setRange(5, 100)
            self.min_track_length_at.setValue(10)
            self.params_layout.addRow("Min Track Length:", self.min_track_length_at)
            
            self.velocity_analysis = QCheckBox()
            self.velocity_analysis.setChecked(True)
            self.params_layout.addRow("Velocity Analysis:", self.velocity_analysis)
            
        elif analysis_type == "Boundary Crossing":
            # Add boundary crossing parameters
            self.use_segmentation = QCheckBox()
            self.use_segmentation.setChecked(True)
            self.params_layout.addRow("Use Segmentation:", self.use_segmentation)
            
            self.boundary_width = QDoubleSpinBox()
            self.boundary_width.setRange(0.1, 10.0)
            self.boundary_width.setSingleStep(0.1)
            self.boundary_width.setValue(1.0)
            self.boundary_width.setSuffix(" px")
            self.params_layout.addRow("Boundary Width:", self.boundary_width)
            
            self.analyze_angles = QCheckBox()
            self.analyze_angles.setChecked(True)
            self.params_layout.addRow("Analyze Angles:", self.analyze_angles)
            
        elif analysis_type == "Dwell Time Analysis":
            # Add dwell time parameters
            self.immobility_threshold = QDoubleSpinBox()
            self.immobility_threshold.setRange(0.1, 10.0)
            self.immobility_threshold.setSingleStep(0.1)
            self.immobility_threshold.setValue(1.0)
            self.immobility_threshold.setSuffix(" px")
            self.params_layout.addRow("Immobility Threshold:", self.immobility_threshold)
            
            self.min_binding_frames = QSpinBox()
            self.min_binding_frames.setRange(2, 100)
            self.min_binding_frames.setValue(5)
            self.params_layout.addRow("Min Binding Frames:", self.min_binding_frames)
            
            self.analyze_escapes = QCheckBox()
            self.analyze_escapes.setChecked(True)
            self.params_layout.addRow("Analyze Cage Escapes:", self.analyze_escapes)
            
        elif analysis_type == "Crowding Effects":
            # Add crowding parameters
            self.subdiff_threshold = QDoubleSpinBox()
            self.subdiff_threshold.setRange(0.1, 1.0)
            self.subdiff_threshold.setSingleStep(0.05)
            self.subdiff_threshold.setValue(0.8)
            self.params_layout.addRow("Subdiffusion Threshold:", self.subdiff_threshold)
            
            self.viscosity_analysis = QCheckBox()
            self.viscosity_analysis.setChecked(True)
            self.params_layout.addRow("Viscosity Analysis:", self.viscosity_analysis)
            
            self.non_gaussian = QCheckBox()
            self.non_gaussian.setChecked(True)
            self.params_layout.addRow("Non-Gaussian Analysis:", self.non_gaussian)
            
        elif analysis_type == "Diffusion Population":
            # Add diffusion population parameters
            self.num_populations = QSpinBox()
            self.num_populations.setRange(1, 5)
            self.num_populations.setValue(2)
            self.params_layout.addRow("Number of Populations:", self.num_populations)
            
            self.fit_method = QComboBox()
            self.fit_method.addItems(["GMM", "KDE", "Histogram"])
            self.params_layout.addRow("Fitting Method:", self.fit_method)
            
            self.segment_trajectories = QCheckBox()
            self.segment_trajectories.setChecked(True)
            self.params_layout.addRow("Segment Trajectories:", self.segment_trajectories)
            
        elif analysis_type == "Gel Structure":
            # Add gel structure parameters
            self.max_jump_distance = QDoubleSpinBox()
            self.max_jump_distance.setRange(1.0, 100.0)
            self.max_jump_distance.setSingleStep(1.0)
            self.max_jump_distance.setValue(20.0)
            self.max_jump_distance.setSuffix(" px")
            self.params_layout.addRow("Max Jump Distance:", self.max_jump_distance)
            
            self.distance_bins = QSpinBox()
            self.distance_bins.setRange(10, 200)
            self.distance_bins.setValue(50)
            self.params_layout.addRow("Distance Bins:", self.distance_bins)
            
            self.analyze_pores = QCheckBox()
            self.analyze_pores.setChecked(True)
            self.params_layout.addRow("Analyze Pore Size:", self.analyze_pores)
            
        elif analysis_type == "Microcompartment Analysis":
            # Add microcompartment parameters
            self.compartment_method = QComboBox()
            self.compartment_method.addItems(["Voronoi", "Density-based", "MSD-based"])
            self.params_layout.addRow("Method:", self.compartment_method)
            
            self.min_compartment_size = QDoubleSpinBox()
            self.min_compartment_size.setRange(0.1, 50.0)
            self.min_compartment_size.setSingleStep(0.1)
            self.min_compartment_size.setValue(1.0)
            self.min_compartment_size.setSuffix(" μm²")
            self.params_layout.addRow("Min Compartment Size:", self.min_compartment_size)
            
            self.min_localizations = QSpinBox()
            self.min_localizations.setRange(1, 1000)
            self.min_localizations.setValue(10)
            self.params_layout.addRow("Min Localizations:", self.min_localizations)
    
    def update_visualization_parameters(self, index):
        """Update the visualization parameters based on the selected visualization type."""
        # Clear existing parameters
        while self.viz_params_layout.count():
            item = self.viz_params_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        
        viz_type = self.viz_type.currentText()
        
        if viz_type == "Track Trajectories":
            # Add track visualization parameters
            self.color_by = QComboBox()
            self.color_by.addItems(["Track ID", "Diffusion Coefficient", "Time", "Speed", "Length"])
            self.viz_params_layout.addRow("Color By:", self.color_by)
            
            self.colormap = QComboBox()
            self.colormap.addItems(["viridis", "plasma", "inferno", "magma", "cividis", "jet", "tab10"])
            self.viz_params_layout.addRow("Colormap:", self.colormap)
            
            self.show_background = QCheckBox()
            self.show_background.setChecked(True)
            self.viz_params_layout.addRow("Show Background:", self.show_background)
            
            self.track_line_width = QDoubleSpinBox()
            self.track_line_width.setRange(0.1, 5.0)
            self.track_line_width.setSingleStep(0.1)
            self.track_line_width.setValue(1.0)
            self.viz_params_layout.addRow("Line Width:", self.track_line_width)
            
        elif viz_type == "Diffusion Map":
            # Add diffusion map parameters
            self.diff_colormap = QComboBox()
            self.diff_colormap.addItems(["viridis", "plasma", "inferno", "magma", "cividis", "jet"])
            self.viz_params_layout.addRow("Colormap:", self.diff_colormap)
            
            self.log_scale = QCheckBox()
            self.log_scale.setChecked(True)
            self.viz_params_layout.addRow("Log Scale:", self.log_scale)
            
            self.show_diff_background = QCheckBox()
            self.show_diff_background.setChecked(True)
            self.viz_params_layout.addRow("Show Background:", self.show_diff_background)
            
            self.diff_min = QDoubleSpinBox()
            self.diff_min.setRange(0.001, 10.0)
            self.diff_min.setSingleStep(0.01)
            self.diff_min.setValue(0.01)
            self.diff_min.setSuffix(" μm²/s")
            self.viz_params_layout.addRow("Min Value:", self.diff_min)
            
            self.diff_max = QDoubleSpinBox()
            self.diff_max.setRange(0.001, 10.0)
            self.diff_max.setSingleStep(0.01)
            self.diff_max.setValue(1.0)
            self.diff_max.setSuffix(" μm²/s")
            self.viz_params_layout.addRow("Max Value:", self.diff_max)
            
        elif viz_type == "MSD Curves":
            # Add MSD curve parameters
            self.msd_viz_max_lag = QSpinBox()
            self.msd_viz_max_lag.setRange(2, 100)
            self.msd_viz_max_lag.setValue(20)
            self.viz_params_layout.addRow("Max Lag:", self.msd_viz_max_lag)
            
            self.msd_viz_group_by = QComboBox()
            self.msd_viz_group_by.addItems(["None", "Diffusion Coefficient", "Alpha Value", "Custom"])
            self.viz_params_layout.addRow("Group By:", self.msd_viz_group_by)
            
            self.msd_viz_num_groups = QSpinBox()
            self.msd_viz_num_groups.setRange(1, 10)
            self.msd_viz_num_groups.setValue(3)
            self.viz_params_layout.addRow("Number of Groups:", self.msd_viz_num_groups)
            
            self.msd_viz_show_fit = QCheckBox()
            self.msd_viz_show_fit.setChecked(True)
            self.viz_params_layout.addRow("Show Fits:", self.msd_viz_show_fit)
            
            self.msd_viz_show_errorbars = QCheckBox()
            self.msd_viz_show_errorbars.setChecked(True)
            self.viz_params_layout.addRow("Show Error Bars:", self.msd_viz_show_errorbars)
            
        elif viz_type == "Jump Distance Histogram":
            # Add jump distance histogram parameters
            self.jd_viz_frame_interval = QSpinBox()
            self.jd_viz_frame_interval.setRange(1, 10)
            self.jd_viz_frame_interval.setValue(1)
            self.viz_params_layout.addRow("Frame Interval:", self.jd_viz_frame_interval)
            
            self.jd_viz_num_bins = QSpinBox()
            self.jd_viz_num_bins.setRange(10, 200)
            self.jd_viz_num_bins.setValue(50)
            self.viz_params_layout.addRow("Number of Bins:", self.jd_viz_num_bins)
            
            self.jd_viz_fit_models = QComboBox()
            self.jd_viz_fit_models.addItems(["None", "Rayleigh", "Multi-Rayleigh", "Custom"])
            self.viz_params_layout.addRow("Fit Model:", self.jd_viz_fit_models)
            
            self.jd_viz_normalize = QCheckBox()
            self.jd_viz_normalize.setChecked(True)
            self.viz_params_layout.addRow("Normalize:", self.jd_viz_normalize)
            
        # Add more visualization types as needed
        
        elif viz_type == "Velocity Distribution":
            # Add velocity distribution parameters
            self.vel_viz_window_size = QSpinBox()
            self.vel_viz_window_size.setRange(2, 10)
            self.vel_viz_window_size.setValue(3)
            self.viz_params_layout.addRow("Window Size:", self.vel_viz_window_size)
            
            self.vel_viz_num_bins = QSpinBox()
            self.vel_viz_num_bins.setRange(10, 200)
            self.vel_viz_num_bins.setValue(50)
            self.viz_params_layout.addRow("Number of Bins:", self.vel_viz_num_bins)
            
            self.vel_viz_max_velocity = QDoubleSpinBox()
            self.vel_viz_max_velocity.setRange(0.1, 100.0)
            self.vel_viz_max_velocity.setSingleStep(0.1)
            self.vel_viz_max_velocity.setValue(10.0)
            self.vel_viz_max_velocity.setSuffix(" μm/s")
            self.viz_params_layout.addRow("Max Velocity:", self.vel_viz_max_velocity)
            
        elif viz_type == "Angle Distribution":
            # Add angle distribution parameters
            self.angle_viz_window_size = QSpinBox()
            self.angle_viz_window_size.setRange(2, 10)
            self.angle_viz_window_size.setValue(3)
            self.viz_params_layout.addRow("Window Size:", self.angle_viz_window_size)
            
            self.angle_viz_num_bins = QSpinBox()
            self.angle_viz_num_bins.setRange(6, 72)
            self.angle_viz_num_bins.setValue(36)
            self.viz_params_layout.addRow("Number of Bins:", self.angle_viz_num_bins)
            
            self.angle_viz_plot_type = QComboBox()
            self.angle_viz_plot_type.addItems(["Rose", "Histogram", "Polar"])
            self.viz_params_layout.addRow("Plot Type:", self.angle_viz_plot_type)
            
        elif viz_type == "Spatial Clusters":
            # Add spatial cluster parameters
            self.cluster_viz_method = QComboBox()
            self.cluster_viz_method.addItems(["DBSCAN", "OPTICS", "HDBSCAN", "KMeans", "Density"])
            self.viz_params_layout.addRow("Method:", self.cluster_viz_method)
            
            self.cluster_viz_min_points = QSpinBox()
            self.cluster_viz_min_points.setRange(3, 100)
            self.cluster_viz_min_points.setValue(5)
            self.viz_params_layout.addRow("Min Points:", self.cluster_viz_min_points)
            
            self.cluster_viz_eps = QDoubleSpinBox()
            self.cluster_viz_eps.setRange(0.1, 100.0)
            self.cluster_viz_eps.setSingleStep(0.1)
            self.cluster_viz_eps.setValue(10.0)
            self.cluster_viz_eps.setSuffix(" px")
            self.viz_params_layout.addRow("Epsilon:", self.cluster_viz_eps)
            
            self.cluster_viz_show_background = QCheckBox()
            self.cluster_viz_show_background.setChecked(True)
            self.viz_params_layout.addRow("Show Background:", self.cluster_viz_show_background)
    
    def new_project(self):
        """Create a new project."""
        # Ask for confirmation if data exists
        if self.image_stack is not None or self.tracks_df is not None:
            reply = QMessageBox.question(
                self, "New Project", 
                "Creating a new project will discard current data. Continue?",
                QMessageBox.Yes | QMessageBox.No, QMessageBox.No
            )
            if reply == QMessageBox.No:
                return
        
        # Open settings dialog
        dialog = ProjectSettingsDialog(self)
        if dialog.exec_():
            self.project_settings = dialog.get_settings()
            
            # Reset data
            self.image_stack = None
            self.current_frame = 0
            self.detections = []
            self.tracks_df = None
            self.analysis_results = {}
            
            # Update UI
            self.update_project_info()
            self.update_data_summary()
            
            self.statusBar().showMessage("New project created", 3000)
    
    def open_project(self):
        """Open an existing project."""
        # Not fully implemented - would load project file with settings and data references
        QMessageBox.information(
            self, "Open Project", 
            "This feature is not fully implemented in this prototype."
        )
    
    def save_project(self):
        """Save the current project."""
        # Not fully implemented - would save project file with settings and data references
        QMessageBox.information(
            self, "Save Project", 
            "This feature is not fully implemented in this prototype."
        )
    
    def edit_project_settings(self):
        """Edit the current project settings."""
        dialog = ProjectSettingsDialog(self, self.project_settings)
        if dialog.exec_():
            self.project_settings = dialog.get_settings()
            self.update_project_info()
    
    def update_project_info(self):
        """Update the project information display."""
        self.project_name_label.setText(self.project_settings["project_name"])
        self.pixel_size_label.setText(f"{self.project_settings['pixel_size']} μm")
        self.frame_interval_label.setText(f"{self.project_settings['frame_interval']} s")
        self.temperature_label.setText(f"{self.project_settings['temperature']} °C")
        self.particle_radius_label.setText(f"{self.project_settings['particle_radius']} nm")
        self.project_notes.setText(self.project_settings["notes"])
    
    def update_data_summary(self):
        """Update the data summary display."""
        if self.image_stack is not None:
            self.image_stack_info.setText(
                f"{self.image_stack.shape[0]} frames, {self.image_stack.shape[1]}×{self.image_stack.shape[2]} pixels"
            )
        else:
            self.image_stack_info.setText("No image stack loaded")
        
        if self.detections:
            total_detections = sum(len(d) for d in self.detections)
            self.detections_info.setText(f"{total_detections} detections in {len(self.detections)} frames")
        else:
            self.detections_info.setText("No detections")
        
        if self.tracks_df is not None:
            num_tracks = len(self.tracks_df["track_id"].unique())
            self.tracks_info.setText(f"{num_tracks} tracks with {len(self.tracks_df)} positions")
        else:
            self.tracks_info.setText("No tracks")
        
        if self.analysis_results:
            self.analysis_info.setText(f"{len(self.analysis_results)} analysis results")
        else:
            self.analysis_info.setText("No analysis results")
    
    def import_image_stack(self):
        """Import an image stack from file."""
        file_dialog = QFileDialog()
        file_path, _ = file_dialog.getOpenFileName(
            self, "Open Image Stack", "", "Image Files (*.tif *.tiff);;All Files (*)"
        )
        
        if file_path:
            try:
                # Show progress in status bar
                self.statusBar().showMessage(f"Loading image stack from {file_path}...")
                
                # Load the image stack
                self.image_stack = load_image_stack(file_path)
                
                # Update frame slider ranges
                self.frame_slider.setMaximum(self.image_stack.shape[0] - 1)
                self.tracking_frame_slider.setMaximum(self.image_stack.shape[0] - 1)
                
                # Reset frame index
                self.current_frame = 0
                self.frame_slider.setValue(0)
                self.tracking_frame_slider.setValue(0)
                
                # Update display
                self.update_display_frame(0)
                self.update_data_summary()
                
                self.statusBar().showMessage(f"Loaded image stack with {self.image_stack.shape[0]} frames", 3000)
            except Exception as e:
                logger.error(f"Error loading image stack: {str(e)}", exc_info=True)
                QMessageBox.critical(self, "Error", f"Failed to load image stack: {str(e)}")
    
    def update_display_frame(self, frame_idx):
        """Update the displayed frame in the image tab."""
        if self.image_stack is None or frame_idx < 0 or frame_idx >= self.image_stack.shape[0]:
            return
        
        self.current_frame = frame_idx
        
        # Display the current frame
        self.image_canvas.axes.clear()
        self.image_canvas.axes.imshow(self.image_stack[frame_idx], cmap='gray')
        self.image_canvas.axes.set_title(f"Frame {frame_idx}")
        self.image_canvas.axes.axis('off')
        self.image_canvas.draw()
    
    def update_tracking_frame(self, frame_idx):
    if self.tracking_animation is not None:
        self.tracking_animation.event_source.stop()
    self.tracking_animation = self.setup_tracking_animation()
        
        # Display the current frame with detections and tracks
        self.tracking_canvas.axes.clear()
        self.tracking_canvas.axes.imshow(self.image_stack[frame_idx], cmap='gray')
        
        # Show detections if available and enabled
        if self.show_detections.isChecked() and self.detections and frame_idx < len(self.detections):
            frame_detections = self.detections[frame_idx]
            if frame_detections is not None and len(frame_detections) > 0:
                x = frame_detections[:, 0]
                y = frame_detections[:, 1]
                self.tracking_canvas.axes.scatter(x, y, s=50, facecolors='none', edgecolors='r')
        
        # Show tracks if available and enabled
        if self.show_tracks.isChecked() and self.tracks_df is not None:
            # Get track history length
            history = self.track_history.value()
            
            # Get unique track IDs
            track_ids = self.tracks_df["track_id"].unique()
            
            # Plot each track's history up to the current frame
            for track_id in track_ids:
                track = self.tracks_df[self.tracks_df["track_id"] == track_id]
                track = track[track["frame"] <= frame_idx]
                
                if len(track) > 0:
                    # Only show up to 'history' previous frames
                    if len(track) > history:
                        track = track.iloc[-history:]
                    
                    # Plot track line
                    self.tracking_canvas.axes.plot(track["x"], track["y"], 'b-', linewidth=1)
                    
                    # Mark current position
                    if track.iloc[-1]["frame"] == frame_idx:
                        self.tracking_canvas.axes.plot(
                            track.iloc[-1]["x"], track.iloc[-1]["y"], 
                            'bo', markersize=8
                        )
        
        self.tracking_canvas.axes.set_title(f"Frame {frame_idx}")
        self.tracking_canvas.axes.axis('off')
        self.tracking_canvas.draw()
    
    def update_tracking_display(self):
        """Update the tracking display based on current settings."""
        current_frame = self.tracking_frame_slider.value()
        self.update_tracking_frame(current_frame)
    
    def apply_image_processing(self):
        """Apply selected image processing operations."""
        if self.image_stack is None:
            QMessageBox.warning(self, "Warning", "No image stack loaded")
            return
        
        try:
            # Get processing parameters
            contrast_method = self.contrast_method.currentText().lower()
            denoise_method = self.denoise_method.currentText().lower()
            denoise_strength = self.denoise_strength.value()
            
            # Make a copy of the current frame for processing
            processed_frame = self.image_stack[self.current_frame].copy()
            
            # Apply contrast enhancement if selected
            if contrast_method != "none":
                processed_frame = enhance_contrast(
                    processed_frame, 
                    percentile=(2, 98), 
                    method=contrast_method
                )
            
            # Apply denoising if selected
            if denoise_method != "none":
                processed_frame = denoise_image(
                    processed_frame,
                    method=denoise_method,
                    strength=denoise_strength
                )
            
            # Display the processed frame
            self.image_canvas.axes.clear()
            self.image_canvas.axes.imshow(processed_frame, cmap='gray')
            self.image_canvas.axes.set_title(f"Frame {self.current_frame} (Processed)")
            self.image_canvas.axes.axis('off')
            self.image_canvas.draw()
            
            # Store the processed frame back to the image stack
            # Note: In a real application, you might want to store processed frames separately
            self.image_stack[self.current_frame] = processed_frame
            
            self.statusBar().showMessage("Image processing applied", 3000)
            
        except Exception as e:
            logger.error(f"Error in image processing: {str(e)}", exc_info=True)
            QMessageBox.critical(self, "Error", f"Image processing failed: {str(e)}")
    
    def reset_image_processing(self):
        """Reset the current frame to its original state."""
        # Note: In a real application, you would need to store the original frames
        QMessageBox.information(
            self, "Reset Processing", 
            "This feature would reset the current frame to its original state. "
            "Not implemented in this prototype."
        )
    
    def detect_particles(self):
        """Detect particles in the image stack."""
        if self.image_stack is None:
            QMessageBox.warning(self, "Warning", "No image stack loaded")
            return
        
        try:
            # Get detection parameters
            detection_method = self.detection_method.currentText().lower()
            min_sigma = self.min_sigma.value()
            max_sigma = self.max_sigma.value()
            num_sigma = self.num_sigma.value()
            threshold = self.threshold.value()
            
            # Prepare detector parameters
            detector_params = {
                "method": detection_method,
                "min_sigma": min_sigma,
                "max_sigma": max_sigma,
                "num_sigma": num_sigma,
                "threshold": threshold,
                "exclude_border": 2,
                "subpixel": True
            }
            
            # Create progress bar
            self.progress_bar.setValue(0)
            self.progress_bar.setVisible(True)
            
            # Create worker thread for detection
            self.worker = WorkerThread(
                "detect_particles", 
                {"detector_params": detector_params, "frames": self.image_stack}
            )
            self.worker.progress_updated.connect(self.update_progress)
            self.worker.operation_completed.connect(self.detection_completed)
            self.worker.error_occurred.connect(self.operation_error)
            
            # Disable UI during detection
            self.setEnabled(False)
            
            # Start the worker
            self.statusBar().showMessage("Detecting particles...")
            self.worker.start()
            
        except Exception as e:
            logger.error(f"Error in particle detection: {str(e)}", exc_info=True)
            QMessageBox.critical(self, "Error", f"Particle detection failed: {str(e)}")
            self.progress_bar.setVisible(False)
    
    def detection_completed(self, results):
        """Handle completion of particle detection."""
        self.detections = results
        
        # Update the current frame display
        self.update_tracking_frame(self.current_frame)
        
        # Update data summary
        self.update_data_summary()
        
        # Reset UI
        self.progress_bar.setVisible(False)
        self.setEnabled(True)
        
        # Show success message
        total_detections = sum(len(d) for d in self.detections)
        self.statusBar().showMessage(
            f"Detected {total_detections} particles across {len(self.detections)} frames", 
            3000
        )
        
        # Switch to tracking tab
        self.tabs.setCurrentIndex(2)  # Index of tracking tab
    
    def link_tracks(self):
        """Link detections into tracks."""
        if not self.detections:
            QMessageBox.warning(self, "Warning", "No particle detections available")
            return
        
        try:
            # Get linking parameters
            linking_method = self.linking_method.currentText().lower()
            max_distance = self.max_distance.value()
            max_gap_closing = self.max_gap_closing.value()
            
            # Prepare linker parameters
            linker_params = {
                "max_distance": max_distance,
                "max_gap": max_gap_closing
            }
            
            # Create progress bar
            self.progress_bar.setValue(0)
            self.progress_bar.setVisible(True)
            
            # Create worker thread for linking
            self.worker = WorkerThread(
                "track_particles", 
                {
                    "detections": self.detections,
                    "linker_method": linking_method,
                    "linker_params": linker_params
                }
            )
            self.worker.progress_updated.connect(self.update_progress)
            self.worker.operation_completed.connect(self.linking_completed)
            self.worker.error_occurred.connect(self.operation_error)
            
            # Disable UI during linking
            self.setEnabled(False)
            
            # Start the worker
            self.statusBar().showMessage("Linking tracks...")
            self.worker.start()
            
        except Exception as e:
            logger.error(f"Error in track linking: {str(e)}", exc_info=True)
            QMessageBox.critical(self, "Error", f"Track linking failed: {str(e)}")
            self.progress_bar.setVisible(False)
    
    def linking_completed(self, results):
        """Handle completion of track linking."""
        # Convert linked tracks to dataframe
        try:
            # This is a simplified conversion - in a real application, 
            # this would be handled by the linker's output format
            tracks_data = []
            for frame_idx, frame_links in enumerate(results):
                if frame_links is not None:
                    for track_id, detection_idx in frame_links.items():
                        # Get the detection coordinates
                        detection = self.detections[frame_idx][detection_idx]
                        x, y = detection[0], detection[1]
                        
                        # Add to tracks data
                        tracks_data.append({
                            "track_id": track_id,
                            "frame": frame_idx,
                            "x": x,
                            "y": y
                        })
            
            # Create tracks dataframe
            self.tracks_df = pd.DataFrame(tracks_data)
            
            # Update the current frame display
            self.update_tracking_frame(self.current_frame)
            
            # Update data summary
            self.update_data_summary()
            
        except Exception as e:
            logger.error(f"Error converting tracks to dataframe: {str(e)}", exc_info=True)
            QMessageBox.critical(self, "Error", f"Error creating tracks dataframe: {str(e)}")
        
        # Reset UI
        self.progress_bar.setVisible(False)
        self.setEnabled(True)
        
        # Show success message
        if self.tracks_df is not None:
            num_tracks = len(self.tracks_df["track_id"].unique())
            self.statusBar().showMessage(
                f"Linked {num_tracks} tracks with {len(self.tracks_df)} positions", 
                3000
            )
    
    def filter_tracks(self):
        """Filter tracks based on length."""
        if self.tracks_df is None:
            QMessageBox.warning(self, "Warning", "No tracks available")
            return
        
        try:
            min_length = self.min_track_length.value()
            
            # Count track lengths
            track_lengths = self.tracks_df.groupby("track_id").size()
            
            # Filter tracks
            valid_tracks = track_lengths[track_lengths >= min_length].index
            
            # Apply filter
            filtered_tracks = self.tracks_df[self.tracks_df["track_id"].isin(valid_tracks)]
            
            # Show results
            removed = len(self.tracks_df["track_id"].unique()) - len(valid_tracks)
            
            reply = QMessageBox.question(
                self, "Filter Tracks", 
                f"Filtering will remove {removed} tracks that are shorter than {min_length} frames. Continue?",
                QMessageBox.Yes | QMessageBox.No, QMessageBox.Yes
            )
            
            if reply == QMessageBox.Yes:
                self.tracks_df = filtered_tracks
                
                # Update the current frame display
                self.update_tracking_frame(self.current_frame)
                
                # Update data summary
                self.update_data_summary()
                
                self.statusBar().showMessage(
                    f"Filtered tracks: {len(valid_tracks)} tracks remaining",
                    3000
                )
        
        except Exception as e:
            logger.error(f"Error filtering tracks: {str(e)}", exc_info=True)
            QMessageBox.critical(self, "Error", f"Track filtering failed: {str(e)}")
    
    def import_tracks(self):
        """Import tracks from file."""
        file_dialog = QFileDialog()
        file_path, _ = file_dialog.getOpenFileName(
            self, "Open Tracks File", "", "CSV Files (*.csv);;HDF5 Files (*.h5);;All Files (*)"
        )
        
        if file_path:
            try:
                # Show progress in status bar
                self.statusBar().showMessage(f"Loading tracks from {file_path}...")
                
                # Load the tracks
                self.tracks_df = load_tracks(file_path)
                
                # Update display
                self.update_tracking_frame(self.current_frame)
                self.update_data_summary()
                
                num_tracks = len(self.tracks_df["track_id"].unique())
                self.statusBar().showMessage(
                    f"Loaded {num_tracks} tracks with {len(self.tracks_df)} positions", 
                    3000
                )
            except Exception as e:
                logger.error(f"Error loading tracks: {str(e)}", exc_info=True)
                QMessageBox.critical(self, "Error", f"Failed to load tracks: {str(e)}")
    
    def export_tracks(self):
        """Export tracks to file."""
        if self.tracks_df is None:
            QMessageBox.warning(self, "Warning", "No tracks available to export")
            return
        
        file_dialog = QFileDialog()
        file_path, _ = file_dialog.getSaveFileName(
            self, "Save Tracks", "", "CSV Files (*.csv);;HDF5 Files (*.h5);;All Files (*)"
        )
        
        if file_path:
            try:
                # Show progress in status bar
                self.statusBar().showMessage(f"Saving tracks to {file_path}...")
                
                # Save the tracks
                save_tracks(self.tracks_df, file_path)
                
                self.statusBar().showMessage(f"Tracks saved to {file_path}", 3000)
            except Exception as e:
                logger.error(f"Error saving tracks: {str(e)}", exc_info=True)
                QMessageBox.critical(self, "Error", f"Failed to save tracks: {str(e)}")
    
    def run_analysis(self):
        """Run the selected analysis on the tracks."""
        if self.tracks_df is None:
            QMessageBox.warning(self, "Warning", "No tracks available for analysis")
            return
        
        try:
            analysis_type = self.analysis_type.currentText()
            
            # Create progress bar
            self.progress_bar.setValue(0)
            self.progress_bar.setVisible(True)
            
            # Set up parameters based on analysis type
            if analysis_type == "Diffusion Analysis":
                # Get diffusion analysis parameters
                max_lag = self.msd_max_lag.value()
                fit_points = self.msd_fit_points.value()
                model = self.model_selection.currentText().lower()
                
                # Create diffusion analysis parameters
                analysis_params = {
                    "tracks_df": self.tracks_df,
                    "pixel_size": self.project_settings["pixel_size"],
                    "frame_interval": self.project_settings["frame_interval"],
                    "max_lag": max_lag,
                    "fit_points": fit_points,
                    "model": model
                }
                
                # Run diffusion analysis (this would be in a worker thread in a real app)
                results = self.run_diffusion_analysis(analysis_params)
                
                # Store results
                self.analysis_results["diffusion"] = results
                
                # Display results
                self.display_analysis_results(results, "diffusion")
                
            elif analysis_type == "Active Transport":
                # Get active transport parameters
                min_alpha = self.min_superdiffusive_alpha.value()
                min_track_length = self.min_track_length_at.value()
                velocity_analysis = self.velocity_analysis.isChecked()
                
                # Create active transport analyzer
                at_analyzer = ActiveTransportAnalyzer(
                    dt=self.project_settings["frame_interval"],
                    min_track_length=min_track_length,
                    min_superdiffusive_alpha=min_alpha
                )
                
                # Run active transport analysis
                results = at_analyzer.analyze_tracks(self.tracks_df)
                
                # Store results
                self.analysis_results["active_transport"] = results
                
                # Display results
                self.display_analysis_results(results, "active_transport")
                
            # Add more analysis types as needed
                
            else:
                QMessageBox.information(
                    self, "Analysis", 
                    f"Analysis type '{analysis_type}' not fully implemented in this prototype."
                )
                self.progress_bar.setVisible(False)
                return
            
            # Update data summary
            self.update_data_summary()
            
            # Reset UI
            self.progress_bar.setVisible(False)
            
            # Show success message
            self.statusBar().showMessage(f"{analysis_type} completed successfully", 3000)
            
        except Exception as e:
            logger.error(f"Error in analysis: {str(e)}", exc_info=True)
            QMessageBox.critical(self, "Error", f"Analysis failed: {str(e)}")
            self.progress_bar.setVisible(False)
    
    def run_diffusion_analysis(self, params):
        """Run diffusion analysis on tracks."""
        # This is a simplified implementation
        try:
            # Import necessary functions
            from spt_analyzer.analysis.diffusion_models import compute_msd, fit_diffusion_models
            
            # Get parameters
            tracks_df = params["tracks_df"]
            pixel_size = params["pixel_size"]
            dt = params["frame_interval"]
            max_lag = params.get("max_lag", 20)
            fit_points = params.get("fit_points", 5)
            
            # Get unique track IDs
            track_ids = tracks_df["track_id"].unique()
            
            # Results container
            results = {
                "track_ids": [],
                "diffusion_coefficients": [],
                "alpha_values": [],
                "confinement_sizes": [],
                "goodness_of_fit": [],
                "selected_models": []
            }
            
            # Process each track
            for i, track_id in enumerate(track_ids):
                # Update progress
                progress = int((i + 1) / len(track_ids) * 100)
                self.progress_bar.setValue(progress)
                QApplication.processEvents()  # Allow GUI updates
                
                # Get track data
                track = tracks_df[tracks_df["track_id"] == track_id]
                
                # Skip short tracks
                if len(track) < 5:
                    continue
                
                # Calculate MSD
                msd = compute_msd(track, pixel_size, dt, max_lag=max_lag)
                
                # Fit diffusion models
                fit_results = fit_diffusion_models(
                    msd, dt, max_fit_points=fit_points
                )
                
                # Store results
                results["track_ids"].append(track_id)
                results["diffusion_coefficients"].append(fit_results["brownian"]["D"])
                results["alpha_values"].append(fit_results["anomalous"]["alpha"])
                results["confinement_sizes"].append(
                    fit_results["confined"]["L"] if "confined" in fit_results else np.nan
                )
                results["goodness_of_fit"].append(fit_results["best_model_r2"])
                results["selected_models"].append(fit_results["best_model"])
            
            # Convert to DataFrames for easier handling
            results_df = pd.DataFrame({
                "track_id": results["track_ids"],
                "D": results["diffusion_coefficients"],
                "alpha": results["alpha_values"],
                "L": results["confinement_sizes"],
                "r2": results["goodness_of_fit"],
                "model": results["selected_models"]
            })
            
            return {
                "results_df": results_df,
                "params": params,
                "summary": {
                    "mean_D": results_df["D"].mean(),
                    "median_D": results_df["D"].median(),
                    "mean_alpha": results_df["alpha"].mean(),
                    "model_counts": results_df["model"].value_counts().to_dict()
                }
            }
            
        except Exception as e:
            logger.error(f"Error in diffusion analysis: {str(e)}", exc_info=True)
            raise
    
    def display_analysis_results(self, results, analysis_type):
        """Display analysis results in the results area."""
        try:
            # Clear current displays
            self.results_summary.clear()
            self.results_table.clear()
            self.results_canvas.axes.clear()
            
            if analysis_type == "diffusion":
                # Display summary
                summary_text = f"""
                Diffusion Analysis Results:
                
                Number of tracks analyzed: {len(results['results_df'])}
                
                Mean diffusion coefficient: {results['summary']['mean_D']:.4f} μm²/s
                Median diffusion coefficient: {results['summary']['median_D']:.4f} μm²/s
                Mean anomalous exponent: {results['summary']['mean_alpha']:.4f}
                
                Model selection:
                """
                
                for model, count in results['summary']['model_counts'].items():
                    summary_text += f"  - {model}: {count} tracks\n"
                
                self.results_summary.setText(summary_text)
                
                # Display table
                table_data = results['results_df']
                self.results_table.setRowCount(len(table_data))
                self.results_table.setColumnCount(6)
                self.results_table.setHorizontalHeaderLabels(
                    ["Track ID", "D (μm²/s)", "Alpha", "L (μm)", "R²", "Model"]
                )
                
                for i, (_, row) in enumerate(table_data.iterrows()):
                    self.results_table.setItem(i, 0, QTableWidgetItem(str(row["track_id"])))
                    self.results_table.setItem(i, 1, QTableWidgetItem(f"{row['D']:.4f}"))
                    self.results_table.setItem(i, 2, QTableWidgetItem(f"{row['alpha']:.4f}"))
                    self.results_table.setItem(i, 3, QTableWidgetItem(f"{row['L']:.4f}" if not np.isnan(row['L']) else "N/A"))
                    self.results_table.setItem(i, 4, QTableWidgetItem(f"{row['r2']:.4f}"))
                    self.results_table.setItem(i, 5, QTableWidgetItem(str(row["model"])))
                
                # Display plot (histogram of diffusion coefficients)
                self.results_canvas.axes.hist(
                    table_data["D"], bins=30, alpha=0.7, color='blue'
                )
                self.results_canvas.axes.set_xlabel('Diffusion Coefficient (μm²/s)')
                self.results_canvas.axes.set_ylabel('Frequency')
                self.results_canvas.axes.set_title('Diffusion Coefficient Distribution')
                
                # Log scale for x-axis often helps with diffusion coefficient visualization
                self.results_canvas.axes.set_xscale('log')
                
                self.results_canvas.draw()
                
            elif analysis_type == "active_transport":
                # Display active transport results (simplified)
                summary_text = "Active Transport Analysis Results:\n\n"
                
                if "superdiffusive_tracks" in results:
                    num_superdiffusive = len(results["superdiffusive_tracks"])
                    total_tracks = len(set(self.tracks_df["track_id"]))
                    
                    summary_text += f"Detected {num_superdiffusive} superdiffusive tracks out of {total_tracks} total tracks.\n"
                    summary_text += f"Percentage of active transport: {num_superdiffusive/total_tracks*100:.1f}%\n\n"
                    
                    if "velocities" in results:
                        mean_velocity = np.mean(results["velocities"])
                        max_velocity = np.max(results["velocities"])
                        
                        summary_text += f"Mean velocity: {mean_velocity:.4f} μm/s\n"
                        summary_text += f"Max velocity: {max_velocity:.4f} μm/s\n"
                
                self.results_summary.setText(summary_text)
                
                # In a real implementation, you would add more detailed results display
            
            # Switch to the results tab
            self.results_tabs.setCurrentIndex(0)  # Summary tab
            
        except Exception as e:
            logger.error(f"Error displaying analysis results: {str(e)}", exc_info=True)
            QMessageBox.critical(self, "Error", f"Failed to display results: {str(e)}")
    
    def generate_visualization(self):
    viz_type = self.viz_type.currentText()

    if viz_type == "Track Trajectories":
        self.create_interactive_track_plot()
    elif viz_type == "Diffusion Map":
        if self.viz_animation is not None:
            self.viz_animation.event_source.stop()
        self.viz_animation = self.setup_diffusion_map_animation()
    elif viz_type == "MSD Curves":
        self.create_msd_plot()
        
        try:
            viz_type = self.viz_type.currentText()
            
            # Clear current canvas
            self.viz_canvas.axes.clear()
            
            if viz_type == "Track Trajectories":
                # Get visualization parameters
                color_by = self.color_by.currentText().lower().replace(" ", "_")
                colormap = self.colormap.currentText()
                show_background = self.show_background.isChecked()
                line_width = self.track_line_width.value()
                
                # Get background image if needed
                background = None
                if show_background and self.image_stack is not None:
                    background = np.max(self.image_stack, axis=0)  # Max projection
                
                # Generate visualization
                if color_by == "track_id":
                    plot_tracks(
                        self.tracks_df, background=background, 
                        colorby="track_id", cmap=colormap, 
                        linewidth=line_width, ax=self.viz_canvas.axes
                    )
                elif color_by == "diffusion_coefficient":
                    # Check if diffusion analysis results exist
                    if "diffusion" in self.analysis_results:
                        # Get diffusion coefficients
                        diff_results = self.analysis_results["diffusion"]["results_df"]
                        
                        # Create a merged dataframe with tracks and diffusion coefficients
                        merged_df = pd.merge(
                            self.tracks_df, 
                            diff_results[["track_id", "D"]], 
                            on="track_id", how="left"
                        )
                        
                        # Plot with diffusion coefficients as color
                        plot_tracks(
                            merged_df, background=background, 
                            colorby="D", cmap=colormap, 
                            linewidth=line_width, ax=self.viz_canvas.axes
                        )
                    else:
                        # Fall back to track_id coloring
                        QMessageBox.warning(
                            self, "Warning", 
                            "No diffusion analysis results available. Using track ID for coloring."
                        )
                        plot_tracks(
                            self.tracks_df, background=background, 
                            colorby="track_id", cmap=colormap, 
                            linewidth=line_width, ax=self.viz_canvas.axes
                        )
                elif color_by == "time":
                    # Color by frame/time
                    plot_tracks(
                        self.tracks_df, background=background, 
                        colorby="frame", cmap=colormap, 
                        linewidth=line_width, ax=self.viz_canvas.axes
                    )
                else:
                    # Default to track_id
                    plot_tracks(
                        self.tracks_df, background=background, 
                        colorby="track_id", cmap=colormap, 
                        linewidth=line_width, ax=self.viz_canvas.axes
                    )
                
            elif viz_type == "Diffusion Map":
                # Get visualization parameters
                colormap = self.diff_colormap.currentText()
                log_scale = self.log_scale.isChecked()
                show_background = self.show_diff_background.isChecked()
                diff_min = self.diff_min.value()
                diff_max = self.diff_max.value()
                
                # Check if diffusion analysis results exist
                if "diffusion" in self.analysis_results:
                    # Get diffusion coefficients
                    diff_results = self.analysis_results["diffusion"]["results_df"]
                    
                    # Get background image if needed
                    background = None
                    if show_background and self.image_stack is not None:
                        background = np.max(self.image_stack, axis=0)  # Max projection
                    
                    # Create a merged dataframe with tracks and diffusion coefficients
                    merged_df = pd.merge(
                        self.tracks_df, 
                        diff_results[["track_id", "D"]], 
                        on="track_id", how="left"
                    )
                    
                    # Generate diffusion map
                    plot_diffusion_map(
                        self.tracks_df, diff_results, background=background,
                        cmap=colormap, ax=self.viz_canvas.axes,
                        pixel_size=self.project_settings["pixel_size"],
                        colorbar_label="D (μm²/s)"
                    )
                    
                    # Set log scale if selected
                    if log_scale:
                        self.viz_canvas.axes.collections[0].norm.vmin = diff_min
                        self.viz_canvas.axes.collections[0].norm.vmax = diff_max
                else:
                    QMessageBox.warning(
                        self, "Warning", 
                        "No diffusion analysis results available. Please run diffusion analysis first."
                    )
                    return
                
            elif viz_type == "MSD Curves":
                # Get visualization parameters
                max_lag = self.msd_viz_max_lag.value()
                group_by = self.msd_viz_group_by.currentText().lower().replace(" ", "_")
                num_groups = self.msd_viz_num_groups.value()
                show_fit = self.msd_viz_show_fit.isChecked()
                show_errorbars = self.msd_viz_show_errorbars.isChecked()
                
                # Check if diffusion analysis was performed
                if "diffusion" not in self.analysis_results:
                    QMessageBox.warning(
                        self, "Warning", 
                        "No diffusion analysis results available. Please run diffusion analysis first."
                    )
                    return
                
                # This is a simplified implementation of MSD curve visualization
                # In a real application, you would calculate MSD curves for all tracks
                
                # Import necessary functions
                from spt_analyzer.analysis.diffusion_models import compute_msd, fit_diffusion_models
                
                # Get track IDs
                track_ids = self.tracks_df["track_id"].unique()
                
                # If grouping, organize tracks by the grouping criterion
                if group_by != "none" and group_by in self.analysis_results["diffusion"]["results_df"].columns:
                    # Get the values for grouping
                    group_values = self.analysis_results["diffusion"]["results_df"][group_by].values
                    
                    # Create groups
                    try:
                        # Use KMeans to group values
                        from sklearn.cluster import KMeans
                        kmeans = KMeans(n_clusters=num_groups, random_state=0)
                        
                        # Reshape for KMeans
                        kmeans.fit(group_values.reshape(-1, 1))
                        
                        # Get group labels
                        group_labels = kmeans.labels_
                        
                        # Map track IDs to groups
                        track_groups = dict(zip(
                            self.analysis_results["diffusion"]["results_df"]["track_id"],
                            group_labels
                        ))
                    except Exception as e:
                        logger.error(f"Error creating groups: {str(e)}", exc_info=True)
                        track_groups = {track_id: 0 for track_id in track_ids}  # Default to single group
                else:
                    # No grouping - all tracks in one group
                    track_groups = {track_id: 0 for track_id in track_ids}
                
                # Calculate and plot MSD curves by group
                pixel_size = self.project_settings["pixel_size"]
                dt = self.project_settings["frame_interval"]
                
                # Prepare colors
                import matplotlib.cm as cm
                import matplotlib.colors as mcolors
                
                num_groups_actual = len(set(track_groups.values()))
                colors = cm.viridis(np.linspace(0, 1, num_groups_actual))
                
                # Plot MSD curves by group
                for group_idx in range(num_groups_actual):
                    # Get tracks in this group
                    group_track_ids = [tid for tid, gid in track_groups.items() if gid == group_idx]
                    
                    # Skip empty groups
                    if not group_track_ids:
                        continue
                    
                    # Calculate average MSD for this group
                    group_msds = []
                    
                    for track_id in group_track_ids:
                        # Get track data
                        track = self.tracks_df[self.tracks_df["track_id"] == track_id]
                        
                        # Skip short tracks
                        if len(track) < 5:
                            continue
                        
                        # Calculate MSD
                        msd = compute_msd(track, pixel_size, dt, max_lag=max_lag)
                        group_msds.append(msd)
                    
                    # Calculate average MSD
                    if group_msds:
                        # Pad shorter MSDs
                        max_len = max(len(m) for m in group_msds)
                        padded_msds = []
                        for m in group_msds:
                            if len(m) < max_len:
                                padded = np.full(max_len, np.nan)
                                padded[:len(m)] = m
                                padded_msds.append(padded)
                            else:
                                padded_msds.append(m)
                        
                        # Stack and calculate mean/std
                        msd_stack = np.vstack(padded_msds)
                        mean_msd = np.nanmean(msd_stack, axis=0)
                        std_msd = np.nanstd(msd_stack, axis=0)
                        
                        # Plot
                        lags = np.arange(1, len(mean_msd) + 1)
                        time_lags = lags * dt
                        
                        label = f"Group {group_idx+1}" if group_by != "none" else "All Tracks"
                        
                        if show_errorbars:
                            self.viz_canvas.axes.errorbar(
                                time_lags, mean_msd, yerr=std_msd,
                                color=colors[group_idx], marker='o', linestyle='-',
                                label=label
                            )
                        else:
                            self.viz_canvas.axes.plot(
                                time_lags, mean_msd,
                                color=colors[group_idx], marker='o', linestyle='-',
                                label=label
                            )
                
                # Set labels and title
                self.viz_canvas.axes.set_xlabel('Time Lag (s)')
                self.viz_canvas.axes.set_ylabel('MSD (μm²)')
                self.viz_canvas.axes.set_title('Mean Square Displacement Curves')
                
                # Add legend
                self.viz_canvas.axes.legend()
                
                # Set log-log scale
                self.viz_canvas.axes.set_xscale('log')
                self.viz_canvas.axes.set_yscale('log')
                
            # Add more visualization types as needed
                
            else:
                QMessageBox.information(
                    self, "Visualization", 
                    f"Visualization type '{viz_type}' not fully implemented in this prototype."
                )
                return
            
            # Update canvas
            self.viz_canvas.fig.tight_layout()
            self.viz_canvas.draw()
            
            # Show success message
            self.statusBar().showMessage(f"{viz_type} visualization generated", 3000)
            
        except Exception as e:
            logger.error(f"Error generating visualization: {str(e)}", exc_info=True)
            QMessageBox.critical(self, "Error", f"Failed to generate visualization: {str(e)}")
    
    def export_results(self):
        """Export analysis results to file."""
        if not self.analysis_results:
            QMessageBox.warning(self, "Warning", "No analysis results available to export")
            return
        
        file_dialog = QFileDialog()
        file_path, _ = file_dialog.getSaveFileName(
            self, "Save Analysis Results", "", 
            "CSV Files (*.csv);;Excel Files (*.xlsx);;JSON Files (*.json);;All Files (*)"
        )
        
        if file_path:
            try:
                # Determine file type
                _, ext = os.path.splitext(file_path)
                
                # Export based on file type
                if ext.lower() == '.csv':
                    # For CSV, export the main results dataframe
                    if "diffusion" in self.analysis_results:
                        self.analysis_results["diffusion"]["results_df"].to_csv(file_path, index=False)
                    else:
                        # Export first available results
                        result_key = list(self.analysis_results.keys())[0]
                        if "results_df" in self.analysis_results[result_key]:
                            self.analysis_results[result_key]["results_df"].to_csv(file_path, index=False)
                        else:
                            QMessageBox.warning(
                                self, "Export Warning", 
                                "No suitable DataFrame found in results for CSV export"
                            )
                            return
                
                elif ext.lower() == '.xlsx':
                    # For Excel, export multiple sheets
                    with pd.ExcelWriter(file_path) as writer:
                        for analysis_type, results in self.analysis_results.items():
                            if "results_df" in results:
                                results["results_df"].to_excel(writer, sheet_name=analysis_type, index=False)
                
                elif ext.lower() == '.json':
                    # For JSON, export all results that can be serialized
                    import json
                    
                    # Convert results to serializable format
                    serializable_results = {}
                    for analysis_type, results in self.analysis_results.items():
                        serializable_results[analysis_type] = {}
                        
                        # Convert DataFrames to dict
                        if "results_df" in results:
                            serializable_results[analysis_type]["results"] = results["results_df"].to_dict(orient="records")
                        
                        # Include summary
                        if "summary" in results:
                            serializable_results[analysis_type]["summary"] = results["summary"]
                    
                    with open(file_path, 'w') as f:
                        json.dump(serializable_results, f, indent=2)
                
                else:
                    # Default to CSV
                    if "diffusion" in self.analysis_results:
                        self.analysis_results["diffusion"]["results_df"].to_csv(file_path, index=False)
                    else:
                        # Export first available results
                        result_key = list(self.analysis_results.keys())[0]
                        if "results_df" in self.analysis_results[result_key]:
                            self.analysis_results[result_key]["results_df"].to_csv(file_path, index=False)
                        else:
                            QMessageBox.warning(
                                self, "Export Warning", 
                                "No suitable DataFrame found in results for export"
                            )
                            return
                
                self.statusBar().showMessage(f"Results exported to {file_path}", 3000)
                
            except Exception as e:
                logger.error(f"Error exporting results: {str(e)}", exc_info=True)
                QMessageBox.critical(self, "Error", f"Failed to export results: {str(e)}")
    
    def export_figure(self):
        """Export the current figure to file."""
        # Determine which canvas to export
        active_tab = self.tabs.currentIndex()
        
        if active_tab == 1:  # Image processing tab
            canvas = self.image_canvas
        elif active_tab == 2:  # Tracking tab
            canvas = self.tracking_canvas
        elif active_tab == 3:  # Analysis tab
            canvas = self.results_canvas
        elif active_tab == 4:  # Visualization tab
            canvas = self.viz_canvas
        else:
            QMessageBox.warning(self, "Warning", "No figure available to export on this tab")
            return
        
        file_dialog = QFileDialog()
        file_path, _ = file_dialog.getSaveFileName(
            self, "Save Figure", "", 
            "PNG Files (*.png);;PDF Files (*.pdf);;SVG Files (*.svg);;All Files (*)"
        )
        
        if file_path:
            try:
                # Save the figure
                canvas.fig.savefig(file_path, dpi=300, bbox_inches='tight')
                
                self.statusBar().showMessage(f"Figure exported to {file_path}", 3000)
                
            except Exception as e:
                logger.error(f"Error exporting figure: {str(e)}", exc_info=True)
                QMessageBox.critical(self, "Error", f"Failed to export figure: {str(e)}")
    
    def add_batch_dataset(self):
        """Add a dataset to the batch processing list."""
        file_dialog = QFileDialog()
        file_path, _ = file_dialog.getOpenFileName(
            self, "Open Image Stack or Tracks File", "", 
            "Image Files (*.tif *.tiff);;CSV Files (*.csv);;HDF5 Files (*.h5);;All Files (*)"
        )
        
        if file_path:
            # Add to list
            self.dataset_list.addItem(file_path)
    
    def remove_batch_dataset(self):
        """Remove the selected dataset from the batch processing list."""
        selected_items = self.dataset_list.selectedItems()
        if not selected_items:
            return
        
        for item in selected_items:
            row = self.dataset_list.row(item)
            self.dataset_list.takeItem(row)
    
    def select_batch_export_dir(self):
        """Select directory for batch export."""
        dir_path = QFileDialog.getExistingDirectory(
            self, "Select Export Directory", ""
        )
        
        if dir_path:
            self.batch_export_dir.setText(dir_path)
    
    def run_batch_process(self):
        """Run batch processing on all datasets."""
        # Get datasets
        datasets = []
        for i in range(self.dataset_list.count()):
            datasets.append(self.dataset_list.item(i).text())
        
        if not datasets:
            QMessageBox.warning(self, "Warning", "No datasets selected for batch processing")
            return
        
        # Get export directory
        export_dir = self.batch_export_dir.text()
        if not export_dir:
            QMessageBox.warning(self, "Warning", "No export directory selected")
            return
        
        # Get analysis type
        analysis_type = self.batch_analysis_type.currentText()
        
        # Confirm batch processing
        reply = QMessageBox.question(
            self, "Batch Processing", 
            f"Run {analysis_type} on {len(datasets)} datasets?\nResults will be saved to {export_dir}",
            QMessageBox.Yes | QMessageBox.No, QMessageBox.No
        )
        
        if reply == QMessageBox.No:
            return
        
        # This is a simplified implementation
        QMessageBox.information(
            self, "Batch Processing", 
            "Batch processing not fully implemented in this prototype."
        )
    
    def update_progress(self, value):
        """Update the progress bar."""
        self.progress_bar.setValue(value)
    
    def operation_error(self, error_message):
        """Handle operation errors."""
        logger.error(f"Operation error: {error_message}")
        QMessageBox.critical(self, "Error", f"Operation failed: {error_message}")
        
        # Reset UI
        self.progress_bar.setVisible(False)
        self.setEnabled(True)
    
    def show_about_dialog(self):
        """Show the about dialog."""
        QMessageBox.about(
            self, "About SPT Analyzer", 
            "SPT Analyzer\n\n"
            "A comprehensive toolkit for Single Particle Tracking analysis.\n\n"
            "This application integrates particle detection, track linking, and "
            "various analysis modules for detailed characterization of particle "
            "dynamics in microscopy data."
        )
    
    def show_documentation(self):
        """Show the documentation."""
        QMessageBox.information(
            self, "Documentation", 
            "Documentation not implemented in this prototype."
        )
    
    def load_app_settings(self):
        """Load application settings."""
        # In a full implementation, this would load window position, size, etc.
        pass
    
    def save_app_settings(self):
        """Save application settings."""
        # In a full implementation, this would save window position, size, etc.
        pass
    
    def closeEvent(self, event):
        """Handle window close event."""
        # Save settings before closing
        self.save_app_settings()
        event.accept()


def main():
    """Main application entry point."""
    # Set up logging
    setup_logging(log_level=logging.INFO)
    
    # Create application
    app = QApplication(sys.argv)
    
    # Create main window
    main_win = SPTAnalyzerGUI()
    main_win.show()
    
    # Run the application
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
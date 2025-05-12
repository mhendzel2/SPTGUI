# visualization/analysis_plots.py

import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Optional
from matplotlib.figure import Figure
import seaborn as sns
Import LogNorm from matplotlib.colors.

class MultiChannelAnalysisWidget(QWidget):
    """Widget for multi-channel analysis in the GUI."""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.manager = MultiChannelManager()
        self.setup_ui()
        
    def setup_ui(self):
        """Set up the UI components."""
        layout = QVBoxLayout()
        
        # Channel Management Group
        channel_group = QGroupBox("Channel Management")
        channel_layout = QVBoxLayout()
        
        # Channel list
        self.channel_list = QTableWidget()
        self.channel_list.setColumnCount(5)
        self.channel_list.setHorizontalHeaderLabels([
            "Name", "Type", "Role", "Particle Type", "Status"
        ])
        channel_layout.addWidget(self.channel_list)
        
        # Channel controls
        controls_layout = QHBoxLayout()
        
        self.add_channel_btn = QPushButton("Add Channel")
        self.remove_channel_btn = QPushButton("Remove Channel")
        self.edit_channel_btn = QPushButton("Edit Channel")
        
        controls_layout.addWidget(self.add_channel_btn)
        controls_layout.addWidget(self.remove_channel_btn)
        controls_layout.addWidget(self.edit_channel_btn)
        
        channel_layout.addLayout(controls_layout)
        channel_group.setLayout(channel_layout)
        layout.addWidget(channel_group)
        
        # Analysis Group
        analysis_group = QGroupBox("Analysis Settings")
        analysis_layout = QFormLayout()
        
        # Compartment selection
        self.compartment_combo = QComboBox()
        self.compartment_combo.addItem("None")
        analysis_layout.addRow("Compartment Channel:", self.compartment_combo)
        
        # Threshold settings
        self.threshold_spinbox = QDoubleSpinBox()
        self.threshold_spinbox.setRange(0, 1000)
        self.threshold_spinbox.setValue(100)
        analysis_layout.addRow("Threshold:", self.threshold_spinbox)
        
        # Analysis type
        self.analysis_type_combo = QComboBox()
        self.analysis_type_combo.addItems([
            "Colocalization",
            "Cross-Correlation",
            "Intensity Correlation",
            "Distance Analysis"
        ])
        analysis_layout.addRow("Analysis Type:", self.analysis_type_combo)
        
        analysis_group.setLayout(analysis_layout)
        layout.addWidget(analysis_group)
        
        # Results Group
        results_group = QGroupBox("Results")
        results_layout = QVBoxLayout()
        
        # Results tabs
        self.results_tabs = QTabWidget()
        
        # Summary tab
        summary_tab = QWidget()
        summary_layout = QVBoxLayout()
        self.results_text = QTextEdit()
        self.results_text.setReadOnly(True)
        summary_layout.addWidget(self.results_text)
        summary_tab.setLayout(summary_layout)
        self.results_tabs.addTab(summary_tab, "Summary")
        
        # Plot tab
        plot_tab = QWidget()
        plot_layout = QVBoxLayout()
        self.plot_canvas = MplCanvas(width=6, height=4)
        plot_layout.addWidget(self.plot_canvas)
        plot_tab.setLayout(plot_layout)
        self.results_tabs.addTab(plot_tab, "Plot")
        
        results_layout.addWidget(self.results_tabs)
class ChannelDialog(QDialog):
    """Dialog for adding or editing a channel."""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Channel Configuration")
        self.setup_ui()
        
    def setup_ui(self):
        """Set up the dialog UI."""
        layout = QFormLayout()
        
        # Channel name
        self.name_edit = QLineEdit()
        layout.addRow("Name:", self.name_edit)
        
        # Channel type
        self.type_combo = QComboBox()
        self.type_combo.addItems(['DAPI', 'Tracking', 'Membrane', 'Other'])
        layout.addRow("Type:", self.type_combo)
        
        # Channel role
        self.role_combo = QComboBox()
        self.role_combo.addItems(['None', 'Compartment Marker', 'Tracking Channel'])
        layout.addRow("Role:", self.role_combo)
        
        # Particle type (for tracking channels)
        self.particle_type_edit = QLineEdit()
        self.particle_type_edit.setEnabled(False)
        layout.addRow("Particle Type:", self.particle_type_edit)
        
        # Import data button
        self.import_btn = QPushButton("Import Channel Data")
        layout.addRow("", self.import_btn)
        
        # Dialog buttons
        buttons = QDialogButtonBox(
            QDialogButtonBox.Ok | QDialogButtonBox.Cancel,
            Qt.Horizontal, self)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addRow(buttons)
        
        self.setLayout(layout)
        
        # Connect signals
        self.role_combo.currentTextChanged.connect(self.update_particle_type_field)
        self.import_btn.clicked.connect(self.import_channel_data)
        
    def update_particle_type_field(self, role):
        """Enable/disable particle type field based on role."""
        self.particle_type_edit.setEnabled(role == 'Tracking Channel')
        
    def import_channel_data(self):
        """Import channel image data."""
        file_name, _ = QFileDialog.getOpenFileName(
            self, "Import Channel Data",
            "", "Image Files (*.tif *.tiff);;All Files (*)"
        )
        if file_name:
            try:
                self.channel_data = tifffile.imread(file_name)
                QMessageBox.information(self, "Success", 
                    f"Imported channel data with shape {self.channel_data.shape}")
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to import data: {str(e)}")
                
    def get_channel_data(self):
        """Return the channel configuration data."""
        return {
            'name': self.name_edit.text(),
            'type': self.type_combo.currentText(),
            'role': self.role_combo.currentText(),
            'particle_type': self.particle_type_edit.text() if self.particle_type_edit.isEnabled() else None,
            'data': getattr(self, 'channel_data', None)
        }
# In SPTAnalyzerGUI class, modify create_analysis_tab method:

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
    """Update analysis parameters based on selected type."""
    analysis_type = self.analysis_type.currentText()

    # Show/hide appropriate widgets
    self.multi_channel_widget.setVisible(analysis_type == "Multi-Channel Analysis")

    # ... rest of existing code ...           
class AnalysisPlotter:
    """Creates visualization plots for various analysis results"""

    @staticmethod
    def plot_msd_curves(self, ax, results):
    """Plot MSD curves with model fits."""
    if self.diffusion_widget.show_individual_tracks.isChecked():
        for track_id, track_data in results['track_results'].items():
            ax.plot(track_data['lag_times'], track_data['msd'],
                   'k-', alpha=0.1, linewidth=0.5)

    if self.diffusion_widget.show_ensemble_average.isChecked():
        ax.plot(results['ensemble_results']['lag_times'],
                results['ensemble_results']['msd'],
                'r-', linewidth=2, label='Ensemble Average')

    if self.diffusion_widget.show_model_fits.isChecked():
        for model_name, fit_data in results['model_fits'].items():
            ax.plot(fit_data['lag_times'], fit_data['fit_curve'],
                   '--', linewidth=1.5, label=f'{model_name} Fit')

    ax.set_xlabel('Time Lag (s)')
    ax.set_ylabel('MSD (μm²)')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.legend()
    ax.grid(True)

def plot_diffusion_map(self, ax, results):
    """Plot spatial map of diffusion coefficients."""
    if self.image_stack is not None:
        ax.imshow(np.max(self.image_stack, axis=0), cmap='gray')

    scatter = ax.scatter(results['positions'][:, 0],
                        results['positions'][:, 1],
                        c=results['D_values'],
                        cmap='viridis',
                        norm=mcolors.LogNorm(),
                        alpha=0.6)

    plt.colorbar(scatter, ax=ax, label='D (μm²/s)')
    ax.set_title('Diffusion Coefficient Map')

def plot_alpha_distribution(self, ax, results):
    """Plot distribution of anomalous exponents."""
    alpha_values = [track_data['alpha'] for track_data in results['track_results'].values()]
    ax.hist(alpha_values, bins=30, density=True)
    ax.axvline(x=1.0, color='r', linestyle='--', label='Normal Diffusion')
    ax.set_xlabel('Anomalous Exponent (α)')
    ax.set_ylabel('Density')
    ax.legend()
    ax.grid(True)

def plot_model_comparison(self, ax, results):
    """Plot comparison of different diffusion models."""
    models = list(results['model_fits'].keys())
    r_squared = [results['model_fits'][model]['r_squared'] for model in models]

    ax.bar(models, r_squared)
    ax.set_xlabel('Model')
    ax.set_ylabel('R²')
    ax.set_title('Model Comparison')
    plt.xticks(rotation=45)

    @staticmethod
    def plot_diffusion_map(tracks_df: pd.DataFrame,
                          analysis_results: Dict,
                          image: Optional[np.ndarray] = None,
                          colormap: str = 'viridis',
                          show_trajectories: bool = False) -> Figure:
        """Create enhanced spatial map of diffusion coefficients"""
        fig, ax = plt.subplots(figsize=(8, 8))

        # Plot background image if provided
        if image is not None:
            ax.imshow(image, cmap='gray')

        results_df = analysis_results['results_df']

        # Create scatter plot of positions colored by D
        scatter = None
        for track_id, D in zip(results_df['track_id'], results_df['D']):
            track = tracks_df[tracks_df['track_id'] == track_id]

            if show_trajectories:
                ax.plot(track['x'], track['y'], alpha=0.3, color='gray')

            scatter = ax.scatter(track['x'], track['y'],
                               c=[D], cmap=colormap,
                               norm=plt.LogNorm())

        if scatter is not None:
            plt.colorbar(scatter, label='D (μm²/s)')

        ax.set_xlabel('X (pixels)')
        ax.set_ylabel('Y (pixels)')

        return fig

    @staticmethod
    def plot_analysis_summary(analysis_results: Dict) -> Figure:
        """Create summary plot with multiple analysis metrics"""
        fig = plt.figure(figsize=(12, 8))
        gs = fig.add_gridspec(2, 2)

        # D distribution
        ax1 = fig.add_subplot(gs[0, 0])
        sns.histplot(data=analysis_results['results_df'], x='D', ax=ax1)
        ax1.set_xlabel('Diffusion Coefficient (μm²/s)')
        ax1.set_xscale('log')

        # Alpha distribution
        ax2 = fig.add_subplot(gs[0, 1])
        sns.histplot(data=analysis_results['results_df'], x='alpha', ax=ax2)
        ax2.set_xlabel('Anomalous Exponent (α)')

        # Track length distribution
        ax3 = fig.add_subplot(gs[1, 0])
        sns.histplot(data=analysis_results['results_df'], x='track_length', ax=ax3)
        ax3.set_xlabel('Track Length (frames)')

        # R² distribution
        ax4 = fig.add_subplot(gs[1, 1])
        sns.histplot(data=analysis_results['results_df'], x='r_squared', ax=ax4)
        ax4.set_xlabel('R² of Fit')

        plt.tight_layout()
        return fig
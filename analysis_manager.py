import numpy as np
import pandas as pd
from typing import Dict, List, Optional

from .detector import ParticleDetector
from .tracker import ParticleTracker
from .linker import get_linker
from .diffusion import DiffusionAnalyzer
from .active_transport import ActiveTransportAnalyzer
from .boundary_crossing import BoundaryCrossingAnalyzer
from .dwell_time import DwellTimeAnalyzer
from .crowding import CrowdingAnalyzer
from .diffusion_population import DiffusionPopulationAnalyzer
from .gel_structure import GelStructureAnalyzer
from .microcompartment import MicrocompartmentAnalyzer
from .diffusion_models import DiffusionModelsAnalyzer
from .multi_channel import MultiChannelAnalyzer
class AnalysisWorker(QObject):
    """Worker class for running analysis in background."""

    finished = pyqtSignal()
    error = pyqtSignal(str)
    result = pyqtSignal(object)
    progress = pyqtSignal(int)

    def __init__(self, analysis_manager, analysis_type, data, params=None):
        super().__init__()
        self.analysis_manager = analysis_manager
        self.analysis_type = analysis_type
        self.data = data
        self.params = params

    def run(self):
        """Run the analysis."""
        try:
            # Report initial progress
            self.progress.emit(0)

            # Run analysis
            results = self.analysis_manager.run_analysis(
                self.analysis_type,
                self.data,
                self.params
            )

            # Report completion
            self.progress.emit(100)
            self.result.emit(results)

        except Exception as e:
            self.error.emit(str(e))

        finally:
            self.finished.emit()
class AnalysisManager:
    """Manages different types of analysis and their results."""

    def __init__(self):
        self.analysis_modules = {
            'active_transport': ActiveTransportAnalyzer(),
            'diffusion': DiffusionAnalyzer(),
            'boundary_crossing': BoundaryCrossingAnalyzer()
        }

        # Analysis results storage
        self.results = {}

        # Signal handling
        self.analysis_completed = pyqtSignal(str, object)  # analysis_type, results
        self.error_occurred = pyqtSignal(str)  # error message

    def run_analysis(self, analysis_type, data, params=None):
        """Run specified analysis type."""
        try:
            if analysis_type not in self.analysis_modules:
                raise ValueError(f"Unknown analysis type: {analysis_type}")

            analyzer = self.analysis_modules[analysis_type]

            # Set parameters if provided
            if params is not None:
                analyzer.set_parameters(params)

            # Run analysis
            results = analyzer.analyze(data)

            # Store results
            self.results[analysis_type] = results

            # Emit completion signal
            self.analysis_completed.emit(analysis_type, results)

            return results

        except Exception as e:
            error_msg = f"Analysis failed: {str(e)}"
            self.error_occurred.emit(error_msg)
            raise

    def get_results(self, analysis_type):
        """Get results for specified analysis type."""
        return self.results.get(analysis_type)

    def clear_results(self, analysis_type=None):
        """Clear analysis results."""
        if analysis_type is None:
            self.results.clear()
        elif analysis_type in self.results:
            del self.results[analysis_type]

    def export_results(self, analysis_type, file_path, export_format='csv'):
        """Export analysis results."""
        try:
            if analysis_type not in self.results:
                raise ValueError(f"No results available for {analysis_type}")

            results = self.results[analysis_type]

            # Use AnalysisExporter to handle export
            exporter = AnalysisExporter()
            success, message = exporter.export_results(
                analysis_type,
                results,
                file_path
            )

            return success, message

        except Exception as e:
            return False, f"Export failed: {str(e)}"
        
        # Core components
        self.detector = None
        self.tracker = None
        self.current_analysis = None

        # Initialize analyzers
        self.analyzers = {
            'active_transport': ActiveTransportAnalyzer(),
            'boundary_crossing': BoundaryCrossingAnalyzer(),
            'diffusion': DiffusionAnalyzer(),
            'diffusion_models': DiffusionModelsAnalyzer(),
            'diffusion_population': DiffusionPopulationAnalyzer(),
            'dwell_time': DwellTimeAnalyzer(),
            'gel_structure': GelStructureAnalyzer(),
            'microcompartment': MicrocompartmentAnalyzer(),
            'multi_channel': MultiChannelAnalyzer(),
            'crowding': CrowdingAnalyzer()
        }

    def setup_detector(self, params: Dict) -> None:
        """Initialize particle detector with parameters"""
        self.detector = ParticleDetector(**params)

    def detect_particles(self, frame: np.ndarray) -> np.ndarray:
        """Detect particles in a single frame"""
        if self.detector is None:
            raise ValueError("Detector not initialized")
        return self.detector.detect(frame)

    def setup_tracker(self, params: Dict) -> None:
        """Initialize particle tracker with parameters"""
        linker = get_linker(params.pop('linking_method', 'hungarian'))
        self.tracker = ParticleTracker(linker=linker, **params)

    def track_particles(self, detections: List[np.ndarray]) -> pd.DataFrame:
        """Link particles into tracks"""
        if self.tracker is None:
            raise ValueError("Tracker not initialized")
        return self.tracker.track(detections)

    def run_analysis(self, analysis_type: str, tracks: pd.DataFrame,
                    params: Optional[Dict] = None) -> Dict:
        """Run selected analysis on tracks"""
        if params is None:
            params = {}

        if analysis_type not in self.analyzers:
            raise ValueError(f"Unknown analysis type: {analysis_type}")

        analyzer = self.analyzers[analysis_type]
        self.current_analysis = analyzer
        return analyzer.analyze(tracks, params)

    # Convenience methods for specific analyses
    def run_active_transport_analysis(self, tracks: pd.DataFrame, params: Dict) -> Dict:
        return self.run_analysis('active_transport', tracks, params)

    def run_boundary_crossing_analysis(self, tracks: pd.DataFrame, params: Dict) -> Dict:
        return self.run_analysis('boundary_crossing', tracks, params)

    def run_diffusion_analysis(self, tracks: pd.DataFrame, params: Dict) -> Dict:
        return self.run_analysis('diffusion', tracks, params)

    def run_diffusion_models_analysis(self, tracks: pd.DataFrame, params: Dict) -> Dict:
        return self.run_analysis('diffusion_models', tracks, params)

    def run_diffusion_population_analysis(self, tracks: pd.DataFrame, params: Dict) -> Dict:
        return self.run_analysis('diffusion_population', tracks, params)

    def run_dwell_time_analysis(self, tracks: pd.DataFrame, params: Dict) -> Dict:
        return self.run_analysis('dwell_time', tracks, params)

    def run_gel_structure_analysis(self, tracks: pd.DataFrame, params: Dict) -> Dict:
        return self.run_analysis('gel_structure', tracks, params)

    def run_microcompartment_analysis(self, tracks: pd.DataFrame, params: Dict) -> Dict:
        return self.run_analysis('microcompartment', tracks, params)

    def run_multi_channel_analysis(self, tracks: pd.DataFrame, params: Dict) -> Dict:
        return self.run_analysis('multi_channel', tracks, params)

    def run_crowding_analysis(self, tracks: pd.DataFrame, params: Dict) -> Dict:
        return self.run_analysis('crowding', tracks, params)
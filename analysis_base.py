# analysis_base.py

from abc import ABC, abstractmethod
import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Dict, Any, Optional, List, Tuple

@dataclass
class AnalysisParameters:
    """Base class for analysis parameters."""
    pass

@dataclass
class AnalysisResults:
    """Base class for analysis results."""
    analysis_type: str
    timestamp: str
    parameters: AnalysisParameters
    summary: Dict[str, Any]
    data: Dict[str, pd.DataFrame]
    plots: Dict[str, Any]

class BaseAnalyzer(ABC):
    """Base class for all analysis modules."""

    def __init__(self):
        self.parameters = None
        self.results = None
        self._validate_requirements()

    @abstractmethod
    def _validate_requirements(self):
        """Validate that all required dependencies are available."""
        pass

    @abstractmethod
    def set_parameters(self, parameters: Dict[str, Any]):
        """Set analysis parameters."""
        pass

    @abstractmethod
    def validate_data(self, data: pd.DataFrame) -> bool:
        """Validate input data format."""
        pass

    @abstractmethod
    def analyze(self, data: pd.DataFrame) -> AnalysisResults:
        """Perform the analysis."""
        pass

    @abstractmethod
    def get_default_parameters(self) -> Dict[str, Any]:
        """Get default parameters for the analysis."""
        pass

    @abstractmethod
    def get_parameter_descriptions(self) -> Dict[str, str]:
        """Get descriptions of all parameters."""
        pass

    def get_results_summary(self) -> Dict[str, Any]:
        """Get summary of analysis results."""
        if self.results is None:
            raise ValueError("No analysis results available")
        return self.results.summary
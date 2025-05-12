import os
import sys
import logging
import traceback

# Import PyQt5 modules
from PyQt5.QtWidgets import (
    QApplication,
    QMainWindow,
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QPushButton,
    QLabel,
    QFileDialog,
    QMessageBox,
    QAction,
    QMenu,
    QMenuBar,
    QStatusBar,
    QTextEdit,
    QComboBox,
    QGroupBox,
    QFormLayout,
    QSpinBox,
    QDoubleSpinBox,
    QCheckBox,
    QRadioButton,
    QButtonGroup,
    QTabWidget,
    QScrollArea,
    QSplitter,
    QTableView,
    QHeaderView,
    QAbstractItemView,
    QDialogButtonBox,
    QDialog
)
from PyQt5.QtGui import (
    QIcon,
    QPixmap,
    QFont,
    QKeySequence
)
from PyQt5.QtCore import (
    Qt,
    QThread,
    pyqtSignal,
    QSettings,
    QSize,
    QTimer
)

# Import Matplotlib and other necessary libraries
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import numpy as np
import pandas as pd

# Import custom modules and classes
from spt.core import SPTProject, SPTAnalysis, DataManager, PlotManager, ConfigurationManager
from spt.gui.widgets import (
    ProjectExplorer,
    AnalysisSettingsWidget,
    DataTableWidget,
    LogViewerWidget,
    PlotViewerWidget
)
from spt.gui.dialogs import (
    NewProjectDialog,
    PreferencesDialog,
    AboutDialog
)

# Constants
APP_NAME = "SPT Analyzer"
APP_VERSION = "1.0"
ORGANIZATION_NAME = "MyCompany"
ORGANIZATION_DOMAIN = "mycompany.com"

# Main application window
class SPTAnalyzerGUI(QMainWindow):
    def __init__(self):
        super().__init__()

        # Initialize attributes
        self.project = None
        self.data_manager = DataManager()
        self.plot_manager = PlotManager()
        self.config_manager = ConfigurationManager()

        # Load settings
        self.settings = QSettings(ORGANIZATION_NAME, APP_NAME)

        # Initialize UI
        self.init_ui()

    def init_ui(self):
        """Initialize the main UI components."""
        self.setWindowTitle(f"{APP_NAME} - {APP_VERSION}")
        self.setGeometry(100, 100, 1200, 800) # x, y, width, height

        # Create main widget and layout
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        main_layout = QVBoxLayout(main_widget)

        # Create menu bar
        self.create_menu_bar()

        # Create main layout (splitter)
        main_splitter = QSplitter(Qt.Horizontal)

        # Left panel (Project Explorer and Analysis Settings)
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        self.project_explorer = ProjectExplorer(self)
        self.analysis_settings = AnalysisSettingsWidget(self)
        left_layout.addWidget(self.project_explorer)
        left_layout.addWidget(self.analysis_settings)
        main_splitter.addWidget(left_panel)

        # Right panel (Data Table and Plot Viewer)
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        self.data_table = DataTableWidget(self)
        self.plot_viewer = PlotViewerWidget(self)
        right_layout.addWidget(self.data_table)
        right_layout.addWidget(self.plot_viewer)
        main_splitter.addWidget(right_panel)

        # Add splitter to main layout
        main_layout.addWidget(main_splitter)

        # Create status bar
        self.statusBar().showMessage("Ready")

    def create_menu_bar(self):
        """Create the main menu bar."""
        menu_bar = self.menuBar()

        # File menu
        file_menu = menu_bar.addMenu("&File")

        new_project_action = QAction("&New Project...", self)
        new_project_action.setShortcut(QKeySequence.New)
        new_project_action.triggered.connect(self.new_project)
        file_menu.addAction(new_project_action)

        open_project_action = QAction("&Open Project...", self)
        open_project_action.setShortcut(QKeySequence.Open)
        open_project_action.triggered.connect(self.open_project)
        file_menu.addAction(open_project_action)

        save_project_action = QAction("&Save Project", self)
        save_project_action.setShortcut(QKeySequence.Save)
        save_project_action.triggered.connect(self.save_project)
        file_menu.addAction(save_project_action)

        save_project_as_action = QAction("Save Project &As...", self)
        save_project_as_action.setShortcut(QKeySequence.SaveAs)
        save_project_as_action.triggered.connect(self.save_project_as)
        file_menu.addAction(save_project_as_action)

        file_menu.addSeparator()

        exit_action = QAction("&Exit", self)
        exit_action.setShortcut("Ctrl+Q")
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)

        # Edit menu (placeholder)
        edit_menu = menu_bar.addMenu("&Edit")
        # Add edit actions here (e.g., undo, redo, copy, paste)

        # View menu (placeholder)
        view_menu = menu_bar.addMenu("&View")
        # Add view actions here (e.g., zoom, full screen)

        # Tools menu (placeholder)
        tools_menu = menu_bar.addMenu("&Tools")
        # Add tool actions here (e.g., options, plugins)

        # Help menu
        help_menu = menu_bar.addMenu("&Help")
        about_action = QAction("&About", self)
        about_action.triggered.connect(self.show_about_dialog)
        help_menu.addAction(about_action)

    def new_project(self):
        """Create a new project."""
        dialog = NewProjectDialog(self)
        if dialog.exec_():
            project_name, project_path = dialog.get_project_details()
            if project_name and project_path:
                self.project = SPTProject(project_name, project_path)
                self.update_ui_for_project()
                self.statusBar().showMessage(f"Project 	'{project_name}	' created.")

    def open_project(self):
        """Open an existing project."""
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        file_name, _ = QFileDialog.getOpenFileName(self, "Open Project", "", "SPT Project Files (*.sptproj);;All Files (*)", options=options)
        if file_name:
            self.project = SPTProject.load(file_name)
            self.update_ui_for_project()
            self.statusBar().showMessage(f"Project 	'{self.project.name}	' loaded.")

    def save_project(self):
        """Save the current project."""
        if self.project:
            if self.project.file_path:
                self.project.save(self.project.file_path)
                self.statusBar().showMessage(f"Project 	'{self.project.name}	' saved.")
            else:
                self.save_project_as()
        else:
            QMessageBox.warning(self, "Save Error", "No active project to save.")

    def save_project_as(self):
        """Save the current project to a new file."""
        if self.project:
            options = QFileDialog.Options()
            options |= QFileDialog.DontUseNativeDialog
            file_name, _ = QFileDialog.getSaveFileName(self, "Save Project As...", "", "SPT Project Files (*.sptproj);;All Files (*)", options=options)
            if file_name:
                self.project.save(file_name)
                self.statusBar().showMessage(f"Project saved to 	'{file_name}	'.")
        else:
            QMessageBox.warning(self, "Save Error", "No active project to save.")

    def show_about_dialog(self):
        """Show the about dialog."""
        QMessageBox.about(self, "About SPT Analyzer",
                          f"<p><b>{APP_NAME}</b></p>" +
                          f"<p>Version: {APP_VERSION}</p>" +
                          "<p>SPT Analyzer is a tool for analyzing single-particle tracking data.</p>" +
                          "<p>Copyright (c) 2023, Manus</p>")

    def update_ui_for_project(self):
        """Update the UI when a project is loaded or created."""
        if self.project:
            self.setWindowTitle(f"{APP_NAME} - {self.project.name}")
            # Update other UI elements as needed (e.g., enable/disable actions)
        else:
            self.setWindowTitle(APP_NAME)
            # Disable project-specific actions

    def closeEvent(self, event):
        """Handle the main window close event."""
        # Add any necessary cleanup or save prompts here
        reply = QMessageBox.question(self, 'Message', "Are you sure you want to quit?", QMessageBox.Yes | QMessageBox.No, QMessageBox.No)

        if reply == QMessageBox.Yes:
            event.accept()
        else:
            event.ignore()

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = SPTAnalyzerGUI()
    window.show()
    sys.exit(app.exec_())


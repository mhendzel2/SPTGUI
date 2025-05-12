
"""
Project management module for SPT Analysis.

This module provides classes and functions for managing SPT analysis projects,
organizing data from multiple cells and treatment groups, and enabling comparative
analysis across populations of cells.
"""

import os
import json
import pickle
import datetime
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Union, Any, Tuple
import logging
import uuid
import copy
from collections import defaultdict
import concurrent.futures
from concurrent.futures import ProcessPoolExecutor
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.cluster import KMeans, DBSCAN
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

logger = logging.getLogger(__name__)

class SPTProject:
    """
    Master project class for organizing and analyzing SPT data from multiple cells.
    
    This class provides methods for managing a hierarchical data structure
    (Project > Treatment > Cell > Tracks) and performing comparative analyses
    across cells and treatment groups.
    
    Parameters
    ----------
    name : str
        Name of the project
    description : str, optional
        Brief description of the project, by default None
    """
    
    def __init__(self, name, description=None):
        self.name = name
        self.description = description or ""
        self.created_at = datetime.datetime.now()
        self.last_modified = self.created_at
        self.project_id = str(uuid.uuid4())
        
        # Data structures
        self.treatments = {}  # {treatment_name: TreatmentGroup}
        self.metadata = {}
        self.version = "1.0"
        self.settings = {}
        
        # Analysis results
        self.comparative_analyses = {}
        self.population_analyses = {}
    
    def add_treatment_group(self, name, description=None, metadata=None):
        """
        Add a new treatment group to the project.
        
        Parameters
        ----------
        name : str
            Name of the treatment group
        description : str, optional
            Brief description of the treatment group, by default None
        metadata : dict, optional
            Additional metadata for the treatment group, by default None
            
        Returns
        -------
        TreatmentGroup
            The newly created treatment group
        """
        if name in self.treatments:
            logger.warning(f"Treatment group '{name}' already exists and will be returned.")
            return self.treatments[name]
        
        treatment = TreatmentGroup(name, description, metadata)
        self.treatments[name] = treatment
        
        self.last_modified = datetime.datetime.now()
        return treatment
    
    def get_treatment_group(self, name):
        """
        Get a treatment group by name.
        
        Parameters
        ----------
        name : str
            Name of the treatment group
            
        Returns
        -------
        TreatmentGroup
            The requested treatment group
            
        Raises
        ------
        KeyError
            If the treatment group does not exist
        """
        if name not in self.treatments:
            raise KeyError(f"Treatment group '{name}' does not exist.")
        
        return self.treatments[name]
    
    def list_treatment_groups(self):
        """
        List all treatment groups in the project.
        
        Returns
        -------
        list
            List of treatment group names
        """
        return list(self.treatments.keys())
    
    def add_cell_to_treatment(self, treatment_name, cell_id, data_file, metadata=None):
        """
        Add a cell to a treatment group.
        
        Parameters
        ----------
        treatment_name : str
            Name of the treatment group
        cell_id : str
            Identifier for the cell
        data_file : str
            Path to the cell's data file
        metadata : dict, optional
            Additional metadata for the cell, by default None
            
        Returns
        -------
        Cell
            The newly added cell
        """
        try:
            treatment = self.get_treatment_group(treatment_name)
            cell = treatment.add_cell(cell_id, data_file, metadata)
            self.last_modified = datetime.datetime.now()
            return cell
        except KeyError:
            # Create treatment group if it doesn't exist
            treatment = self.add_treatment_group(treatment_name)
            cell = treatment.add_cell(cell_id, data_file, metadata)
            self.last_modified = datetime.datetime.now()
            return cell
    
    def save_project(self, directory, filename=None):
        """
        Save the project to disk.
        
        Parameters
        ----------
        directory : str
            Directory to save the project
        filename : str, optional
            Filename for the project file, by default None (uses project name)
            
        Returns
        -------
        str
            Path to the saved project file
        """
        os.makedirs(directory, exist_ok=True)
        
        if filename is None:
            filename = f"{self.name.replace(' ', '_')}_project.spt"
        
        filepath = os.path.join(directory, filename)
        
        # Update metadata
        self.last_modified = datetime.datetime.now()
        
        try:
            with open(filepath, 'wb') as f:
                pickle.dump(self, f)
            
            logger.info(f"Project saved to {filepath}")
            return filepath
        except Exception as e:
            logger.error(f"Error saving project: {str(e)}")
            raise
    
    @classmethod
    def load_project(cls, filepath):
        """
        Load a project from disk.
        
        Parameters
        ----------
        filepath : str
            Path to the project file
            
        Returns
        -------
        SPTProject
            The loaded project
        """
        try:
            with open(filepath, 'rb') as f:
                project = pickle.load(f)
            
            logger.info(f"Project loaded from {filepath}")
            return project
        except Exception as e:
            logger.error(f"Error loading project: {str(e)}")
            raise
    
    def compare_treatments(self, parameter_name, test_type='ttest', alpha=0.05):
        """
        Compare a specific parameter across treatment groups.
        
        Parameters
        ----------
        parameter_name : str
            Name of the parameter to compare (e.g., 'diffusion_coefficient')
        test_type : str, optional
            Statistical test to use, by default 'ttest'
        alpha : float, optional
            Significance level, by default 0.05
            
        Returns
        -------
        dict
            Results of statistical comparison
        """
        try:
            # Collect parameter values for each treatment
            treatment_values = {}
            
            for treatment_name, treatment in self.treatments.items():
                values = treatment.get_pooled_parameter(parameter_name)
                if values:
                    treatment_values[treatment_name] = values
            
            if len(treatment_values) < 2:
                return {
                    'status': 'Insufficient treatments for comparison',
                    'n_treatments': len(treatment_values)
                }
            
            # Perform statistical tests
            comparisons = {}
            treatment_names = list(treatment_values.keys())
            
            for i in range(len(treatment_names)):
                for j in range(i+1, len(treatment_names)):
                    name_i = treatment_names[i]
                    name_j = treatment_names[j]
                    
                    values_i = treatment_values[name_i]
                    values_j = treatment_values[name_j]
                    
                    # Skip if not enough data
                    if len(values_i) < 3 or len(values_j) < 3:
                        comparisons[f"{name_i} vs {name_j}"] = {
                            'status': 'Insufficient data',
                            'n_i': len(values_i),
                            'n_j': len(values_j)
                        }
                        continue
                    
                    # Perform statistical test
                    if test_type == 'ttest':
                        stat, pval = stats.ttest_ind(values_i, values_j, equal_var=False)
                    elif test_type == 'mannwhitney':
                        stat, pval = stats.mannwhitneyu(values_i, values_j)
                    elif test_type == 'kstest':
                        stat, pval = stats.ks_2samp(values_i, values_j)
                    else:
                        raise ValueError(f"Unknown test type: {test_type}")
                    
                    # Calculate effect size (Cohen's d)
                    mean_i = np.mean(values_i)
                    mean_j = np.mean(values_j)
                    std_i = np.std(values_i)
                    std_j = np.std(values_j)
                    
                    pooled_std = np.sqrt(((len(values_i) - 1) * std_i**2 + 
                                        (len(values_j) - 1) * std_j**2) / 
                                       (len(values_i) + len(values_j) - 2))
                    
                    effect_size = abs(mean_i - mean_j) / pooled_std if pooled_std > 0 else 0
                    
                    # Store comparison results
                    comparisons[f"{name_i} vs {name_j}"] = {
                        'test_type': test_type,
                        'statistic': stat,
                        'p_value': pval,
                        'significant': pval < alpha,
                        'effect_size': effect_size,
                        'mean_i': mean_i,
                        'mean_j': mean_j,
                        'std_i': std_i,
                        'std_j': std_j,
                        'n_i': len(values_i),
                        'n_j': len(values_j)
                    }
            
            # Store in project
            if parameter_name not in self.comparative_analyses:
                self.comparative_analyses[parameter_name] = {}
            
            self.comparative_analyses[parameter_name][test_type] = {
                'timestamp': datetime.datetime.now(),
                'alpha': alpha,
                'comparisons': comparisons
            }
            
            return self.comparative_analyses[parameter_name][test_type]
        
        except Exception as e:
            logger.error(f"Error in treatment comparison: {str(e)}")
            raise
    
    def detect_subpopulations(self, parameters, method='kmeans', treatment_name=None, min_samples=5):
        """
        Detect potential subpopulations across all cells or within a treatment group.
        
        Parameters
        ----------
        parameters : list
            List of parameters to use for clustering (e.g., ['diffusion_coefficient', 'alpha'])
        method : str, optional
            Clustering method to use, by default 'kmeans'
        treatment_name : str, optional
            Name of treatment group to analyze, by default None (all treatments)
        min_samples : int, optional
            Minimum number of samples required, by default 5
            
        Returns
        -------
        dict
            Results of subpopulation detection
        """
        try:
            # Collect cells
            if treatment_name:
                if treatment_name not in self.treatments:
                    return {
                        'status': f"Treatment '{treatment_name}' not found"
                    }
                
                treatments = {treatment_name: self.treatments[treatment_name]}
            else:
                treatments = self.treatments
            
            # Extract parameter values for each cell
            cell_parameters = []
            cell_ids = []
            cell_treatments = []
            
            for treatment_name, treatment in treatments.items():
                for cell_id, cell in treatment.cells.items():
                    # Check if cell has all required parameters
                    parameters_present = True
                    param_values = []
                    
                    for param in parameters:
                        if not hasattr(cell, 'analysis_results') or not cell.analysis_results:
                            parameters_present = False
                            break
                            
                        # Extract parameter value (mean across all tracks)
                        values = []
                        for track_result in cell.analysis_results:
                            if param in track_result:
                                values.append(track_result[param])
                        
                        if values:
                            param_values.append(np.mean(values))
                        else:
                            parameters_present = False
                            break
                    
                    if parameters_present:
                        cell_parameters.append(param_values)
                        cell_ids.append(cell_id)
                        cell_treatments.append(treatment_name)
            
            # Normalize parameter values and convert to array
            cell_parameters_array = np.array(cell_parameters)
            
            if cell_parameters_array.shape[0] < min_samples:
                return {
                    'status': f"Insufficient data: {cell_parameters_array.shape[0]} cells, minimum {min_samples} required"
                }
            
            # Normalize data
            param_mean = np.mean(cell_parameters_array, axis=0)
            param_std = np.std(cell_parameters_array, axis=0)
            param_std[param_std == 0] = 1  # Avoid division by zero
            normalized_params = (cell_parameters_array - param_mean) / param_std
            
            # Perform dimensionality reduction for visualization
            if cell_parameters_array.shape[1] > 2:
                pca = PCA(n_components=2)
                reduced_params = pca.fit_transform(normalized_params)
                explained_variance = pca.explained_variance_ratio_
            else:
                reduced_params = normalized_params
                explained_variance = [1.0, 1.0] if normalized_params.shape[1] == 2 else [1.0]
            
            # Detect optimal number of clusters
            max_clusters = min(10, cell_parameters_array.shape[0] // 3)
            silhouette_scores = []
            
            for n_clusters in range(2, max_clusters + 1):
                if method == 'kmeans':
                    clusterer = KMeans(n_clusters=n_clusters, random_state=42)
                    cluster_labels = clusterer.fit_predict(normalized_params)
                else:  # dbscan or other methods
                    continue
                
                # Skip if only one cluster was found
                if len(np.unique(cluster_labels)) < 2:
                    silhouette_scores.append(-1)
                    continue
                
                score = silhouette_score(normalized_params, cluster_labels)
                silhouette_scores.append(score)
            
            # Determine optimal number of clusters
            if silhouette_scores:
                optimal_n_clusters = np.argmax(silhouette_scores) + 2  # +2 because we start from 2 clusters
                best_score = silhouette_scores[optimal_n_clusters - 2]
                
                # Only consider multiple clusters if silhouette score is reasonable
                if best_score > 0.3:
                    multiple_populations = True
                else:
                    multiple_populations = False
                    optimal_n_clusters = 1
            else:
                multiple_populations = False
                optimal_n_clusters = 1
            
            # Assign clusters using optimal number
            if multiple_populations:
                if method == 'kmeans':
                    clusterer = KMeans(n_clusters=optimal_n_clusters, random_state=42)
                    cluster_labels = clusterer.fit_predict(normalized_params)
                elif method == 'dbscan':
                    # Automatically determine epsilon
                    from sklearn.neighbors import NearestNeighbors
                    nn = NearestNeighbors(n_neighbors=min(5, len(normalized_params)-1))
                    nn.fit(normalized_params)
                    distances, _ = nn.kneighbors(normalized_params)
                    epsilon = np.sort(distances[:, -1])[int(len(distances) * 0.9)]
                    
                    clusterer = DBSCAN(eps=epsilon, min_samples=min(3, len(normalized_params)//5))
                    cluster_labels = clusterer.fit_predict(normalized_params)
                    
                    # Update optimal_n_clusters
                    unique_labels = np.unique(cluster_labels)
                    optimal_n_clusters = len(unique_labels[unique_labels != -1])
                else:
                    raise ValueError(f"Unknown clustering method: {method}")
            else:
                # Assign all cells to the same cluster
                cluster_labels = np.zeros(len(cell_ids), dtype=int)
            
            # Calculate statistics for each cluster
            clusters = {}
            
            for cluster_idx in range(max(optimal_n_clusters, np.max(cluster_labels) + 1)):
                cluster_mask = cluster_labels == cluster_idx
                
                if np.sum(cluster_mask) == 0:
                    continue
                
                # Extract cluster cells
                cluster_cells = [cell_ids[i] for i in range(len(cell_ids)) if cluster_mask[i]]
                cluster_treatments = [cell_treatments[i] for i in range(len(cell_treatments)) if cluster_mask[i]]
                cluster_params = cell_parameters_array[cluster_mask]
                
                # Calculate parameter statistics
                param_stats = {}
                for i, param in enumerate(parameters):
                    param_values = cluster_params[:, i]
                    param_stats[param] = {
                        'mean': np.mean(param_values),
                        'std': np.std(param_values),
                        'min': np.min(param_values),
                        'max': np.max(param_values),
                        'n': len(param_values)
                    }
                
                # Count treatments in this cluster
                treatment_counts = {}
                for treatment in cluster_treatments:
                    treatment_counts[treatment] = treatment_counts.get(treatment, 0) + 1
                
                # Store cluster information
                clusters[f"Cluster_{cluster_idx}"] = {
                    'size': np.sum(cluster_mask),
                    'cells': cluster_cells,
                    'treatments': cluster_treatments,
                    'treatment_counts': treatment_counts,
                    'parameter_statistics': param_stats,
                    'centroid': np.mean(cluster_params, axis=0)
                }
            
            # Store results in project
            subpopulation_results = {
                'timestamp': datetime.datetime.now(),
                'parameters': parameters,
                'method': method,
                'normalized_parameters': normalized_params,
                'reduced_parameters': reduced_params,
                'explained_variance': explained_variance,
                'multiple_populations_detected': multiple_populations,
                'optimal_n_clusters': optimal_n_clusters,
                'silhouette_scores': silhouette_scores,
                'best_silhouette_score': best_score if multiple_populations else None,
                'cluster_labels': cluster_labels,
                'cell_ids': cell_ids,
                'cell_treatments': cell_treatments,
                'clusters': clusters
            }
            
            # Store in project
            key = 'all' if treatment_name is None else treatment_name
            if key not in self.population_analyses:
                self.population_analyses[key] = {}
            
            param_key = '_'.join(parameters)
            self.population_analyses[key][param_key] = subpopulation_results
            
            return subpopulation_results
        
        except Exception as e:
            logger.error(f"Error in subpopulation detection: {str(e)}")
            raise
    
    def pool_analysis_results(self, parameter_name, treatment_name=None, subpopulation=None):
        """
        Pool analysis results across cells, with optional filtering by treatment and subpopulation.
        
        Parameters
        ----------
        parameter_name : str
            Name of the parameter to pool (e.g., 'diffusion_coefficient')
        treatment_name : str, optional
            Name of treatment group to analyze, by default None (all treatments)
        subpopulation : str, optional
            Name of subpopulation to analyze, by default None (all cells)
            
        Returns
        -------
        dict
            Pooled parameter statistics
        """
        try:
            # Determine which cells to include
            if treatment_name:
                if treatment_name not in self.treatments:
                    return {
                        'status': f"Treatment '{treatment_name}' not found"
                    }
                
                treatments = {treatment_name: self.treatments[treatment_name]}
            else:
                treatments = self.treatments
            
            # Apply subpopulation filter if specified
            cell_filter = None
            if subpopulation:
                # Find the most recent subpopulation analysis
                key = 'all' if treatment_name is None else treatment_name
                if key in self.population_analyses:
                    # Find first analysis with this subpopulation
                    for param_key, analysis in self.population_analyses[key].items():
                        if subpopulation in analysis.get('clusters', {}):
                            # Extract cells in this subpopulation
                            cell_filter = analysis['clusters'][subpopulation]['cells']
                            break
                
                if cell_filter is None:
                    return {
                        'status': f"Subpopulation '{subpopulation}' not found"
                    }
            
            # Collect parameter values
            parameter_values = []
            cell_ids = []
            treatment_names = []
            
            for treatment_name, treatment in treatments.items():
                for cell_id, cell in treatment.cells.items():
                    # Apply subpopulation filter
                    if cell_filter is not None and cell_id not in cell_filter:
                        continue
                        
                    # Extract parameter value from cell's analysis results
                    if hasattr(cell, 'analysis_results') and cell.analysis_results:
                        for track_result in cell.analysis_results:
                            if parameter_name in track_result:
                                parameter_values.append(track_result[parameter_name])
                                cell_ids.append(cell_id)
                                treatment_names.append(treatment_name)
            
            if not parameter_values:
                return {
                    'status': f"No values found for parameter '{parameter_name}'"
                }
            
            # Calculate statistics
            mean_value = np.mean(parameter_values)
            std_value = np.std(parameter_values)
            median_value = np.median(parameter_values)
            min_value = np.min(parameter_values)
            max_value = np.max(parameter_values)
            
            # Test for normality
            if len(parameter_values) >= 8:
                _, normality_p = stats.shapiro(parameter_values)
                is_normal = normality_p > 0.05
            else:
                normality_p = None
                is_normal = None
            
            # Determine confidence interval
            if is_normal:
                # Normal CI
                ci_low, ci_high = stats.norm.interval(0.95, loc=mean_value, scale=std_value/np.sqrt(len(parameter_values)))
            elif len(parameter_values) >= 10:
                # Bootstrap CI
                bootstrap_means = []
                for _ in range(1000):
                    bootstrap_sample = np.random.choice(parameter_values, size=len(parameter_values), replace=True)
                    bootstrap_means.append(np.mean(bootstrap_sample))
                
                ci_low = np.percentile(bootstrap_means, 2.5)
                ci_high = np.percentile(bootstrap_means, 97.5)
            else:
                ci_low = mean_value - 1.96 * std_value/np.sqrt(len(parameter_values))
                ci_high = mean_value + 1.96 * std_value/np.sqrt(len(parameter_values))
            
            # Return pooled results
            return {
                'parameter': parameter_name,
                'n_values': len(parameter_values),
                'n_cells': len(set(cell_ids)),
                'treatments': list(set(treatment_names)),
                'mean': mean_value,
                'std': std_value,
                'median': median_value,
                'min': min_value,
                'max': max_value,
                'ci_95_low': ci_low,
                'ci_95_high': ci_high,
                'normality_p': normality_p,
                'is_normal': is_normal,
                'values': parameter_values
            }
        
        except Exception as e:
            logger.error(f"Error in pooling analysis results: {str(e)}")
            raise
    
    def batch_process_cells(self, analyzer_function, treatment_name=None, max_workers=None):
        """
        Apply an analysis function to multiple cells in parallel.
        
        Parameters
        ----------
        analyzer_function : callable
            Function to apply to each cell
        treatment_name : str, optional
            Name of treatment group to process, by default None (all treatments)
        max_workers : int, optional
            Maximum number of worker processes, by default None (auto)
            
        Returns
        -------
        dict
            Results of batch processing
        """
        try:
            # Collect cells to process
            cells_to_process = []
            cell_info = []
            
            if treatment_name:
                if treatment_name not in self.treatments:
                    return {
                        'status': f"Treatment '{treatment_name}' not found"
                    }
                
                treatments = {treatment_name: self.treatments[treatment_name]}
            else:
                treatments = self.treatments
            
            for treatment_name, treatment in treatments.items():
                for cell_id, cell in treatment.cells.items():
                    cells_to_process.append(cell)
                    cell_info.append((treatment_name, cell_id))
            
            if not cells_to_process:
                return {
                    'status': "No cells to process"
                }
            
            # Process cells in parallel
            max_workers = max_workers or min(os.cpu_count(), len(cells_to_process))
            
            with ProcessPoolExecutor(max_workers=max_workers) as executor:
                # Start the load operations and mark each future with its cell
                future_to_cell = {executor.submit(analyzer_function, cell): (i, cell) 
                                 for i, cell in enumerate(cells_to_process)}
                
                results = []
                errors = []
                
                for future in concurrent.futures.as_completed(future_to_cell):
                    cell_idx, cell = future_to_cell[future]
                    treatment_name, cell_id = cell_info[cell_idx]
                    
                    try:
                        result = future.result()
                        results.append({
                            'treatment': treatment_name,
                            'cell_id': cell_id,
                            'result': result
                        })
                    except Exception as exc:
                        errors.append({
                            'treatment': treatment_name,
                            'cell_id': cell_id,
                            'error': str(exc)
                        })
                        logger.error(f"Cell {cell_id} in treatment {treatment_name} generated an exception: {exc}")
            
            # Return batch processing results
            return {
                'timestamp': datetime.datetime.now(),
                'n_cells': len(cells_to_process),
                'n_successful': len(results),
                'n_failed': len(errors),
                'results': results,
                'errors': errors
            }
        
        except Exception as e:
            logger.error(f"Error in batch processing: {str(e)}")
            raise


class TreatmentGroup:
    """
    Class for managing a group of cells with the same treatment.
    
    This class organizes cells and provides methods for group-level analyses.
    
    Parameters
    ----------
    name : str
        Name of the treatment group
    description : str, optional
        Brief description of the treatment group, by default None
    metadata : dict, optional
        Additional metadata for the treatment group, by default None
    """
    
    def __init__(self, name, description=None, metadata=None):
        self.name = name
        self.description = description or ""
        self.metadata = metadata or {}
        self.created_at = datetime.datetime.now()
        
        self.cells = {}  # {cell_id: Cell}
        
        # Analysis results
        self.pooled_results = {}
    
    def add_cell(self, cell_id, data_file, metadata=None):
        """
        Add a cell to the treatment group.
        
        Parameters
        ----------
        cell_id : str
            Identifier for the cell
        data_file : str
            Path to the cell's data file
        metadata : dict, optional
            Additional metadata for the cell, by default None
            
        Returns
        -------
        Cell
            The newly added cell
        """
        if cell_id in self.cells:
            logger.warning(f"Cell '{cell_id}' already exists in treatment group '{self.name}' and will be returned.")
            return self.cells[cell_id]
        
        cell = Cell(cell_id, data_file, metadata)
        self.cells[cell_id] = cell
        
        return cell
    
    def get_cell(self, cell_id):
        """
        Get a cell by ID.
        
        Parameters
        ----------
        cell_id : str
            Identifier for the cell
            
        Returns
        -------
        Cell
            The requested cell
            
        Raises
        ------
        KeyError
            If the cell does not exist
        """
        if cell_id not in self.cells:
            raise KeyError(f"Cell '{cell_id}' does not exist in treatment group '{self.name}'.")
        
        return self.cells[cell_id]
    
    def list_cells(self):
        """
        List all cells in the treatment group.
        
        Returns
        -------
        list
            List of cell IDs
        """
        return list(self.cells.keys())
    
    def get_pooled_parameter(self, parameter_name):
        """
        Get pooled values of a parameter across all cells.
        
        Parameters
        ----------
        parameter_name : str
            Name of the parameter to pool
            
        Returns
        -------
        list
            List of parameter values
        """
        pooled_values = []
        
        for cell in self.cells.values():
            if hasattr(cell, 'analysis_results') and cell.analysis_results:
                for track_result in cell.analysis_results:
                    if parameter_name in track_result:
                        pooled_values.append(track_result[parameter_name])
        
        return pooled_values
    
    def pool_analysis_results(self, parameter_name):
        """
        Pool analysis results for a specific parameter across all cells.
        
        Parameters
        ----------
        parameter_name : str
            Name of the parameter to pool
Copy
    def pool_analysis_results(self, parameter_name):
        """
        Pool analysis results for a specific parameter across all cells.
        
        Parameters
        ----------
        parameter_name : str
            Name of the parameter to pool
            
        Returns
        -------
        dict
            Pooled parameter statistics
        """
        values = self.get_pooled_parameter(parameter_name)
        
        if not values:
            return {
                'parameter': parameter_name,
                'status': 'No values found',
                'n_values': 0
            }
        
        # Calculate statistics
        mean_value = np.mean(values)
        std_value = np.std(values)
        median_value = np.median(values)
        min_value = np.min(values)
        max_value = np.max(values)
        
        # Store in pooled results
        pooled_result = {
            'parameter': parameter_name,
            'n_values': len(values),
            'mean': mean_value,
            'std': std_value,
            'median': median_value,
            'min': min_value,
            'max': max_value,
            'values': values
        }
        
        self.pooled_results[parameter_name] = pooled_result
        
        return pooled_result


class Cell:
    """
    Class for managing data and analysis results for a single cell.
    
    Parameters
    ----------
    cell_id : str
        Identifier for the cell
    data_file : str
        Path to the cell's data file
    metadata : dict, optional
        Additional metadata for the cell, by default None
    """
    
    def __init__(self, cell_id, data_file, metadata=None):
        self.cell_id = cell_id
        self.data_file = data_file
        self.metadata = metadata or {}
        self.created_at = datetime.datetime.now()
        
        self.tracks_df = None
        self.analysis_results = []
        self.figures = {}
    
    def load_tracks(self, format=None):
        """
        Load tracks from the data file.
        
        Parameters
        ----------
        format : str, optional
            File format, by default None (auto-detect)
            
        Returns
        -------
        pandas.DataFrame
            DataFrame with track data
        """
        try:
            from ..utils.io import load_tracks
            
            self.tracks_df = load_tracks(self.data_file, format)
            
            return self.tracks_df
        
        except Exception as e:
            logger.error(f"Error loading tracks for cell {self.cell_id}: {str(e)}")
            raise
    
    def analyze_diffusion(self, pixel_size=0.1, frame_interval=0.1, min_track_length=10):
        """
        Perform diffusion analysis on tracks.
        
        Parameters
        ----------
        pixel_size : float, optional
            Pixel size in μm, by default 0.1
        frame_interval : float, optional
            Frame interval in seconds, by default 0.1
        min_track_length : int, optional
            Minimum track length to consider, by default 10
            
        Returns
        -------
        dict
            Diffusion analysis results
        """
        try:
            from ..analysis.diffusion import DiffusionAnalyzer
            
            # Load tracks if not already loaded
            if self.tracks_df is None:
                self.load_tracks()
            
            # Create analyzer
            analyzer = DiffusionAnalyzer(pixel_size=pixel_size, frame_interval=frame_interval)
            
            # Compute MSD
            msd_df, ensemble_df = analyzer.compute_msd(
                self.tracks_df, 
                max_lag=20, 
                min_track_length=min_track_length
            )
            
            # Fit diffusion models
            fit_results = analyzer.fit_diffusion_models(ensemble_df)
            
            # Compute diffusion coefficients for each track
            diffusion_df = analyzer.compute_diffusion_coefficient(
                self.tracks_df, 
                method='individual', 
                min_track_length=min_track_length
            )
            
            # Classify tracks
            classification_df, cluster_df = analyzer.classify_tracks(
                self.tracks_df, 
                min_track_length=min_track_length
            )
            
            # Store results for each track
            analysis_results = []
            
            for _, row in diffusion_df.iterrows():
                track_id = row['track_id']
                
                # Get motion type for this track
                motion_type = 'Unknown'
                if classification_df is not None:
                    track_classification = classification_df[classification_df['track_id'] == track_id]
                    if not track_classification.empty:
                        motion_type = track_classification['motion_type'].iloc[0]
                
                track_result = {
                    'track_id': track_id,
                    'diffusion_coefficient': row['D'],
                    'r_squared': row['r_squared'],
                    'track_length': row['track_length'],
                    'motion_type': motion_type
                }
                
                analysis_results.append(track_result)
            
            # Add ensemble results
            ensemble_result = {
                'ensemble_diffusion_coefficient': fit_results['simple_diffusion']['D'],
                'ensemble_r_squared': fit_results['simple_diffusion']['r_squared'],
                'best_model': fit_results['best_model']
            }
            
            if 'anomalous_diffusion' in fit_results:
                ensemble_result['alpha'] = fit_results['anomalous_diffusion']['alpha']
            
            # Create figure
            fig = plt.figure(figsize=(8, 6))
            
            # Plot MSD curve
            x = ensemble_df['time_lag'].values
            y = ensemble_df['msd'].values
            
            plt.plot(x, y, 'o-', label='Ensemble MSD')
            
            # Plot fit curve
            if 'simple_diffusion' in fit_results:
                x_fit = np.linspace(x[0], x[-1], 100)
                y_fit = 4 * fit_results['simple_diffusion']['D'] * x_fit
                plt.plot(x_fit, y_fit, '--', label=f"D = {fit_results['simple_diffusion']['D']:.4f} μm²/s")
            
            plt.xscale('log')
            plt.yscale('log')
            plt.xlabel('Time Lag (s)')
            plt.ylabel('MSD (μm²)')
            plt.title(f"Diffusion Analysis - Cell {self.cell_id}")
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            # Save figure
            self.figures['diffusion_analysis'] = fig
            
            # Close figure
            plt.close(fig)
            
            # Store analysis results
            self.analysis_results = analysis_results
            
            return {
                'cell_id': self.cell_id,
                'ensemble_results': ensemble_result,
                'track_results': analysis_results,
                'n_tracks': len(analysis_results)
            }
        
        except Exception as e:
            logger.error(f"Error in diffusion analysis for cell {self.cell_id}: {str(e)}")
            raise
    
    def analyze_motion(self, pixel_size=0.1, frame_interval=0.1):
        """
        Perform motion analysis on tracks.
        
        Parameters
        ----------
        pixel_size : float, optional
            Pixel size in μm, by default 0.1
        frame_interval : float, optional
            Frame interval in seconds, by default 0.1
            
        Returns
        -------
        dict
            Motion analysis results
        """
        try:
            from ..analysis.motion import MotionAnalyzer
            
            # Load tracks if not already loaded
            if self.tracks_df is None:
                self.load_tracks()
            
            # Create analyzer
            analyzer = MotionAnalyzer(pixel_size=pixel_size, frame_interval=frame_interval)
            
            # Compute velocities
            velocity_df = analyzer.compute_velocities(self.tracks_df)
            
            # Compute turning angles
            turning_df = analyzer.compute_turning_angles(self.tracks_df)
            
            # Compute track shapes
            shape_df = analyzer.analyze_track_shape(self.tracks_df)
            
            # Store results for each track
            analysis_results = []
            
            for _, row in velocity_df.iterrows():
                track_id = row['track_id']
                
                # Get shape metrics for this track
                track_shape = shape_df[shape_df['track_id'] == track_id]
                
                track_result = {
                    'track_id': track_id,
                    'avg_speed': row['avg_speed'],
                    'max_speed': row['max_speed'],
                    'straightness': row['net_displacement'] / row['total_distance'] if row['total_distance'] > 0 else 0,
                    'asphericity': track_shape['asphericity'].iloc[0] if not track_shape.empty else 0,
                    'n_points': row['n_points']
                }
                
                # Store in cell's analysis results if not already there
                if not any(result['track_id'] == track_id for result in self.analysis_results):
                    self.analysis_results.append(track_result)
                else:
                    # Update existing record
                    for result in self.analysis_results:
                        if result['track_id'] == track_id:
                            result.update(track_result)
                            break
            
            # Create figure
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
            
            # Plot speed distribution
            ax1.hist(velocity_df['avg_speed'], bins=20, alpha=0.7)
            ax1.set_xlabel('Average Speed (μm/s)')
            ax1.set_ylabel('Count')
            ax1.set_title('Speed Distribution')
            
            # Plot straightness vs. speed
            ax2.scatter(velocity_df['avg_speed'], velocity_df['net_displacement'] / velocity_df['total_distance'], alpha=0.7)
            ax2.set_xlabel('Average Speed (μm/s)')
            ax2.set_ylabel('Straightness')
            ax2.set_title('Speed vs. Straightness')
            
            plt.tight_layout()
            
            # Save figure
            self.figures['motion_analysis'] = fig
            
            # Close figure
            plt.close(fig)
            
            return {
                'cell_id': self.cell_id,
                'n_tracks': len(velocity_df),
                'mean_speed': velocity_df['avg_speed'].mean(),
                'mean_straightness': (velocity_df['net_displacement'] / velocity_df['total_distance']).mean()
            }
        
        except Exception as e:
            logger.error(f"Error in motion analysis for cell {self.cell_id}: {str(e)}")
            raise
    
    def analyze_clustering(self, pixel_size=0.1):
        """
        Perform spatial clustering analysis on tracks.
        
        Parameters
        ----------
        pixel_size : float, optional
            Pixel size in μm, by default 0.1
            
        Returns
        -------
        dict
            Clustering analysis results
        """
        try:
            from ..analysis.clustering import ClusterAnalyzer
            
            # Load tracks if not already loaded
            if self.tracks_df is None:
                self.load_tracks()
            
            # Create analyzer
            analyzer = ClusterAnalyzer(pixel_size=pixel_size)
            
            # Perform clustering analysis
            cluster_df, clustered_df = analyzer.analyze_clusters(self.tracks_df)
            
            # Compute Ripley's K function
            ripley_k = analyzer.compute_ripley_k(self.tracks_df)
            
            # Create figure
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
            
            # Plot clusters
            from ..visualization.clustering import plot_spatial_clusters
            plot_spatial_clusters(self.tracks_df, clustered_df, cluster_df, ax=ax1, pixel_size=pixel_size)
            
            # Plot Ripley's K function
            from ..visualization.clustering import plot_ripley_k
            plot_ripley_k(ripley_k, ax=ax2)
            
            plt.tight_layout()
            
            # Save figure
            self.figures['clustering_analysis'] = fig
            
            # Close figure
            plt.close(fig)
            
            return {
                'cell_id': self.cell_id,
                'n_clusters': len(cluster_df),
                'clustering_results': cluster_df.to_dict('records')
            }
        
        except Exception as e:
            logger.error(f"Error in clustering analysis for cell {self.cell_id}: {str(e)}")
            raise

            
        Returns
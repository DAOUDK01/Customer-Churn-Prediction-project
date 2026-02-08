"""
Model governance and versioning utilities for Customer Churn ML project.

Handles model versioning, metadata tracking, and governance artifacts
for production ML systems.
"""

import json
import joblib
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional
import pandas as pd
import numpy as np
from dataclasses import dataclass, asdict
import logging

from .config import MODELS_PATH

logger = logging.getLogger(__name__)


@dataclass
class ModelMetadata:
    """Structured metadata for ML models."""
    version: str
    created_date: str
    model_type: str
    training_data_shape: tuple
    features: list
    performance: Dict[str, float]
    hyperparameters: Dict[str, Any]
    business_impact: Dict[str, Any]
    validation_schema: Dict[str, str]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)


class ModelRegistry:
    """Production model registry with versioning and governance."""
    
    def __init__(self, registry_path: Optional[Path] = None):
        """Initialize model registry.
        
        Args:
            registry_path: Path to model registry directory
        """
        self.registry_path = registry_path or MODELS_PATH
        self.registry_path.mkdir(exist_ok=True)
        self.registry_file = self.registry_path / "model_registry.json"
        self._load_registry()
    
    def _load_registry(self) -> None:
        """Load existing model registry or create new one."""
        if self.registry_file.exists():
            with open(self.registry_file, 'r') as f:
                self.registry = json.load(f)
        else:
            self.registry = {
                "models": {},
                "champion": None,
                "created": datetime.now().isoformat()
            }
    
    def _save_registry(self) -> None:
        """Save model registry to disk."""
        with open(self.registry_file, 'w') as f:
            json.dump(self.registry, f, indent=2, default=str)
    
    def register_model(self, 
                      model_pipeline,
                      metadata: ModelMetadata,
                      set_as_champion: bool = True) -> str:
        """Register a new model version.
        
        Args:
            model_pipeline: Trained model pipeline
            metadata: Model metadata
            set_as_champion: Whether to set as current champion
            
        Returns:
            Model version string
        """
        version = metadata.version
        version_dir = self.registry_path / version
        version_dir.mkdir(exist_ok=True)
        
        # Save model artifacts
        model_path = version_dir / "model.joblib"
        metadata_path = version_dir / "metadata.json"
        
        joblib.dump(model_pipeline, model_path)
        with open(metadata_path, 'w') as f:
            json.dump(metadata.to_dict(), f, indent=2, default=str)
        
        # Update registry
        self.registry["models"][version] = {
            "created": metadata.created_date,
            "model_type": metadata.model_type,
            "performance": metadata.performance,
            "business_impact": metadata.business_impact,
            "model_path": str(model_path),
            "metadata_path": str(metadata_path)
        }
        
        if set_as_champion:
            self.set_champion(version)
        
        self._save_registry()
        logger.info(f"Model {version} registered successfully")
        
        return version
    
    def set_champion(self, version: str) -> None:
        """Set a model version as champion.
        
        Args:
            version: Model version to set as champion
        """
        if version not in self.registry["models"]:
            raise ValueError(f"Model version {version} not found")
        
        self.registry["champion"] = version
        self.registry["champion_updated"] = datetime.now().isoformat()
        
        # Copy champion to main models directory for backward compatibility
        champion_path = self.registry_path / version / "model.joblib"
        main_path = self.registry_path / "churn_prediction_pipeline.joblib"
        
        if champion_path.exists():
            joblib.dump(joblib.load(champion_path), main_path)
        
        self._save_registry()
        logger.info(f"Champion model updated to version {version}")
    
    def get_champion(self) -> Dict[str, Any]:
        """Get current champion model information.
        
        Returns:
            Champion model metadata
        """
        champion_version = self.registry.get("champion")
        if not champion_version:
            raise ValueError("No champion model set")
        
        return self.registry["models"][champion_version]
    
    def load_champion(self):
        """Load the current champion model.
        
        Returns:
            Loaded model pipeline
        """
        champion_info = self.get_champion()
        return joblib.load(champion_info["model_path"])
    
    def list_models(self) -> Dict[str, Any]:
        """List all registered models.
        
        Returns:
            Dictionary of all model versions and metadata
        """
        return self.registry["models"]
    
    def compare_models(self, metric: str = "test_roc_auc") -> pd.DataFrame:
        """Compare performance across model versions.
        
        Args:
            metric: Performance metric to compare
            
        Returns:
            DataFrame with model comparison
        """
        models_data = []
        for version, info in self.registry["models"].items():
            models_data.append({
                "version": version,
                "created": info["created"],
                "model_type": info["model_type"],
                metric: info["performance"].get(metric, 0),
                "is_champion": version == self.registry.get("champion")
            })
        
        return pd.DataFrame(models_data).sort_values(metric, ascending=False)


class BusinessImpactCalculator:
    """Calculate business impact metrics for churn models."""
    
    def __init__(self, 
                 avg_customer_value: float = 1200,
                 churn_rate: float = 0.17,
                 intervention_cost: float = 75,
                 intervention_success_rate: float = 0.6):
        """Initialize business parameters.
        
        Args:
            avg_customer_value: Annual revenue per customer
            churn_rate: Baseline churn rate
            intervention_cost: Cost per retention campaign
            intervention_success_rate: Success rate of interventions
        """
        self.avg_customer_value = avg_customer_value
        self.churn_rate = churn_rate
        self.intervention_cost = intervention_cost
        self.intervention_success_rate = intervention_success_rate
    
    def calculate_impact(self, 
                        recall: float, 
                        precision: float, 
                        customer_base: int = 10000) -> Dict[str, Any]:
        """Calculate comprehensive business impact.
        
        Args:
            recall: Model recall score
            precision: Model precision score  
            customer_base: Total customer base size
            
        Returns:
            Dictionary with business impact metrics
        """
        # Calculate baseline scenario
        annual_churners = customer_base * self.churn_rate
        baseline_loss = annual_churners * self.avg_customer_value
        
        # Calculate model scenario
        caught_churners = annual_churners * recall
        saved_customers = caught_churners * self.intervention_success_rate
        revenue_saved = saved_customers * self.avg_customer_value
        
        # Calculate costs (including false positives)
        total_predicted_churners = caught_churners / precision if precision > 0 else 0
        false_positives = total_predicted_churners - caught_churners
        total_interventions = total_predicted_churners
        total_cost = total_interventions * self.intervention_cost
        
        # Net impact
        net_benefit = revenue_saved - total_cost
        roi = net_benefit / total_cost if total_cost > 0 else 0
        
        return {
            "baseline_loss": baseline_loss,
            "revenue_saved": revenue_saved,
            "intervention_cost": total_cost,
            "net_benefit": net_benefit,
            "roi_ratio": roi,
            "customers_saved": int(saved_customers),
            "false_positives": int(false_positives),
            "total_interventions": int(total_interventions),
            "churn_reduction_pct": (saved_customers / annual_churners) * 100
        }


def generate_model_card(metadata: ModelMetadata, 
                       business_impact: Dict[str, Any],
                       output_path: Optional[Path] = None) -> str:
    """Generate model card documentation.
    
    Args:
        metadata: Model metadata
        business_impact: Business impact calculations
        output_path: Path to save model card
        
    Returns:
        Model card content as string
    """
    model_card = f"""# Model Card: Customer Churn Prediction {metadata.version}

## Model Overview
- **Model Type**: {metadata.model_type}
- **Version**: {metadata.version}
- **Created**: {metadata.created_date}
- **Purpose**: Predict customer churn risk for proactive retention

## Intended Use
- **Primary Use**: Identify high-risk customers for targeted retention campaigns
- **Intended Users**: Customer success teams, marketing, business analysts
- **Out-of-Scope**: Real-time decision making, regulatory compliance decisions

## Performance Metrics
- **Accuracy**: {metadata.performance.get('test_accuracy', 0):.3f}
- **Precision**: {metadata.performance.get('test_precision', 0):.3f}
- **Recall**: {metadata.performance.get('test_recall', 0):.3f} (Primary metric for business impact)
- **ROC-AUC**: {metadata.performance.get('test_roc_auc', 0):.3f}

## Business Impact
- **Revenue Protected**: ${business_impact.get('revenue_saved', 0):,.0f} annually
- **ROI**: {business_impact.get('roi_ratio', 0):.1f}:1
- **Customers Saved**: {business_impact.get('customers_saved', 0):,}
- **Churn Reduction**: {business_impact.get('churn_reduction_pct', 0):.1f}%

## Model Architecture
- **Algorithm**: XGBoost Classifier
- **Features**: {len(metadata.features)} features
- **Hyperparameters**: {json.dumps(metadata.hyperparameters, indent=2)}

## Training Data
- **Shape**: {metadata.training_data_shape}
- **Source**: Customer transaction and behavior data
- **Time Period**: Last 24 months
- **Label Definition**: Customer churned within 30 days

## Limitations & Risks
- **Data Drift**: Model performance may degrade with changing customer behavior
- **Bias**: May underperform for customer segments with limited historical data
- **False Positives**: ~{business_impact.get('false_positives', 0)} unnecessary interventions per period
- **Refresh Needed**: Recommend retraining every 3-6 months

## Monitoring Requirements
- Track prediction distribution shifts
- Monitor intervention success rates
- Measure actual vs predicted churn rates
- Alert on performance degradation

## Approval & Governance
- **Model Owner**: ML Engineering Team
- **Business Owner**: Customer Success
- **Review Frequency**: Quarterly
- **Risk Level**: Medium (business impact, no regulatory)

---
*Generated automatically on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*
"""
    
    if output_path:
        output_path.write_text(model_card)
        logger.info(f"Model card saved to {output_path}")
    
    return model_card
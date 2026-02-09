"""
Schema validation utilities for Customer Churn ML project.

Provides data validation, schema checking, and input sanitization
for production ML systems.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional, Union
from dataclasses import dataclass
from enum import Enum
import logging
import json

logger = logging.getLogger(__name__)


class DataType(Enum):
    """Supported data types for validation."""
    NUMERIC = "numeric"
    CATEGORICAL = "categorical"
    BOOLEAN = "boolean"
    DATE = "date"
    STRING = "string"


@dataclass
class FieldSchema:
    """Schema definition for a single field."""
    name: str
    data_type: DataType
    required: bool = True
    min_value: Optional[float] = None
    max_value: Optional[float] = None
    allowed_values: Optional[List[str]] = None
    nullable: bool = False
    description: Optional[str] = None


class DataSchema:
    """Data schema validator for ML inputs."""
    
    def __init__(self, fields: List[FieldSchema]):
        """Initialize schema validator.
        
        Args:
            fields: List of field schema definitions
        """
        self.fields = {field.name: field for field in fields}
        self.field_names = set(field.name for field in fields)
        self.required_fields = set(field.name for field in fields if field.required)
    
    @classmethod
    def from_dict(cls, schema_dict: Dict[str, Any]) -> 'DataSchema':
        """Create schema from dictionary definition.
        
        Args:
            schema_dict: Dictionary with schema definitions
            
        Returns:
            DataSchema instance
        """
        fields = []
        for field_name, field_def in schema_dict.items():
            field = FieldSchema(
                name=field_name,
                data_type=DataType(field_def["data_type"]),
                required=field_def.get("required", True),
                min_value=field_def.get("min_value"),
                max_value=field_def.get("max_value"), 
                allowed_values=field_def.get("allowed_values"),
                nullable=field_def.get("nullable", False),
                description=field_def.get("description")
            )
            fields.append(field)
        
        return cls(fields)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert schema to dictionary.
        
        Returns:
            Schema as dictionary
        """
        schema_dict = {}
        for field_name, field in self.fields.items():
            schema_dict[field_name] = {
                "data_type": field.data_type.value,
                "required": field.required,
                "nullable": field.nullable
            }
            if field.min_value is not None:
                schema_dict[field_name]["min_value"] = field.min_value
            if field.max_value is not None:
                schema_dict[field_name]["max_value"] = field.max_value
            if field.allowed_values is not None:
                schema_dict[field_name]["allowed_values"] = field.allowed_values
            if field.description is not None:
                schema_dict[field_name]["description"] = field.description
        
        return schema_dict
    
    def validate(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Validate data against schema.
        
        Args:
            data: DataFrame to validate
            
        Returns:
            Validation results
        """
        validation_results = {
            "valid": True,
            "errors": [],
            "warnings": [],
            "summary": {
                "total_rows": len(data),
                "total_columns": len(data.columns),
                "missing_fields": [],
                "extra_fields": [],
                "invalid_fields": []
            }
        }
        
        # Check for missing required fields
        missing_fields = self.required_fields - set(data.columns)
        if missing_fields:
            validation_results["valid"] = False
            validation_results["errors"].append(f"Missing required fields: {missing_fields}")
            validation_results["summary"]["missing_fields"] = list(missing_fields)
        
        # Check for extra fields
        extra_fields = set(data.columns) - self.field_names
        if extra_fields:
            validation_results["warnings"].append(f"Extra fields found: {extra_fields}")
            validation_results["summary"]["extra_fields"] = list(extra_fields)
        
        # Validate each field
        for field_name, field_schema in self.fields.items():
            if field_name not in data.columns:
                continue  # Already handled in missing fields check
            
            field_result = self._validate_field(data[field_name], field_schema)
            if not field_result["valid"]:
                validation_results["valid"] = False
                validation_results["errors"].extend(field_result["errors"])
                validation_results["summary"]["invalid_fields"].append(field_name)
            
            if field_result["warnings"]:
                validation_results["warnings"].extend(field_result["warnings"])
        
        return validation_results
    
    def _validate_field(self, series: pd.Series, field_schema: FieldSchema) -> Dict[str, Any]:
        """Validate a single field.
        
        Args:
            series: Data series to validate
            field_schema: Schema for the field
            
        Returns:
            Field validation results
        """
        result = {"valid": True, "errors": [], "warnings": []}
        field_name = field_schema.name
        
        # Check for null values
        null_count = series.isnull().sum()
        if null_count > 0 and not field_schema.nullable:
            result["valid"] = False
            result["errors"].append(f"{field_name}: Contains {null_count} null values")
        
        # Skip further validation for null values
        non_null_series = series.dropna()
        if len(non_null_series) == 0:
            return result
        
        # Data type specific validation
        if field_schema.data_type == DataType.NUMERIC:
            result.update(self._validate_numeric_field(non_null_series, field_schema))
        elif field_schema.data_type == DataType.CATEGORICAL:
            result.update(self._validate_categorical_field(non_null_series, field_schema))
        elif field_schema.data_type == DataType.BOOLEAN:
            result.update(self._validate_boolean_field(non_null_series, field_schema))
        elif field_schema.data_type == DataType.STRING:
            result.update(self._validate_string_field(non_null_series, field_schema))
        
        return result
    
    def _validate_numeric_field(self, series: pd.Series, field_schema: FieldSchema) -> Dict[str, Any]:
        """Validate numeric field."""
        result = {"valid": True, "errors": [], "warnings": []}
        field_name = field_schema.name
        
        # Check if data is numeric
        try:
            numeric_series = pd.to_numeric(series, errors='coerce')
            invalid_count = numeric_series.isnull().sum()
            if invalid_count > 0:
                result["errors"].append(f"{field_name}: {invalid_count} non-numeric values")
                result["valid"] = False
                return result
            
            # Check range constraints
            if field_schema.min_value is not None:
                below_min = (numeric_series < field_schema.min_value).sum()
                if below_min > 0:
                    result["errors"].append(f"{field_name}: {below_min} values below minimum {field_schema.min_value}")
                    result["valid"] = False
            
            if field_schema.max_value is not None:
                above_max = (numeric_series > field_schema.max_value).sum()
                if above_max > 0:
                    result["errors"].append(f"{field_name}: {above_max} values above maximum {field_schema.max_value}")
                    result["valid"] = False
            
            # Check for outliers (warning only)
            q1, q3 = numeric_series.quantile([0.25, 0.75])
            iqr = q3 - q1
            outlier_bounds = (q1 - 3 * iqr, q3 + 3 * iqr)
            outliers = ((numeric_series < outlier_bounds[0]) | (numeric_series > outlier_bounds[1])).sum()
            if outliers > len(series) * 0.05:  # More than 5% outliers
                result["warnings"].append(f"{field_name}: {outliers} potential outliers detected")
                
        except Exception as e:
            result["errors"].append(f"{field_name}: Numeric validation failed - {str(e)}")
            result["valid"] = False
        
        return result
    
    def _validate_categorical_field(self, series: pd.Series, field_schema: FieldSchema) -> Dict[str, Any]:
        """Validate categorical field."""
        result = {"valid": True, "errors": [], "warnings": []}
        field_name = field_schema.name
        
        if field_schema.allowed_values:
            invalid_values = set(series) - set(field_schema.allowed_values)
            if invalid_values:
                result["errors"].append(f"{field_name}: Invalid values found: {invalid_values}")
                result["valid"] = False
        
        # Check for high cardinality (warning)
        unique_count = series.nunique()
        if unique_count > 50:
            result["warnings"].append(f"{field_name}: High cardinality ({unique_count} unique values)")
        
        return result
    
    def _validate_boolean_field(self, series: pd.Series, field_schema: FieldSchema) -> Dict[str, Any]:
        """Validate boolean field."""
        result = {"valid": True, "errors": [], "warnings": []}
        field_name = field_schema.name
        
        # Check if values are boolean-like
        valid_boolean_values = {True, False, 0, 1, '0', '1', 'true', 'false', 'True', 'False', 'yes', 'no', 'Yes', 'No'}
        invalid_values = set(series) - valid_boolean_values
        
        if invalid_values:
            result["errors"].append(f"{field_name}: Non-boolean values found: {invalid_values}")
            result["valid"] = False
        
        return result
    
    def _validate_string_field(self, series: pd.Series, field_schema: FieldSchema) -> Dict[str, Any]:
        """Validate string field."""
        result = {"valid": True, "errors": [], "warnings": []}
        field_name = field_schema.name
        
        # Check for empty strings
        empty_strings = (series == '').sum()
        if empty_strings > 0:
            result["warnings"].append(f"{field_name}: {empty_strings} empty strings found")
        
        # Check string length (basic validation)
        if series.dtype == 'object':
            str_lengths = series.astype(str).str.len()
            if str_lengths.max() > 1000:
                result["warnings"].append(f"{field_name}: Very long strings detected (max length: {str_lengths.max()})")
        
        return result
    
    def sanitize_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """Sanitize data according to schema.
        
        Args:
            data: DataFrame to sanitize
            
        Returns:
            Sanitized DataFrame
        """
        sanitized_data = data.copy()
        
        for field_name, field_schema in self.fields.items():
            if field_name not in sanitized_data.columns:
                continue
            
            # Apply field-specific sanitization
            if field_schema.data_type == DataType.NUMERIC:
                sanitized_data[field_name] = pd.to_numeric(
                    sanitized_data[field_name], errors='coerce'
                )
                
                # Apply range constraints
                if field_schema.min_value is not None:
                    sanitized_data[field_name] = sanitized_data[field_name].clip(lower=field_schema.min_value)
                if field_schema.max_value is not None:
                    sanitized_data[field_name] = sanitized_data[field_name].clip(upper=field_schema.max_value)
            
            elif field_schema.data_type == DataType.CATEGORICAL:
                if field_schema.allowed_values:
                    # Replace invalid values with NaN
                    mask = ~sanitized_data[field_name].isin(field_schema.allowed_values)
                    sanitized_data.loc[mask, field_name] = np.nan
            
            elif field_schema.data_type == DataType.BOOLEAN:
                # Convert to boolean
                sanitized_data[field_name] = self._convert_to_boolean(sanitized_data[field_name])
        
        logger.info("Data sanitization completed")
        return sanitized_data
    
    def _convert_to_boolean(self, series: pd.Series) -> pd.Series:
        """Convert series to boolean values."""
        # Create mapping for common boolean representations
        bool_mapping = {
            'true': True, 'false': False,
            'True': True, 'False': False,
            'yes': True, 'no': False,
            'Yes': True, 'No': False,
            'y': True, 'n': False,
            'Y': True, 'N': False,
            '1': True, '0': False,
            1: True, 0: False
        }
        
        return series.map(bool_mapping).astype('boolean')


"""High-level validators and helpers for churn data."""

# Predefined schema for customer churn data
CHURN_DATA_SCHEMA = DataSchema([
    FieldSchema(name="customerID", data_type=DataType.STRING, required=False),
    FieldSchema(name="gender", data_type=DataType.CATEGORICAL, 
               allowed_values=["Male", "Female"]),
    FieldSchema(name="SeniorCitizen", data_type=DataType.BOOLEAN),
    FieldSchema(name="Partner", data_type=DataType.CATEGORICAL,
               allowed_values=["Yes", "No"]),
    FieldSchema(name="Dependents", data_type=DataType.CATEGORICAL,
               allowed_values=["Yes", "No"]),
    FieldSchema(name="tenure", data_type=DataType.NUMERIC, 
               min_value=0, max_value=100),
    FieldSchema(name="PhoneService", data_type=DataType.CATEGORICAL,
               allowed_values=["Yes", "No"]),
    FieldSchema(name="MultipleLines", data_type=DataType.CATEGORICAL,
               allowed_values=["Yes", "No", "No phone service"]),
    FieldSchema(name="InternetService", data_type=DataType.CATEGORICAL,
               allowed_values=["DSL", "Fiber optic", "No"]),
    FieldSchema(name="OnlineSecurity", data_type=DataType.CATEGORICAL,
               allowed_values=["Yes", "No", "No internet service"]),
    FieldSchema(name="OnlineBackup", data_type=DataType.CATEGORICAL,
               allowed_values=["Yes", "No", "No internet service"]),
    FieldSchema(name="DeviceProtection", data_type=DataType.CATEGORICAL,
               allowed_values=["Yes", "No", "No internet service"]),
    FieldSchema(name="TechSupport", data_type=DataType.CATEGORICAL,
               allowed_values=["Yes", "No", "No internet service"]),
    FieldSchema(name="StreamingTV", data_type=DataType.CATEGORICAL,
               allowed_values=["Yes", "No", "No internet service"]),
    FieldSchema(name="StreamingMovies", data_type=DataType.CATEGORICAL,
               allowed_values=["Yes", "No", "No internet service"]),
    FieldSchema(name="Contract", data_type=DataType.CATEGORICAL,
               allowed_values=["Month-to-month", "One year", "Two year"]),
    FieldSchema(name="PaperlessBilling", data_type=DataType.CATEGORICAL,
               allowed_values=["Yes", "No"]),
    FieldSchema(name="PaymentMethod", data_type=DataType.CATEGORICAL,
               allowed_values=["Electronic check", "Mailed check", 
                             "Bank transfer (automatic)", "Credit card (automatic)"]),
    FieldSchema(name="MonthlyCharges", data_type=DataType.NUMERIC,
               min_value=0, max_value=1000),
    FieldSchema(name="TotalCharges", data_type=DataType.NUMERIC,
               min_value=0, max_value=10000)
])


class DataValidator:
    """Simple wrapper around CHURN_DATA_SCHEMA for legacy code paths.

    Provides a validate_customer_data method used by the API and batch
    pipeline. It normalizes single-record dictionaries into a DataFrame
    and returns a compact {valid, issues} structure expected by callers.
    """

    def __init__(self, schema: DataSchema = CHURN_DATA_SCHEMA):
        self.schema = schema

    def validate_customer_data(self, customer: Dict[str, Any]) -> Dict[str, Any]:
        """Validate a single customer dictionary.

        Args:
            customer: Mapping of field name to value for one customer.

        Returns:
            Dict with keys:
            - valid: bool
            - issues: list or None with human-readable messages
        """
        # Convert single record to DataFrame for schema validator
        df = pd.DataFrame([customer])
        result = self.schema.validate(df)

        issues: List[str] = []
        errors = result.get("errors") or []
        warnings = result.get("warnings") or []
        issues.extend(errors)
        issues.extend(warnings)

        return {
            "valid": bool(result.get("valid", False)),
            "issues": issues or None,
        }


def validate_churn_data(data: pd.DataFrame) -> Dict[str, Any]:
    """Validate customer churn data using predefined schema.
    
    Args:
        data: Customer data DataFrame
        
    Returns:
        Validation results
    """
    return CHURN_DATA_SCHEMA.validate(data)


def sanitize_churn_data(data: pd.DataFrame) -> pd.DataFrame:
    """Sanitize customer churn data using predefined schema.
    
    Args:
        data: Customer data DataFrame
        
    Returns:
        Sanitized DataFrame
    """
    return CHURN_DATA_SCHEMA.sanitize_data(data)
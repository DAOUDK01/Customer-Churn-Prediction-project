#!/usr/bin/env python
"""
Production Deployment Test Suite
Comprehensive testing of all production components
"""

import sys
import time
import json
import requests
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import subprocess
import threading
import signal
import os

# Add project root to path
project_root = Path(__file__).parent
sys.path.append(str(project_root))

class ProductionTestSuite:
    """Complete production system testing"""
    
    def __init__(self):
        self.api_process = None
        self.base_url = "http://localhost:8000"
        self.auth_token = "demo-token-2026"
        self.test_results = []
    
    def run_all_tests(self):
        """Run comprehensive production test suite"""
        print("🎯 PRODUCTION DEPLOYMENT TEST SUITE")
        print("=" * 60)
        
        try:
            # 1. Component Tests
            self._test_component_imports()
            
            # 2. API Server Tests
            self._test_api_deployment()
            
            # 3. Batch Processing Tests
            self._test_batch_pipeline()
            
            # 4. Monitoring System Tests
            self._test_monitoring_system()
            
            # 5. Integration Tests
            self._test_end_to_end_workflow()
            
            # Generate test report
            self._generate_test_report()
            
            return True
            
        except Exception as e:
            print(f"❌ Production test suite failed: {e}")
            return False
        finally:
            self._cleanup()
    
    def _test_component_imports(self):
        """Test all production components can be imported"""
        print("\n1️⃣ COMPONENT IMPORT TESTS")
        
        components = [
            ("API Server", "from src.api import app"),
            ("Batch Pipeline", "from src.batch_pipeline import BatchPredictionPipeline"),
            ("Advanced Monitoring", "from src.advanced_monitoring import AdvancedModelMonitor"),
            ("Prediction Service", "from src.predict import ChurnPredictor"),
            ("Validation", "from src.validation import DataValidator"),
            ("Configuration", "from src.config import API_CONFIG")
        ]
        
        for component_name, import_statement in components:
            try:
                exec(import_statement)
                print(f"   ✅ {component_name} imported successfully")
                self.test_results.append({
                    'test': f'Import {component_name}',
                    'status': 'PASS',
                    'timestamp': datetime.now().isoformat()
                })
            except Exception as e:
                print(f"   ❌ {component_name} import failed: {e}")
                self.test_results.append({
                    'test': f'Import {component_name}',
                    'status': 'FAIL',
                    'error': str(e),
                    'timestamp': datetime.now().isoformat()
                })
    
    def _test_api_deployment(self):
        """Test API server deployment and endpoints"""
        print("\n2️⃣ API DEPLOYMENT TESTS")
        
        # Start API server
        try:
            print("   🚀 Starting API server...")
            self._start_api_server()
            
            # Wait for server to start
            time.sleep(5)
            
            # Test endpoints
            self._test_health_endpoint()
            self._test_prediction_endpoints()
            self._test_metrics_endpoint()
            
        except Exception as e:
            print(f"   ❌ API deployment test failed: {e}")
    
    def _start_api_server(self):
        """Start the API server in background"""
        try:
            api_script = project_root / "src" / "api.py"
            cmd = [sys.executable, str(api_script)]
            
            # Start process
            self.api_process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                cwd=str(project_root)
            )
            
            print("   ✅ API server started")
            
        except Exception as e:
            print(f"   ❌ Failed to start API server: {e}")
    
    def _test_health_endpoint(self):
        """Test API health endpoint"""
        try:
            response = requests.get(f"{self.base_url}/health", timeout=10)
            
            if response.status_code == 200:
                health_data = response.json()
                print(f"   ✅ Health endpoint: {health_data.get('status', 'unknown')}")
                self.test_results.append({
                    'test': 'Health Endpoint',
                    'status': 'PASS',
                    'response_time_ms': response.elapsed.total_seconds() * 1000
                })
            else:
                print(f"   ❌ Health endpoint failed: {response.status_code}")
                
        except Exception as e:
            print(f"   ❌ Health endpoint error: {e}")
    
    def _test_prediction_endpoints(self):
        """Test prediction endpoints"""
        try:
            # Test single prediction
            headers = {"Authorization": f"Bearer {self.auth_token}"}\n            payload = {\n                "tenure": 24,\n                "MonthlyCharges": 65.0,\n                "TotalCharges": 1500.0\n            }\n            \n            response = requests.post(\n                f"{self.base_url}/predict",\n                json=payload,\n                headers=headers,\n                timeout=10\n            )\n            \n            if response.status_code == 200:\n                prediction_data = response.json()\n                print(f"   ✅ Single prediction: {prediction_data.get('churn_risk', 'unknown')}")\n                self.test_results.append({\n                    'test': 'Single Prediction',\n                    'status': 'PASS',\n                    'prediction': prediction_data.get('churn_risk'),\n                    'probability': prediction_data.get('churn_probability')\n                })\n            else:\n                print(f"   ❌ Single prediction failed: {response.status_code}")\n                \n        except Exception as e:\n            print(f"   ❌ Prediction endpoint error: {e}")\n    \n    def _test_metrics_endpoint(self):\n        """Test metrics endpoint"""\n        try:\n            headers = {"Authorization": f"Bearer {self.auth_token}"}\n            response = requests.get(\n                f"{self.base_url}/metrics",\n                headers=headers,\n                timeout=10\n            )\n            \n            if response.status_code == 200:\n                metrics_data = response.json()\n                print(f"   ✅ Metrics endpoint: {len(metrics_data)} metrics retrieved")\n                self.test_results.append({\n                    'test': 'Metrics Endpoint',\n                    'status': 'PASS',\n                    'metrics_count': len(metrics_data)\n                })\n            else:\n                print(f"   ❌ Metrics endpoint failed: {response.status_code}")\n                \n        except Exception as e:\n            print(f"   ❌ Metrics endpoint error: {e}")\n    \n    def _test_batch_pipeline(self):\n        """Test batch prediction pipeline"""\n        print("\n3️⃣ BATCH PIPELINE TESTS")\n        \n        try:\n            from src.batch_pipeline import BatchPredictionPipeline\n            \n            # Initialize pipeline\n            pipeline = BatchPredictionPipeline()\n            \n            if pipeline.initialize_pipeline():\n                print("   ✅ Batch pipeline initialized")\n                \n                # Test with sample data\n                sample_data = pd.DataFrame({\n                    'tenure': [12, 24, 36, 48],\n                    'MonthlyCharges': [50.0, 65.0, 80.0, 45.0],\n                    'TotalCharges': [600.0, 1560.0, 2880.0, 2160.0]\n                })\n                \n                # Save sample data\n                sample_file = project_root / "test_batch_input.csv"\n                sample_data.to_csv(sample_file, index=False)\n                \n                # Process batch\n                result = pipeline.process_csv_file(\n                    str(sample_file),\n                    include_business_impact=True\n                )\n                \n                if result['success']:\n                    print(f"   ✅ Batch processing: {result['records_processed']} records")\n                    self.test_results.append({\n                        'test': 'Batch Processing',\n                        'status': 'PASS',\n                        'records_processed': result['records_processed'],\n                        'processing_time': result['processing_time_seconds']\n                    })\n                else:\n                    print(f"   ❌ Batch processing failed: {result.get('error')}")\n                \n                # Cleanup\n                if sample_file.exists():\n                    sample_file.unlink()\n            else:\n                print("   ❌ Batch pipeline initialization failed")\n                \n        except Exception as e:\n            print(f"   ❌ Batch pipeline test failed: {e}")\n    \n    def _test_monitoring_system(self):\n        """Test monitoring system"""\n        print("\n4️⃣ MONITORING SYSTEM TESTS")\n        \n        try:\n            from src.advanced_monitoring import AdvancedModelMonitor\n            \n            # Initialize monitor\n            monitor = AdvancedModelMonitor()\n            \n            # Test with sample data\n            sample_features = pd.DataFrame({\n                'tenure': np.random.normal(32, 24, 100),\n                'MonthlyCharges': np.random.normal(65, 30, 100),\n                'TotalCharges': np.random.normal(2000, 2000, 100)\n            })\n            \n            sample_predictions = np.random.beta(2, 5, 100)\n            \n            # Monitor batch\n            monitoring_result = monitor.monitor_prediction_batch(\n                features=sample_features,\n                predictions=sample_predictions\n            )\n            \n            if 'error' not in monitoring_result:\n                print(f"   ✅ Monitoring system: {len(monitoring_result.get('alerts', []))} alerts generated")\n                print(f"   ✅ Drift detection: {'Enabled' if monitoring_result.get('drift_detected') is not None else 'Disabled'}")\n                \n                self.test_results.append({\n                    'test': 'Monitoring System',\n                    'status': 'PASS',\n                    'alerts_generated': len(monitoring_result.get('alerts', [])),\n                    'drift_checked': monitoring_result.get('drift_detected') is not None\n                })\n            else:\n                print(f"   ❌ Monitoring system failed: {monitoring_result['error']}")\n                \n        except Exception as e:\n            print(f"   ❌ Monitoring system test failed: {e}")\n    \n    def _test_end_to_end_workflow(self):\n        """Test complete end-to-end workflow"""\n        print("\n5️⃣ END-TO-END INTEGRATION TESTS")\n        \n        try:\n            # Create test scenario\n            test_customers = [\n                {"tenure": 2, "MonthlyCharges": 85.0, "TotalCharges": 170.0, "expected": "HIGH"},\n                {"tenure": 48, "MonthlyCharges": 35.0, "TotalCharges": 1680.0, "expected": "LOW"},\n                {"tenure": 12, "MonthlyCharges": 75.0, "TotalCharges": 900.0, "expected": "MEDIUM"}\n            ]\n            \n            successful_predictions = 0\n            \n            for i, customer in enumerate(test_customers):\n                try:\n                    headers = {"Authorization": f"Bearer {self.auth_token}"}\n                    payload = {\n                        "tenure": customer["tenure"],\n                        "MonthlyCharges": customer["MonthlyCharges"],\n                        "TotalCharges": customer["TotalCharges"]\n                    }\n                    \n                    response = requests.post(\n                        f"{self.base_url}/predict",\n                        json=payload,\n                        headers=headers,\n                        timeout=10\n                    )\n                    \n                    if response.status_code == 200:\n                        prediction = response.json()\n                        predicted_risk = prediction.get('churn_risk')\n                        print(f"   ✅ E2E Test {i+1}: {predicted_risk} risk (expected: {customer['expected']})")\n                        successful_predictions += 1\n                    else:\n                        print(f"   ❌ E2E Test {i+1}: API error {response.status_code}")\n                        \n                except Exception as e:\n                    print(f"   ❌ E2E Test {i+1}: {e}")\n            \n            if successful_predictions == len(test_customers):\n                print(f"   🎉 End-to-end workflow: All {successful_predictions} tests passed")\n                self.test_results.append({\n                    'test': 'End-to-End Workflow',\n                    'status': 'PASS',\n                    'successful_predictions': successful_predictions,\n                    'total_tests': len(test_customers)\n                })\n            else:\n                print(f"   ⚠️ End-to-end workflow: {successful_predictions}/{len(test_customers)} tests passed")\n                \n        except Exception as e:\n            print(f"   ❌ End-to-end test failed: {e}")\n    \n    def _generate_test_report(self):\n        """Generate comprehensive test report"""\n        print("\n📊 TEST RESULTS SUMMARY")\n        print("=" * 60)\n        \n        total_tests = len(self.test_results)\n        passed_tests = len([t for t in self.test_results if t.get('status') == 'PASS'])\n        failed_tests = total_tests - passed_tests\n        \n        print(f"Total Tests: {total_tests}")\n        print(f"Passed: {passed_tests} ✅")\n        print(f"Failed: {failed_tests} ❌")\n        print(f"Success Rate: {passed_tests/total_tests*100:.1f}%")\n        \n        # Save detailed report\n        report_file = project_root / "test_report.json"\n        with open(report_file, 'w') as f:\n            json.dump({\n                'timestamp': datetime.now().isoformat(),\n                'summary': {\n                    'total_tests': total_tests,\n                    'passed': passed_tests,\n                    'failed': failed_tests,\n                    'success_rate': passed_tests/total_tests if total_tests > 0 else 0\n                },\n                'detailed_results': self.test_results\n            }, f, indent=2)\n        \n        print(f"\\n💾 Detailed report saved: {report_file}")\n        \n        if passed_tests == total_tests:\n            print("\\n🎉 ALL TESTS PASSED - SYSTEM READY FOR PRODUCTION!")\n        else:\n            print(f"\\n⚠️ {failed_tests} TESTS FAILED - REVIEW REQUIRED")\n    \n    def _cleanup(self):\n        """Cleanup test resources"""\n        try:\n            if self.api_process:\n                self.api_process.terminate()\n                self.api_process.wait(timeout=5)\n                print("\\n🧹 API server stopped")\n        except Exception as e:\n            print(f"\\n⚠️ Cleanup warning: {e}")\n\ndef main():\n    \"\"\"Run production test suite\"\"\"\n    test_suite = ProductionTestSuite()\n    \n    def signal_handler(signum, frame):\n        print("\\n\\n🛑 Test interrupted - cleaning up...")\n        test_suite._cleanup()\n        sys.exit(1)\n    \n    signal.signal(signal.SIGINT, signal_handler)\n    \n    success = test_suite.run_all_tests()\n    return 0 if success else 1\n\nif __name__ == "__main__":\n    sys.exit(main())